# Expert Cache Implementation

## What It Is

An LRU cache that keeps a small number of MoE expert weight replicas on GPU while all experts live permanently on CPU pinned RAM. Cache hits read from GPU memory (~3 TB/s). Cache misses copy one expert from CPU to GPU over PCIe (~32 GB/s), evicting the least-recently-used expert.

## Files Changed

### 1. `expert_cache.py` (NEW) — The Cache Object

`ExpertCache` holds:

- `cache_w13[cache_size, ...]`, `cache_w2[cache_size, ...]` — pre-allocated GPU buffers. Fixed number of "slots" that hold expert weight replicas.
- `cpu_w13[num_experts, ...]`, `cpu_w2[num_experts, ...]` — references to the full expert weights on CPU pinned RAM. Never modified. Source of truth.
- `expert_to_slot: dict[int, int]` — maps expert ID to its GPU slot (e.g. expert 47 → slot 3).
- `lru_order: OrderedDict` — tracks usage order. Oldest entry = next to evict. All operations are O(1).
- `transfer_stream` — dedicated CUDA stream so DMA copies don't block compute.
- `hits`, `misses` — counters for logging.

**Methods:**

| Method | What it does |
|---|---|
| `_warm_cache` | Copies experts 0..cache_size-1 from CPU→GPU at init time |
| `ensure_experts_loaded(ids)` | Two-pass: (1) classify hits/misses, (2) for each miss, evict the coldest expert not needed in this batch, DMA the new expert into its slot. Waits for all transfers before returning. |
| `mark_recently_used(ids)` | Moves experts to the most-recently-used end of the LRU list |

The two-pass eviction is important: it avoids evicting an expert that is itself needed in the same batch.

### 2. `offload.py` (MODIFIED) — Config

Two fields added to `OffloadConfig`:

- `expert_offload: bool` — master switch. When True, expert weights go to CPU and a GPU cache is created.
- `expert_cache_size: int` — how many expert slots to keep on GPU per layer.

### 3. `arg_utils.py` (MODIFIED) — CLI

Wires the config into CLI flags: `--expert-offload` and `--expert-cache-size`.

### 4. `layer.py` (MODIFIED) — FusedMoE Integration

Three changes to the `FusedMoE` class:

**A. `__init__`** — After `create_weights` allocates expert tensors on GPU, moves them to CPU pinned RAM immediately. This frees GPU memory so the next layer can reuse it. Dense weights (attention, layernorm, router) are unaffected and stay on GPU.

**B. `_maybe_init_expert_cache`** — Called after all weights are loaded and post-processed. Creates the `ExpertCache` and warms it. By this point weights are in the format the kernel expects (any shuffling for AITER/FlashInfer has already been applied and the data moved back to CPU).

**C. `_forward_with_expert_cache`** — The forward pass with caching. Described in the workflow below.

---

## Complete Workflow

### Setup: Mixtral 8x7B, BF16, cache_size=4

8 experts total per layer, 4 cached on GPU, 4 on CPU only.

### Model Loading

```
For each MoE layer:
  1. create_weights allocates w13[8,...] and w2[8,...] on GPU (empty)
  2. Our code moves them to CPU pinned RAM, freeing GPU
  3. Weight loader fills them from checkpoint (writes go to CPU)
  4. Post-processing temporarily moves to GPU, transforms, moves back
  5. ExpertCache created: allocates cache_w13[4,...] and cache_w2[4,...]
     on GPU, copies experts 0-3 into them (warm cache)
```

After loading:
```
CPU pinned RAM: all 8 experts per layer (source of truth)
GPU: cache with experts 0,1,2,3 per layer + dense weights + KV cache
```

### Inference — Forward Pass

A batch arrives. Router picks experts {1, 5, 7} for this batch.

```
_forward_with_expert_cache(hidden_states, router_logits):

  1. ROUTE: router.select_experts → topk_ids = [[1],[5],[7]]

  2. UNIQUE: needed = [1, 5, 7]

  3. ENSURE LOADED: expert_cache.ensure_experts_loaded([1, 5, 7])
       expert 1: in cache at slot 1  → HIT
       expert 5: not in cache        → MISS → evict expert 0 (coldest),
                                       DMA expert 5 into slot 0
       expert 7: not in cache        → MISS → evict expert 2 (next coldest),
                                       DMA expert 7 into slot 2
       Wait for DMA to finish.

  4. REMAP IDs: topk_ids pointed to global expert IDs.
       The kernel needs to index into the cache buffers, so remap
       to slot indices: expert 1→slot 1, expert 5→slot 0, expert 7→slot 2
       remapped_ids = [[1],[0],[2]]

  5. SWAP: temporarily replace layer attributes:
       w13_weight.data = cache_w13 (the full [4,...] GPU buffer)
       w2_weight.data  = cache_w2
       global_num_experts = 4  (cache_size, not 8)
       expert_map = None

  6. KERNEL: quant_method.apply runs the MoE kernel.
       It reads w13_weight and w2_weight — which now point at the
       GPU cache buffers. It uses remapped_ids to index into them.
       All reads are from GPU memory. Zero PCIe traffic for cache hits.

  7. RESTORE: put original CPU weights and attributes back (in a
       finally block, so restoration happens even if the kernel errors).

  8. LRU UPDATE: mark experts 1, 5, 7 as most recently used.
```

Cache state after:
```
slot 0: expert 5  (was expert 0, evicted)
slot 1: expert 1  (kept, was a hit)
slot 2: expert 7  (was expert 2, evicted)
slot 3: expert 3  (untouched)

lru_order (oldest → newest): [3, 1, 5, 7]
Next eviction victim: expert 3
```

If the next batch also uses experts {1, 5, 7}: 100% cache hits, zero DMA.

### Why the Swap Works

The MoE kernel reads weights from `layer.w13_weight` and uses `layer.global_num_experts` to know how many experts exist. By swapping these to point at the cache buffers and setting `global_num_experts = cache_size`, the kernel sees only the cached experts. The remapped `topk_ids` (slot indices instead of global IDs) index correctly into the cache. The kernel doesn't know or care that the weights are replicas from a cache.

---

## Current Limitations

- **Unquantized models only.** Quantized models (FP8, FP4, GPTQ) have per-expert scale tensors that are not cached or remapped. The kernel would read wrong scales, producing silently incorrect results.
- **No shared experts.** Models like DeepSeek-V3 have "shared experts" that run on every token. The cache path skips them entirely (`shared_experts_input=None`).
- **Single GPU or tensor-parallel only.** No expert parallelism support.
- **No CUDA graph support.** Cache misses trigger variable-length DMA which can't be captured in a graph.

## CLI Usage

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --expert-offload \
  --expert-cache-size 4
```
