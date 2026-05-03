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

**A. `__init__`** — Stores `_expert_cache_size` from config. Does NOT move weights to CPU here — that would strip the `weight_loader` attribute that vLLM's weight loading system needs.

**B. `_maybe_init_expert_cache`** — Called after all weights are loaded and post-processed (triggered from `gpu_model_runner.py`). Moves expert weights from GPU to CPU pinned RAM, then creates the `ExpertCache` and warms it. By this point weights are in the format the kernel expects.

**C. `_forward_with_expert_cache`** — The forward pass with caching. Described in the workflow below.

### 5. `gpu_model_runner.py` (MODIFIED) — Expert Cache Init Hook

Added an explicit expert cache initialization loop after `prepare_communication_buffer_for_model` in `load_model()`. This iterates over all `FusedMoE` modules and calls `_maybe_init_expert_cache()`. Necessary because the normal init path (`maybe_init_modular_kernel`) only runs via EP communicators, which are absent on single-GPU setups. This runs after weights are loaded but before the profile/compile run.

---

## Complete Workflow

### Setup: OLMoE-1B-7B, BF16, cache_size=8

64 experts total per layer, 8 cached on GPU, 56 on CPU only.

### Model Loading

```
For each MoE layer:
  1. create_weights allocates w13[64,...] and w2[64,...] on GPU (empty)
  2. Weight loader fills them from checkpoint (on GPU, weight_loader intact)
  3. After all weights loaded, gpu_model_runner calls _maybe_init_expert_cache:
     a. Moves w13 and w2 from GPU to CPU pinned RAM
     b. ExpertCache created: allocates cache_w13[8,...] and cache_w2[8,...]
        on GPU, copies experts 0-7 into them (warm cache)
```

After loading:
```
CPU pinned RAM: all 64 experts per layer (source of truth)
GPU: cache with experts 0-7 per layer + dense weights + KV cache
```

### Inference — Forward Pass

A batch arrives. Router picks experts {1, 5, 7} for this batch.

```
_forward_with_expert_cache(hidden_states, router_logits):

  1. ROUTE: router.select_experts → topk_ids = [[1],[5],[7]]

  2. UNIQUE: needed = [1, 5, 7]

  3. ENSURE LOADED: expert_cache.ensure_experts_loaded([1, 5, 7])
       expert 1: in cache at slot 1  → HIT
       expert 5: in cache at slot 5  → HIT
       expert 7: in cache at slot 7  → HIT
       (all in warm cache, no DMA needed)

  4. REMAP IDs: topk_ids pointed to global expert IDs.
       The kernel needs to index into the cache buffers, so remap
       to slot indices: expert 1→slot 1, expert 5→slot 5, expert 7→slot 7
       remapped_ids = [[1],[5],[7]]

  5. SWAP: temporarily replace layer attributes:
       w13_weight.data = cache_w13 (the full [8,...] GPU buffer)
       w2_weight.data  = cache_w2
       global_num_experts = 8  (cache_size, not 64)
       expert_map = None

  6. KERNEL: quant_method.apply runs the MoE kernel.
       It reads w13_weight and w2_weight — which now point at the
       GPU cache buffers. It uses remapped_ids to index into them.
       All reads are from GPU memory. Zero PCIe traffic for cache hits.

  7. RESTORE: put original CPU weights and attributes back (in a
       finally block, so restoration happens even if the kernel errors).

  8. LRU UPDATE: mark experts 1, 5, 7 as most recently used.
```

Later batch uses experts {1, 42, 63}:
```
  expert 1:  in cache → HIT
  expert 42: not in cache → MISS → evict expert 0 (coldest, LRU),
             DMA expert 42 into slot 0
  expert 63: not in cache → MISS → evict expert 2 (next coldest),
             DMA expert 63 into slot 2

  remapped_ids: expert 1→slot 1, expert 42→slot 0, expert 63→slot 2
```

### Why the Swap Works

The MoE kernel reads weights from `layer.w13_weight` and uses `layer.global_num_experts` to know how many experts exist. By swapping these to point at the cache buffers and setting `global_num_experts = cache_size`, the kernel sees only the cached experts. The remapped `topk_ids` (slot indices instead of global IDs) index correctly into the cache. The kernel doesn't know or care that the weights are replicas from a cache.

---

## Benchmarks (OLMoE-1B-7B, 5 prompts, 100 in / 100 out tokens, enforce-eager)

| Config | TPOT (ms) | Output tok/s | TTFT (ms) |
|---|---|---|---|
| No offloading | 8.14 | 119.82 | 28.47 |
| Expert cache 64 (all cached, no eviction) | 13.15 | 73.74 | 53.72 |
| Expert cache 12 | 66.24 | 14.38 | 397.77 |
| UVA offload all experts | 66.34 | 14.36 | 396.28 |

**Observations:**
- Cache-64 vs no offloading: ~60% overhead from Python-level remapping loop and attribute swapping, even with zero cache misses.
- Cache-12 ≈ UVA offload: with only 12/64 experts cached and random prompts, nearly every step triggers cache misses and CPU→GPU DMA, bottlenecking on PCIe like UVA.
- The LRU cache value shows on workloads with temporal locality (same experts reused across consecutive tokens/batches). Random prompts don't exhibit that.

## Current Limitations

- **Requires `--enforce-eager`.** The expert cache forward path uses data-dependent ops (`unique().tolist()`) and attribute swapping that are incompatible with torch.compile/dynamo tracing.
- **Prefill can exceed cache capacity.** During prefill, many tokens are processed at once (e.g., 100 tokens × top_k=8), activating most or all experts simultaneously. If the number of unique experts needed in a single batch exceeds `expert_cache_size`, the cache cannot hold them all and the engine crashes (`StopIteration`). **Workaround:** use `--max-num-batched-tokens N` where N is small enough that the unique experts per batch stay within `cache_size`. Setting `--max-num-batched-tokens 1` guarantees at most top_k experts per step (always fits), but makes prefill very slow. A proper fix would chunk the prefill tokens into sub-batches grouped by expert affinity.
- **Unquantized models only.** Quantized models (FP8, FP4, GPTQ) have per-expert scale tensors that are not cached or remapped. The kernel would read wrong scales, producing silently incorrect results.
- **No shared experts.** Models like DeepSeek-V3 have "shared experts" that run on every token. The cache path skips them entirely (`shared_experts_input=None`).
- **Single GPU or tensor-parallel only.** No expert parallelism support.
- **Remapping overhead.** The per-expert `for` loop doing `remapped_ids[topk_ids == global_id]` is O(num_unique_experts) full tensor scans. Could be replaced with a GPU-side lookup tensor.

## CLI Usage

```bash
python -m vllm.entrypoints.openai.api_server \
  --model allenai/OLMoE-1B-7B-0924 \
  --expert-offload \
  --expert-cache-size 12 \
  --enforce-eager \
  --max-num-batched-tokens 1
```
