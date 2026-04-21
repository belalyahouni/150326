# Unified Per-Layer Page Pool — Implementation

> Phase 1, "Design D." Companion plan: `unified-pool-mvp-plan.md`. Baseline this extends:
> `expert-cache-implementation.md` (the existing `--expert-offload` flag).

## What it is

A new flag `--expert-unified-pool` that lets KV cache and MoE expert cache
share one page pool on GPU. Cold experts and cold KV blocks compete in a
single LRU; whichever is colder is evicted first. Requires
`--expert-offload --enable-prefix-caching` and a uniprocess executor
(`tensor_parallel_size == 1`, `pipeline_parallel_size == 1`).

The scheduler's existing `BlockPool` **is** the LRU — it already maintains a
doubly linked free list keyed by recency (HEAD = LRU, TAIL = MRU). Expert
pages are made to look like free KV blocks in that same queue. When an MoE
layer uses an expert, the worker pins the page (`touch`). When it's done, it
unpins (`free_blocks`), which appends the page at TAIL. The scheduler's
`get_new_blocks` pops HEAD and may pop an expert page; an on-alloc callback
tells the worker to forget the expert mapping first.

No second LRU, no separate eviction policy. Both swap directions — evict
expert for KV, evict KV for expert — become the same code path.

### Design D: static staging (simulation-fidelity scaffolding)

The Triton `fused_moe_kernel` is not modified in this phase. To avoid
touching kernel internals, each MoE layer keeps a **static** GPU tensor
`staging_w13[num_experts, ...]` + `staging_w2[num_experts, ...]`, filled
once from CPU at startup and never rewritten. Every forward step swaps
`w13_weight.data = staging_w13` so the unmodified kernel's native
`b_ptr + expert_id * stride_be` indexing is correct without any
`topk_ids` remap.

Real CPU→GPU DMA still happens on pool misses (to keep miss latency
measurable), but the kernel never reads from pool pages. A
`wait_stream(transfer_stream)` barrier in `ensure_experts_loaded` forces
the kernel's start time to equal `max(DMA completion)`, matching what
Phase 2 (direct pool-page reads) would pay.

The staging tensors are excluded from the simulated GPU budget — they are
allocated and filled **before** `determine_available_memory` runs, so the
memory profile sees them as already-consumed. The scheduler then gets the
same dynamic budget the baseline gets.

---

## Files changed

| File | Role |
|---|---|
| `vllm/config/offload.py` | New `expert_unified_pool: bool` field on `OffloadConfig` + validator (requires `expert_offload=True`). |
| `vllm/engine/arg_utils.py` | New `--expert-unified-pool` CLI flag, dataclass attr `expert_unified_pool`, and `OffloadConfig(...)` constructor argument. |
| `vllm/v1/core/block_pool.py` | `_on_alloc_callbacks: list` + `register_on_alloc_callback(cb)`. Callbacks fire at the end of `get_new_blocks` for each returned block, after prefix-cache eviction and ref-count increment. |
| `vllm/model_executor/layers/fused_moe/unified_pool.py` | **New file.** `PerLayerPool` dataclass and `UnifiedPagePoolManager` class. |
| `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE.__init__` reserves `_unified_pool = None` and records `_unified_pool_enabled`. `move_experts_to_cpu()` helper extracted from `_maybe_init_expert_cache` (which now early-exits when unified pool is on). `forward_native` dispatches to `_forward_with_unified_pool` first. New `_forward_with_unified_pool` method. |
| `vllm/v1/worker/gpu_model_runner.py` | Hook at model-load time splits into unified-pool vs. old expert-cache branches. New `_init_unified_pool_metadata` (runs inside `load_model`, before memory profile: moves experts to CPU, validates uniform shapes, computes `block_size_tokens`, mutates `cache_config.block_size`, pre-allocates staging). New `_setup_unified_pool_manager(block_pool)` (builds the pool manager, aliases the raw KV tensor as `pool_buffer`, registers the callback, warms the pool). Saves `self._kv_cache_raw_tensors` so the raw byte view is reachable later. |
| `vllm/v1/worker/gpu_worker.py` | Thin passthrough methods `setup_unified_pool(block_pool)` and `get_unified_pool_block_count()` delegating to the model runner. |
| `vllm/v1/engine/core.py` | After `self.scheduler = Scheduler(...)`, if `expert_unified_pool` is set: assert the executor is `UniProcExecutor`, then `collective_rpc("setup_unified_pool", args=(self.scheduler.kv_cache_manager.block_pool,))`. |
| `scripts/sync_to_venv.sh` | New helper: rsyncs modified `vllm/vllm/**/*.py` into `venv/lib/python3.12/site-packages/vllm/` so the venv-installed server picks up edits without reinstall. |

No changes to `kv_cache_utils.py`, `fused_moe_kernel`, attention backends,
or the prefix-cache hash logic.

---

## Key data structures

```python
@dataclass
class PerLayerPool:
    layer_idx:     int
    pool_buffer:   Tensor       # int8, aliased to the attention layer's raw KV tensor
    n_pages:       int
    page_bytes:    int          # expert_slot_bytes, rounded up to 16*bytes_per_token_per_layer
    w13_shape, w2_shape, expert_dtype, w13_bytes, w2_bytes
    cpu_w13, cpu_w2:       Tensor  # [num_experts, ...], pinned
    staging_w13, staging_w2: Tensor  # [num_experts, ...], GPU, static
    expert_to_page: dict[int, int]  # eid -> block_id currently holding its bytes
    page_to_expert: dict[int, int]
    hits, misses:  int
    slot_w13_view(page)  -> Tensor  # view into pool_buffer[page*page_bytes + 0 : + w13_bytes]
    slot_w2_view(page)   -> Tensor
    invalidate_page(page)           # clear expert_to_page / page_to_expert for this page
```

```python
class UnifiedPagePoolManager:
    pools:                  dict[int, PerLayerPool]   # keyed by layer_idx
    block_pool:             BlockPool                  # scheduler's, shared reference
    transfer_stream:        cuda.Stream
    expert_layers_at_page:  dict[int, set[int]]        # block_id -> layers that mapped an expert there
    _pending_release:       dict[int, list[int]]       # layer_idx -> pool pages pinned this step

    on_block_allocated(block_id)        # callback, fired by BlockPool.get_new_blocks
    warm_pool_all_layers(n_per_layer)   # one-shot startup residency seeding
    ensure_experts_loaded(layer_idx, needed_ids)
    release_after_forward(layer_idx)
    log_stats() / maybe_log_stats(every=100)
```

One `PerLayerPool` per MoE layer; one shared `BlockPool` (scheduler's) for
all layers and all KV.

---

## Initialization flow

Order matters because `block_size` determines `num_blocks` which determines
the pool size. The worker and EngineCore share one `vllm_config` object
under `UniProcExecutor`, so mutations on either side are visible to the
other later in init.

1. **`init_device`, `load_model`** — unchanged. Expert weights land on GPU.

2. **Inside `load_model`, after `prepare_communication_buffer_for_model`**
   (`gpu_model_runner.py` around line 4590), when `expert_offload=True`:
   - If `expert_unified_pool=True` → `self._init_unified_pool_metadata()`.
   - Else → the old `module._maybe_init_expert_cache()` loop.

3. **`_init_unified_pool_metadata`**:
   - Iterate `FusedMoE` modules, call `module.move_experts_to_cpu()` (shared helper that moves `w13_weight` / `w2_weight` to CPU pinned RAM).
   - Validate every layer has identical `w13_shape`, `w2_shape`, `expert_dtype`, `num_experts`.
   - Compute `expert_slot_bytes = w13_bytes + w2_bytes`.
   - Compute `bytes_per_token_per_layer = num_kv_heads * head_size * 2 * kv_dtype_size` using `self.model_config.get_num_kv_heads(self.parallel_config)`, `get_head_size()`, and `self.kv_cache_dtype`.
   - Round `expert_slot_bytes` up to `alignment = 16 * bytes_per_token_per_layer` (so the derived `block_size_tokens` is divisible by the attention-backend kernel block size of 16).
   - `block_size_tokens = padded_page_bytes // bytes_per_token_per_layer` and assert it is divisible by 16.
   - Mutate `self.vllm_config.cache_config.block_size = block_size_tokens`.
   - Allocate per-layer staging on GPU (`torch.empty((num_experts, *w13_shape), dtype=expert_dtype, device=...)` + w2 analogue) and copy all experts from CPU. This happens **before** `determine_available_memory`, so staging is already-used memory for the profiler.
   - Stash everything on `self._unified_pool_metadata`.

4. **Memory profile** (`determine_available_memory`) runs unchanged and reports `(free_memory - staging)` as the available pool budget.

5. **EngineCore computes `num_blocks`** via `get_kv_cache_configs`. Because `page_size_bytes == padded_page_bytes == expert_slot_bytes (rounded)`, we get `num_blocks = budget // (num_layers * page_bytes)`.

6. **Worker `initialize_kv_cache_tensors`** allocates the raw int8 buffers per layer via `_allocate_kv_cache_tensors`. Added step: if `expert_unified_pool` is active, save the `kv_cache_raw_tensors` dict on `self._kv_cache_raw_tensors` before the local reference falls out of scope. `_reshape_kv_cache_tensors` then views it as the attention shape — no second allocation.

7. **EngineCore constructs `self.scheduler = Scheduler(...)`** → `BlockPool(num_gpu_blocks=N_pages, …)` exists.

8. **New hop in `EngineCore.__init__`**: if `expert_unified_pool`, assert `isinstance(self.model_executor, UniProcExecutor)`, then
   ```python
   block_pool = self.scheduler.kv_cache_manager.block_pool
   self.model_executor.collective_rpc("setup_unified_pool", args=(block_pool,))
   ```

9. **Worker `setup_unified_pool`** → `GPUModelRunner._setup_unified_pool_manager(block_pool)`:
   - For each MoE layer, find its `layer_idx` via `extract_layer_index(module.layer_name)`.
   - Look up the attention layer's raw int8 tensor in `self._kv_cache_raw_tensors` by matching `extract_layer_index(layer_name)` → `layer_idx`.
   - Build `PerLayerPool` with `pool_buffer=raw_tensor` (aliased, no copy).
   - Build `UnifiedPagePoolManager(pools, block_pool, device)`.
   - `block_pool.register_on_alloc_callback(manager.on_block_allocated)`.
   - Set `module._unified_pool = manager` on every `FusedMoE`.
   - Check `expert_cache_size * num_moe_layers <= num_gpu_blocks - 1` (`-1` for the null block); reduce and warn if needed.
   - Call `manager.warm_pool_all_layers(expert_cache_size)` to seed initial residency. Warm-up pages are freed back (ref_cnt → 0, land at TAIL) so they're available for KV or hot experts — they are not pinned permanently.

10. Scheduler starts serving requests.

### Simulation-fidelity accounting

Both the staging overhead (`num_experts × (w13+w2) × num_moe_layers`) and
the chosen `block_size_tokens` are logged at startup:

```
Unified pool: page_bytes=12582912 (w13=8388608 w2=4194304 pad=0)
  block_size_tokens=3072 num_moe_layers=16 num_experts=64
  staging_overhead=12.00 GiB
```

This exists so reviewers can verify baseline vs. unified runs operate on
equal dynamic pool budget even though staging inflates the unified run's
absolute footprint.

---

## Per-forward flow (one MoE layer L)

Driven by `FusedMoE._forward_with_unified_pool`:

```python
topk_weights, topk_ids = router.select_experts(...)
needed = topk_ids.unique().tolist()            # e.g. 8 experts for OLMoE

manager.ensure_experts_loaded(L, needed)        # see below
layer_pool = manager.pools[L]

orig_w13 = w13_weight.data
orig_w2  = w2_weight.data
try:
    w13_weight.data = layer_pool.staging_w13    # full [num_experts, ...], static
    w2_weight.data  = layer_pool.staging_w2
    result = quant_method.apply(                # unmodified Triton kernel
        layer=self, x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,                      # NOT remapped
        shared_experts_input=None,
    )
finally:
    w13_weight.data = orig_w13
    w2_weight.data  = orig_w2

manager.release_after_forward(L)
manager.maybe_log_stats()
```

`global_num_experts` is **not** mutated, `topk_ids` is **not** remapped —
staging is full-width, so the kernel's native indexing already works. This
is the main simplification Design D buys over the `_forward_with_expert_cache`
path, which has to compact and remap because its `cache_w13` tensor is
`[cache_size, ...]`.

### Inside `ensure_experts_loaded(L, needed_ids)`

1. Classify each `eid` in `needed_ids`:
   - **Hit**: `eid in layer.expert_to_page`. Append the existing `KVCacheBlock` to `hit_blocks`. `layer.hits += 1`.
   - **Miss**: `layer.misses += 1`. Queue in `miss_ids`.

2. If any hits: `block_pool.touch(hit_blocks)` — atomically removes each from the free queue (if `ref_cnt==0`) and bumps `ref_cnt += 1`. After this, hits cannot be evicted by our own miss allocations in the next step.

3. If any misses: `new_blocks = block_pool.get_new_blocks(len(miss_ids))`. This:
   - Pops HEAD × N from the free queue (coldest first).
   - Evicts each block from the prefix-cache hash (`_maybe_evict_cached_block`).
   - Sets `ref_cnt = 1` on each returned block.
   - Fires `_on_alloc_callbacks` → `manager.on_block_allocated(block_id)` → clears any stale `expert_to_page` / `page_to_expert` in any layer that had mapped an expert at that page. This is the cross-layer collateral invalidation.

4. DMA on `transfer_stream`:
   ```python
   with torch.cuda.stream(transfer_stream):
       for eid, block in zip(miss_ids, new_blocks):
           layer.slot_w13_view(block.block_id).copy_(layer.cpu_w13[eid], non_blocking=True)
           layer.slot_w2_view(block.block_id).copy_(layer.cpu_w2[eid], non_blocking=True)
           # update expert_to_page / page_to_expert / expert_layers_at_page
   torch.cuda.current_stream(device).wait_stream(transfer_stream)
   ```
   `slot_w13_view(page)` = `pool_buffer[page*page_bytes : page*page_bytes + w13_bytes].view(expert_dtype).view(w13_shape)` — a typed view into the shared int8 buffer.

5. Record `_pending_release[L] = [all pinned pages (hits + misses)]`.

The `wait_stream` at the end is the simulation-fidelity barrier. Without it,
the unmodified kernel would launch early (it reads staging, which is already
ready) and miss-heavy forwards would be artificially fast. With it, the
GEMM is ordered behind the DMAs, matching Phase 2 behavior.

### Inside `release_after_forward(L)`

```python
pages = self._pending_release.pop(L, [])
blocks = [block_pool.blocks[p] for p in pages]
block_pool.free_blocks(blocks)    # ref_cnt -= 1; ref_cnt==0 → append at TAIL
```

No `cuda.synchronize()`. The kernel reads `staging_w13/w2`, never
`pool_buffer`, so freeing a pool page right after the kernel returns
cannot cause a read-after-free. The DMA→kernel ordering from
`wait_stream` in step 5 of `ensure_experts_loaded` also incidentally
orders any compute-stream op in step N+1 after the step-N DMAs, so no
cross-step race either.

### End-of-step invariant

After every MoE layer's forward, every page used this step is at TAIL of
the free queue with `ref_cnt=0`. Pages untouched this step drift left
toward HEAD. The next `get_new_blocks` call anywhere in the system — by
the scheduler for a new KV block, or by a later MoE layer for a miss —
pops the globally coldest page, regardless of whether it was holding KV
data or expert bytes.

---

## The two swap directions

Both are the same code path:

**Expert → KV** (Scenario A: shared long prefix, KV-hot). Scheduler keeps
issuing `get_new_blocks` for new KV chunks. Over many steps, pages holding
cold experts drift to HEAD and get popped. Callback clears their expert
mapping; scheduler writes KV into them. The affected MoE layer takes a
miss on its next use of that expert.

**KV → expert** (Scenario D: diverse inputs, expert-hot). Router picks a
non-resident expert. `manager.ensure_experts_loaded` calls
`get_new_blocks(1)` → pops HEAD → that page may hold cold KV. If the KV
block had a prefix-cache hash, `_maybe_evict_cached_block` clears it. DMA
writes the expert bytes. The KV block is gone; a future request that
needed that prefix will re-prefill (the normal prefix-cache-miss cost).

Neither direction requires a second LRU or any workload detection.

---

## Memory layout

Per MoE layer (OLMoE-1B-7B-0924 numbers, BF16, 64 experts, 16 layers):

```
pool_buffer  = [int8 * n_pages * page_bytes]        aliased to the attn layer's raw KV tensor
page_bytes   = ~12.6 MB   (w13: 2*1024*2048*2 = 8.4 MB; w2: 2048*1024*2 = 4.2 MB; aligned up)
block_size_tokens = ~3200 (= page_bytes / (16_kv_heads * 64_head_dim * 2 * 2))
staging_w13  = [num_experts=64, w13_shape...]  GPU, static, ~576 MB per layer
staging_w2   = [num_experts=64, w2_shape...]   GPU, static, ~288 MB per layer
cpu_w13, cpu_w2  = [num_experts=64, ...]       CPU pinned, source of truth
```

One page in the pool holds either a whole expert (`w13 | w2` at byte
offsets `[0, w13_bytes)` and `[w13_bytes, w13_bytes + w2_bytes)`) or
`block_size_tokens` worth of KV (the usual attention layout). Pages rotate
between the two uses purely by who calls `get_new_blocks` next.

Staging overhead (all 16 layers): ~13.8 GB for OLMoE. Prohibitive at real
scale — this is the Phase 1 scaffolding tax. Phase 2 eliminates it by
modifying `fused_moe_kernel` to read from pool pages directly.

---

## Instrumentation

`UnifiedPagePoolManager.log_stats()` prints per-manager aggregates:

```
UnifiedPool stats: hits=12847 misses=391 hit_rate=97.0% | HEAD(32): kv=21 expert=8 empty=3
```

`HEAD(N)` is the composition of the first `N` free-queue entries — the
next eviction victims. In Scenario A the HEAD should skew KV-dominated
(cold KV blocks queued for eviction when experts are hotter); in Scenario
D it should skew expert-dominated. This is the core dissertation evidence
that dynamic rebalancing is happening.

`maybe_log_stats(every=100)` is called per-layer-forward, so with 16
layers it logs every ~6 steps; adjust `every` as needed.

---

## CLI usage

```bash
venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model allenai/OLMoE-1B-7B-0924 \
    --expert-offload \
    --expert-unified-pool \
    --expert-cache-size 12 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.40 \
    --max-model-len 131072
```

Validation enforces:
- `--expert-unified-pool` requires `--expert-offload` (else `ValidationError` at config time).
- The EngineCore asserts `UniProcExecutor` at startup (fails fast if TP>1 or PP>1 was passed).
- Prefix caching must be on for the KV eviction side to have anything interesting to evict; the unified pool does not enforce this at config time, but Scenario A will be meaningless without it.

---

## What is NOT handled (Phase 2+)

- **Modifying `fused_moe_kernel`** to read from pool pages directly via a page-table argument. Eliminates staging entirely.
- **Per-layer expert eviction**: today a KV eviction of page P invalidates the expert at page P *in every layer* that held one there. Fine for MVP; a true per-layer system would require separate page pools per layer.
- **TP > 1, PP > 1, multi-process executors.** `BlockPool` is not picklable across process boundaries; the RPC would need a different shape.
- **Quantized experts** (FP8, INT8, AWQ): scales tensors aren't cached or remapped.
- **Hybrid attention, Mamba, MLA, sliding window.** Not refused at config time yet — caveat emptor; the block-size derivation assumes a simple attention spec.
- **OOM fallback.** If `get_new_blocks(1)` raises `ValueError("Cannot get … free blocks")`, we re-raise. In single-concurrency benchmarks this should not happen; if it does in production it's a signal that the page budget is too small for `max_num_seqs × max_needed_experts`.

---

## Verification

Steps run so far (see `scripts/sync_to_venv.sh` for the venv sync helper):

1. ✅ `from vllm.model_executor.layers.fused_moe.unified_pool import UnifiedPagePoolManager, PerLayerPool` — imports clean.
2. ✅ `OffloadConfig(expert_offload=True, expert_unified_pool=True)` constructs; `OffloadConfig(expert_offload=False, expert_unified_pool=True)` raises `ValidationError`.
3. ✅ `BlockPool.register_on_alloc_callback` fires exactly once per block returned by `get_new_blocks`.
4. ✅ CLI registration: `--expert-unified-pool` is present in `EngineArgs.add_cli_args`.
5. ✅ End-to-end import chain (`EngineCore`, `Worker`, `GPUModelRunner`, `FusedMoE`) loads; all new methods (`setup_unified_pool`, `_init_unified_pool_metadata`, `_setup_unified_pool_manager`, `_forward_with_unified_pool`, `move_experts_to_cpu`) are present.

Still to run against a GPU (per `unified-pool-mvp-plan.md` §12):

- Server boot on OLMoE with the flags above. Confirm the logged
  `Unified pool: page_bytes=… block_size_tokens=… staging_overhead=…` line.
- Correctness smoke vs. an `--expert-offload`-only baseline.
- Scenario A (KV-hot) targeting TTFT ~A-12 even when started with
  `--expert-cache-size 64`.
- Scenario D (expert-hot) targeting TPOT ~D-64 even when started with
  `--expert-cache-size 12`.
- Per-100-step HEAD composition log flipping KV↔expert between scenarios.
