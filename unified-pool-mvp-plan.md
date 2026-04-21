# Unified Per-Layer Page Pool — MVP Implementation Plan

> **For the implementing agent**: This plan assumes you start from `origin/main` (commit `cdf91da64a` or later), which contains the working `--expert-offload` / `--expert-cache-size` implementation. Do **not** start from the `expert-offload` branch — that branch currently only contains a *revert* of the expert-cache changes and has no MoE-offload code to extend.
>
> After reading the plan top-to-bottom, implement on a fresh branch off `origin/main`, sync modified files into the venv (`venv/lib/python3.12/site-packages/vllm/...`) after every significant edit, and verify with the smoke test in §12 before moving on.

---

## 1. Context & Research Goal

The dissertation argues that statically partitioning GPU VRAM between KV cache and expert cache is suboptimal. The unified pool replaces both with a single memory pool and a single LRU; whichever is colder (an idle expert page or a cold KV block) gets evicted when pressure rises.

Target test matrix:

| | High KV reuse | Low KV reuse |
|---|---|---|
| Small expert cache | Baseline OK | Baseline OK |
| Large expert cache | Unified pool wins (evicts cold experts → more KV) | Baseline OK |

Exact benchmark commands live in `motivation-benchmarks.md`. Expected wins:
- **Scenario A** (shared 70K prefix, KV-hot): pool run matches `A-12` TTFT even when started with `--expert-cache-size 64`.
- **Scenario D** (diverse inputs, expert-hot): pool run matches `D-64` TPOT even when started with `--expert-cache-size 12`.

---

## 2. Accepted Suboptimalities (MVP Scope)

We choose the simplest design that exhibits both swap directions. These are the costs:

1. **Page size = one expert slot** (`page_bytes = w13_bytes + w2_bytes`). Concrete sizes:
   - 2(bf16) = 4096 B`, so one page holds `12.6 MB / 4096 ≈ 3200` tokens. Coarse, but acceptable.
2. **Synced KV physical placement across layers**: logical KV block `B` always lives at physical page `B` in every layer's pool OLMoE-1B-7B BF16: hidden=2048, intermediate=1024, top-k=8. `w13 = 2*1024 * 2048 * 2 = 8.39 MB`, `w2 = 2048 * 1024 * 2 = 4.19 MB`, `page_bytes ≈ 12.6 MB`.
   - Mixtral 8x7B BF16: `w13 ≈ 229 MB`, `w2 ≈ 115 MB`, `page_bytes ≈ 344 MB`.
   - OLMoE KV-per-token-per-layer: `num_kv_heads(16) * head_dim(64) * 2(K+V) * buffer. This keeps vLLM's single shared block table valid — zero attention-backend changes. Each layer is still free to place **experts** at different pages (no cross-layer expert coordination).
3. **Collateral expert eviction**: when the scheduler claims page `P` for KV, every layer with an expert at `P` loses that expert. Fine for MVP.
4. **Static full-expert staging (Design D simulation)**: a dense per-layer staging tensor `[num_experts, *w13_shape]` (and w2 analogue) is filled **once at startup** with *all* experts from CPU, and is **never modified again** for the remainder of the run. Every forward hands this static tensor to the unmodified `fused_moe_kernel` as `w13_weight.data` — no per-forward gather, no `topk_ids` remap, no GEMM-related stream sync. The pool still does real CPU→GPU DMA on expert misses (so latency dynamics are measured honestly), but those DMAs land in pool pages that the kernel never reads. This simulates Phase 2's behavior: Phase 2's modified kernel would read directly from pool pages; Design D fakes it by giving the unmodified kernel the full expert table (which contains the same bytes Phase 2 would have gathered). Staging is Phase 1 scaffolding, **excluded from the simulated GPU budget** (see §14). Baseline and unified runs are compared on equal *dynamic* budget so the experimental variable is only "static vs. dynamic pool split".
5. **Uniprocess only** (`UniProcExecutor`). The worker calls `BlockPool` methods directly on a reference handed over from EngineCore. Multi-process / TP > 1 is out of scope.
6. **BF16, MoE-only, single node, single attention layer group** (no hybrid/mamba, no sliding window, no MLA). Same envelope as current `--expert-offload`.
7. **Requires `--enable-prefix-caching`** for the KV LRU to do anything useful — without it, freed KV blocks don't stay in the free queue with their hashes, and Scenario A can't hit.

---

**Terminology note.** Two distinct quantities that were previously conflated:
- `expert_cache_size` = warm-up count for pool expert pages and rough LRU pressure target. Controls how many expert pages are resident in the pool at startup. The actual resident count fluctuates with KV pressure.
- `staging_capacity` = `num_experts` (fixed per-layer). The staging tensor that is actually handed to the GEMM. In Design D it's filled once with all experts and never modified.

**Design D in one sentence.** The pool is the real residency store and drives the LRU; staging is a static full copy of all experts that the unmodified `fused_moe_kernel` reads from. Pool misses still trigger real CPU→GPU DMAs (cost is measured), but the kernel does not read from pool pages — it reads from staging, which already has every expert.

---

## 3. Core Mechanic

vLLM's scheduler already owns a true LRU over blocks: `FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py:158`. HEAD = LRU, TAIL = MRU. `popleft()` evicts. `append()` inserts at TAIL. `remove()` + `append()` = bump to MRU.

**The unified-pool trick**: the worker makes an *expert page* look like any other KV block in that queue. Every time a layer uses an expert, the worker moves that page to TAIL via `BlockPool.bump([P])`. Cold experts drift to HEAD. When the scheduler pops HEAD for a new KV block, it may pop an expert page; a registered callback tells the worker to forget its expert-mapping for that page before the KV write happens.

Both swap directions become the same code path:
- Worker needs a new expert → `BlockPool.get_new_blocks(1)` → HEAD could be a cached KV block → its hash is cleared, expert is DMA'd in.
- Scheduler needs a new KV block → `BlockPool.get_new_blocks(1)` → HEAD could be an expert page → callback clears the expert mapping, KV is written in.

No separate worker-side LRU.

---

## 4. Per-Forward Data Flow (Design D)

For each MoE layer `L` during a forward step:

1. Router picks unique experts `E = {e_1, ...}`.
2. Worker calls `manager.ensure_experts_loaded(L, E)`:
   - For each `e` already in the layer's `expert_to_page`: call `BlockPool.touch([block])` — pins (ref_cnt++) and removes the block from the free queue so the scheduler can't evict it mid-step. Hit counter++.
   - For each miss: call `BlockPool.get_new_blocks(1)` → pops HEAD, fires `on_block_allocated` callback (invalidates any stale per-layer expert mapping at that page), returns block with `ref_cnt=1`. DMA expert weights from CPU-pinned tensors into `pool_buffer[page * page_bytes : ...]` on the dedicated transfer stream. Record `expert_to_page[e] = page`, `page_to_expert[page] = e`, `expert_layers_at_page[page].add(L)`. Miss counter++.
   - `torch.cuda.current_stream().wait_stream(transfer_stream)` — **this is the simulation-fidelity barrier**. In Phase 2 the kernel genuinely depends on the miss DMAs (it reads from pool pages). In Design D the kernel reads static staging and has no real data dependency on those DMAs, so without this wait the GEMM would launch early and a miss-heavy forward would be artificially fast. The `wait_stream` forces the GEMM's start time to be `max(DMA completion times)`, exactly what Phase 2 would pay. Secondary benefit: any later compute-stream op (the next step's KV write into a now-freed pool page) is also ordered after DMA completion, preventing scheduler-race corruption.
   - **No gather.** Staging already has every expert at its canonical row. Return nothing (or just stats).
3. `FusedMoE._forward_with_unified_pool` swaps `w13_weight.data = staging_w13` and `w2_weight.data = staging_w2` (the full static tensors, shape `[num_experts, ...]`), leaves `global_num_experts = num_experts` unchanged, **does not remap** `topk_ids`, runs `quant_method.apply(...)`, then restores the originals in a `finally` block.
4. After the kernel call, **no stream sync is required before releasing pages** — the kernel doesn't read from pool pages, only from staging. Call `manager.release_after_forward(L)` which does `BlockPool.free_blocks([pinned_blocks])` — each block's `ref_cnt` goes back to 0 and lands at TAIL of the free queue. Pages are now evictable again.

### Sync discipline in Design D

Two different sync concerns, easily conflated:

1. **Kernel must wait for DMAs (simulation fidelity).** The kernel doesn't *physically* need the DMAs to complete (it reads staging), but Phase 2's kernel *would* need them (it reads pool pages). We impose the dependency artificially via `wait_stream` inside `ensure_experts_loaded` so measured latency matches Phase 2. **This is mandatory for the benchmark to be meaningful.**
2. **No post-GEMM sync needed before `free_blocks`.** In Design B (per-step gather), freeing a pool page while the kernel was still reading it would corrupt results — that's why the original plan had a `cuda.synchronize()` before release. In Design D the kernel never reads pool pages, so there's no kernel-vs-free race. We can call `release_after_forward` immediately after `quant_method.apply` returns.

The `wait_stream` in (1) also incidentally covers cross-step races: any compute-stream op in step N+1 (a KV write into a page that was a miss DMA destination in step N) is ordered after the step-N DMA completion because the step-N `wait_stream` barrier is still in the compute stream's history.

### Why no `topk_ids` remap is needed

The unmodified `fused_moe_kernel` (`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:315`) indexes `b_ptr + expert_id * stride_be`. With `w13_weight.data = staging_w13[num_experts, ...]` where row `i` is expert `i`, the kernel's native indexing is already correct. Compare with current `_forward_with_expert_cache`: it shrinks `global_num_experts` to `cache_size` and remaps `topk_ids` to slot indices because `cache_w13` is a *compacted* tensor. We don't compact — staging is full-width — so no remap.

---

## 5. Cross-Layer Invalidation Callback

Register with `BlockPool` at startup:

```python
def on_block_allocated(block_id: int) -> None:
    for layer_idx in manager.expert_layers_at_page.pop(block_id, ()):
        layer = manager.pools[layer_idx]
        eid = layer.page_to_expert.pop(block_id, None)
        if eid is not None:
            layer.expert_to_page.pop(eid, None)
```

The callback fires inside `BlockPool.get_new_blocks` **after** `_maybe_evict_cached_block` (so the prefix-cache hash is already cleared) and **before** the block is returned to the caller (so the caller — scheduler for KV, worker for expert DMA — writes fresh data into the page *after* the worker's expert state is invalidated).

Do **not** fire the callback from `touch()`: `touch` is only called on blocks that are *already* valid KV with a prefix-cache hash. Those blocks cannot hold a current expert mapping (they were allocated as KV previously, which would already have cleared any expert mapping at that page via the callback).

---

## 6. Initialization Order (Critical)

This is the trickiest part. Expert slot size determines `block_size`, which determines `num_blocks`, which determines pool-buffer size. In uniprocess, the worker's in-process `vllm_config` is the same object EngineCore reads, so mutations in the worker are visible to the scheduler later in init.

1. **`init_device`, `load_model`** (unchanged). Expert weights are loaded on GPU as part of the model.
2. **Inside `load_model`**, after `prepare_communication_buffer_for_model`, if `expert_offload and expert_unified_pool`:
   - Iterate `FusedMoE` modules, move `w13_weight` / `w2_weight` to CPU-pinned tensors (frees GPU memory). Store `expert_slot_bytes = w13.nbytes + w2.nbytes`, `w13_shape`, `w2_shape`, `expert_dtype`, `num_experts`, `cpu_w13`, `cpu_w2`, and the resolved `layer_idx` per module on `self._expert_offload_metadata`.
   - Compute `bytes_per_token_per_layer = num_kv_heads * head_dim * 2 * dtype_bytes` from the first attention layer's KV cache spec. Then `block_size_tokens = expert_slot_bytes // bytes_per_token_per_layer`. Sanity-check that this value is divisible by the largest supported *kernel* block size for the backend in use (usually 16 or a `MultipleOf(16)` — see `backend.get_supported_kernel_block_sizes()`).
   - Mutate `self.vllm_config.cache_config.block_size = block_size_tokens` and mark it user-specified so downstream doesn't overwrite.
3. **Memory profile** (`determine_available_memory`, unchanged).
4. **EngineCore computes `num_blocks`** via `get_kv_cache_config_from_groups` (unchanged code path). Because `UniformTypeKVCacheSpecs.page_size_bytes = sum(per_layer_page_size)` and each `per_layer_page_size` now equals `expert_slot_bytes`, `num_blocks = available_memory // (num_layers * expert_slot_bytes)`. This is `N_pages`.
5. **`_allocate_kv_cache_tensors`** (unchanged): for each attention layer, allocates `torch.zeros(num_blocks * page_size_bytes, dtype=int8, device=...)`. Per layer this is exactly `N_pages * expert_slot_bytes` bytes — the unified-pool buffer. No code change needed here.
6. **`_reshape_kv_cache_tensors`** (unchanged): views each raw tensor as the attention shape. Because `block_size_tokens` was chosen so `page_size_bytes = expert_slot_bytes`, the reshape succeeds with no modification.
7. **`initialize_from_config` finishes**, scheduler creates `KVCacheManager` → `BlockPool(num_gpu_blocks=N_pages, ...)`.
8. **New post-init step in EngineCore** (or at the end of `initialize_from_config`): if unified pool is active, grab `scheduler.kv_cache_manager.block_pool` and hand it to the worker via a new `collective_rpc("setup_unified_pool", block_pool)` call. The worker:
   - Builds one `PerLayerPool` per MoE layer. `pool_buffer` is the **same tensor** the KV reshape already produced (aliased via the runner's `kv_caches[layer_name]` underlying storage — use the `.untyped_storage()` / raw-tensor view stored in `_allocate_kv_cache_tensors`). Do **not** allocate a second pool buffer.
   - Staging tensors were pre-allocated in `_init_unified_pool_metadata` (step 2, before memory profile) and **filled with all experts from CPU** (`staging_w13[i].copy_(cpu_w13[i], non_blocking=True)` for `i in range(num_experts)`, w2 analogue). Attach them to the `PerLayerPool`. This is a one-shot load; staging is read-only for the rest of the run.
   - Creates `UnifiedPagePoolManager`, registers `manager.on_block_allocated` as a callback on `block_pool`, sets `module._unified_pool = manager` on each FusedMoE module.
   - Warms the first `expert_cache_size` experts per layer in the *pool* (not staging — staging already holds all experts): for each layer, for `eid in range(expert_cache_size)`, call `block_pool.get_new_blocks(1)`, DMA expert `eid` from CPU into the pool page (so pool bytes are realistic, matching Phase 2 residency), update the three mappings (`expert_to_page`, `page_to_expert`, `expert_layers_at_page`), then `block_pool.free_blocks([block])` so ref_cnt goes back to 0 and the block sits at TAIL. Callback fires during `get_new_blocks` but is a no-op because `expert_layers_at_page` is populated *after* the call.
9. Scheduler starts serving requests.

### Null block

`BlockPool.__init__` pops block 0 as a permanent `null_block`. Usable blocks = `N_pages - 1`. Account for this — when computing `expert_cache_size` bounds, assert `expert_cache_size * num_moe_layers <= N_pages - 1`.

---

## 7. Files to Change (Concrete, with Line References on `origin/main`)

Paths relative to `/home/belal/150326/vllm/`.

### 7.1 `vllm/config/offload.py` (+1 field)

After the existing `expert_cache_size` field (around line 102 of the `OffloadConfig` dataclass):

```python
expert_unified_pool: bool = False
"""Enable the unified per-layer page pool that shares GPU VRAM between KV
cache and expert cache via the scheduler's free-block LRU. Requires
expert_offload=True and enable_prefix_caching=True."""
```

Add validation in `validate_offload_config`: if `expert_unified_pool and not expert_offload`, raise. (The prefix-caching requirement is validated elsewhere — see §11.)

### 7.2 `vllm/engine/arg_utils.py` (+1 CLI flag)

Mirror the pattern used for `--expert-offload` (around lines 453, 1031, 1917):

- Add `expert_unified_pool: bool = OffloadConfig.expert_unified_pool` to `EngineArgs`.
- Add `offload_group.add_argument("--expert-unified-pool", **offload_kwargs["expert_unified_pool"])`.
- Pass `expert_unified_pool=self.expert_unified_pool` into the `OffloadConfig(...)` construction.

### 7.3 `vllm/v1/core/block_pool.py` (small additions)

- Add `self._on_alloc_callbacks: list[Callable[[int], None]] = []` in `__init__`.
- Add:

  ```python
  def register_on_alloc_callback(self, cb: Callable[[int], None]) -> None:
      self._on_alloc_callbacks.append(cb)
  ```

- Modify `get_new_blocks` (line 320): after the `_maybe_evict_cached_block` / ref-count loop and before `return ret`, add:

  ```python
  if self._on_alloc_callbacks:
      for blk in ret:
          for cb in self._on_alloc_callbacks:
              cb(blk.block_id)
  ```

  Make sure this runs regardless of `enable_caching`.

- **Do not** add a separate `bump` method; `free_blocks` already appends freed blocks at TAIL (that is the desired MRU behaviour).

### 7.4 `vllm/v1/core/kv_cache_utils.py`

No changes. `FreeKVCacheBlockQueue` already exposes what we need.

### 7.5 `vllm/model_executor/layers/fused_moe/unified_pool.py` (NEW FILE)

Create a new file (none of this code currently exists — the existing `expert_cache.py` stays in place for the non-unified path). Recommended structure:

```python
class PerLayerPool:
    layer_idx: int
    pool_buffer: torch.Tensor          # aliased int8 tensor; DO NOT allocate anew
    n_pages: int
    page_bytes: int
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    expert_dtype: torch.dtype
    w13_bytes: int
    w2_bytes: int
    cpu_w13: torch.Tensor              # [num_experts, *w13_shape], pinned
    cpu_w2: torch.Tensor
    expert_to_page: dict[int, int]
    page_to_expert: dict[int, int]
    staging_w13: torch.Tensor          # [num_experts, *w13_shape], static, filled once at init
    staging_w2: torch.Tensor
    hits: int = 0
    misses: int = 0

    def slot_w13_view(self, page: int) -> torch.Tensor:
        off = page * self.page_bytes
        return self.pool_buffer[off:off + self.w13_bytes].view(self.expert_dtype).view(self.w13_shape)

    def slot_w2_view(self, page: int) -> torch.Tensor:
        off = page * self.page_bytes + self.w13_bytes
        return self.pool_buffer[off:off + self.w2_bytes].view(self.expert_dtype).view(self.w2_shape)

    def invalidate_page(self, page: int) -> None:
        eid = self.page_to_expert.pop(page, None)
        if eid is not None:
            self.expert_to_page.pop(eid, None)


class UnifiedPagePoolManager:
    pools: dict[int, PerLayerPool]
    block_pool: BlockPool
    transfer_stream: torch.cuda.Stream
    expert_layers_at_page: dict[int, set[int]]
    _pending_release: dict[int, list[int]]   # layer_idx -> pool pages pinned this step

    def ensure_experts_loaded(self, layer_idx: int, needed_ids: list[int]) -> None:
        """Update pool residency + LRU. No return value — kernel reads from
        the static staging tensor.
        - Pass 1: classify hits/misses, increment counters.
        - Pass 2: touch hits (pin), get_new_blocks + DMA misses into pool pages.
        - Pass 3: wait_stream on transfer stream so no DMA is in flight when
                  free_blocks is called later.
        - Record pinned pages in self._pending_release[layer_idx].
        """
        # ... see §4 ...

    def release_after_forward(self, layer_idx: int) -> None:
        """Called by FusedMoE after the GEMM. Safe to call immediately — the
        kernel does not read pool pages (it reads staging), so no GEMM sync
        is required."""
        pages = self._pending_release.pop(layer_idx, [])
        if pages:
            blocks = [self.block_pool.blocks[p] for p in pages]
            self.block_pool.free_blocks(blocks)

    def on_block_allocated(self, block_id: int) -> None:
        for li in self.expert_layers_at_page.pop(block_id, ()):
            self.pools[li].invalidate_page(block_id)

    def fill_staging_all_layers(self) -> None:
        """One-shot: copy every expert from CPU into staging. Called once at
        setup; staging is read-only for the rest of the run."""
        with torch.cuda.stream(self.transfer_stream):
            for layer in self.pools.values():
                for eid in range(layer.cpu_w13.shape[0]):
                    layer.staging_w13[eid].copy_(layer.cpu_w13[eid], non_blocking=True)
                    layer.staging_w2[eid].copy_(layer.cpu_w2[eid], non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

    def warm_pool_all_layers(self, n_per_layer: int) -> None:
        """Pre-populate the pool with the first n_per_layer experts of each
        layer. Independent of staging — pool is the real residency store for
        LRU accounting; staging already has everything."""
        for li, layer in self.pools.items():
            n = min(n_per_layer, layer.cpu_w13.shape[0])
            for eid in range(n):
                [block] = self.block_pool.get_new_blocks(1)
                p = block.block_id
                layer.slot_w13_view(p).copy_(layer.cpu_w13[eid])
                layer.slot_w2_view(p).copy_(layer.cpu_w2[eid])
                layer.expert_to_page[eid] = p
                layer.page_to_expert[p] = eid
                self.expert_layers_at_page.setdefault(p, set()).add(li)
                self.block_pool.free_blocks([block])

    def log_stats(self) -> None: ...
```

Implementation notes:
- **No return value from `ensure_experts_loaded`**. The kernel reads the static `staging_w13` / `staging_w2` directly (exposed as `PerLayerPool.staging_w13`). Caller accesses `manager.pools[L].staging_w13` / `.staging_w2`.
- **OOM handling (no free block returnable)**: `BlockPool.get_new_blocks` raises `ValueError("Cannot get ... free blocks from the pool")` — catch `ValueError` (NOT `OutOfBlocksError`, which does not exist in vLLM). MVP response: log an error and raise; do not silently fall back. The scheduler is supposed to bound concurrent sequences so all pages being pinned simultaneously should never happen in our test workloads.
- **Classify unique experts exactly as today's `_forward_with_expert_cache` does** (`topk_ids.unique().tolist()`). No sizing assertion needed — staging is always `num_experts`-wide.
- Record `self._pending_release[layer_idx] = list(pinned_pages)` before returning from `ensure_experts_loaded`.

### 7.6 `vllm/model_executor/layers/fused_moe/layer.py` (small modification)

Base code: the existing `_forward_with_expert_cache` (around line 1575 on `origin/main`). Changes:

1. In `__init__`, add `self._unified_pool = None` alongside `self._expert_cache = None`.
2. In `forward_native` (around line 1568), dispatch unified pool first:

   ```python
   if self._unified_pool is not None:
       return self._forward_with_unified_pool(hidden_states, router_logits)
   if self._expert_cache is not None:
       return self._forward_with_expert_cache(hidden_states, router_logits)
   return self.runner.forward(hidden_states, router_logits)
   ```
3. Add `_forward_with_unified_pool` — simpler than `_forward_with_expert_cache` because Design D is static:
   - `self._unified_pool.ensure_experts_loaded(self.layer_id, needed_expert_ids)` — updates pool residency + LRU, DMAs on miss, returns nothing.
   - `layer_pool = self._unified_pool.pools[self.layer_id]`.
   - Swap `self.w13_weight.data = layer_pool.staging_w13`, `self.w2_weight.data = layer_pool.staging_w2`. **Leave `self.global_num_experts` unchanged** (staging is full width = `num_experts`). **Do not remap `topk_ids`** — global expert IDs index correctly into full-width staging.
   - Run `quant_method.apply(...)` inside a try/finally that restores the original `w13_weight.data` / `w2_weight.data`.
   - Call `self._unified_pool.release_after_forward(self.layer_id)` immediately after the kernel call (no `cuda.synchronize()` needed — the kernel doesn't read pool pages).
4. Do **not** call `_maybe_init_expert_cache` when unified pool is on — the unified-pool path has its own setup (`fill_staging_all_layers` + `warm_pool_all_layers`).
5. Expose a `move_experts_to_cpu()` helper on `FusedMoE` (the unified-pool setup calls this instead of duplicating the CPU-pin logic). The body matches `_maybe_init_expert_cache`'s CPU-pinning lines today.

Use `self.layer_id` (already exists, line ~775 on `origin/main`, via `extract_layer_index(self.layer_name)`) as the layer index.

### 7.7 `vllm/v1/worker/gpu_model_runner.py` (orchestration)

On `origin/main` the file already contains the expert-offload hook around line 4590. Extend it:

1. Replace the existing block:
   ```python
   if self.vllm_config.offload_config.expert_offload:
       from vllm.model_executor.layers.fused_moe.layer import FusedMoE
       for module in self.model.modules():
           if isinstance(module, FusedMoE):
               module._maybe_init_expert_cache()
   ```
   with a branch:
   ```python
   if self.vllm_config.offload_config.expert_offload:
       if self.vllm_config.offload_config.expert_unified_pool:
           self._init_unified_pool_metadata()
       else:
           for module in self.model.modules():
               if isinstance(module, FusedMoE):
                   module._maybe_init_expert_cache()
   ```

2. Add `_init_unified_pool_metadata`:
   - Iterate FusedMoE modules, call `module.move_experts_to_cpu()`.
   - Gather and validate that all layers have identical `w13_shape`, `w2_shape`, `expert_dtype`, `num_experts`.
   - Compute `expert_slot_bytes`, `bytes_per_token_per_layer` (from any attention layer's metadata — `self.model_config.get_num_kv_heads(self.parallel_config)`, `self.model_config.get_head_size()`, `self.kv_cache_dtype`).
   - Compute `block_size_tokens = expert_slot_bytes // bytes_per_token_per_layer`. Assert `expert_slot_bytes % bytes_per_token_per_layer == 0` and that `block_size_tokens` is divisible by 16 (or otherwise supported). If not divisible, round the expert slot bytes up to the next alignment and pad the page.
   - Mutate `self.vllm_config.cache_config.block_size = block_size_tokens` and `self.vllm_config.cache_config.user_specified_block_size = True`.
   - **Reserve staging bytes outside the memory-profile budget** (see §14). Before `determine_available_memory` runs, pre-allocate the per-layer staging tensors (`num_experts × (w13_bytes + w2_bytes) × num_moe_layers` total) so they show up as already-used GPU memory and the profiler subtracts them from the budget it reports to the scheduler. This keeps baseline vs. unified comparisons on equal *pool* budget.
   - Store `self._unified_pool_metadata` with `moe_modules`, `cpu_w13`s, `cpu_w2`s, `expert_slot_bytes`, per-layer shapes, `expert_cache_size`, and the pre-allocated staging tensors.

3. Add `_setup_unified_pool_manager(block_pool)`: called via a new RPC from EngineCore after `initialize_from_config`.
   - For each MoE layer, look up the raw int8 tensor in `kv_caches` / the stored `kv_cache_raw_tensors` via the attention layer name that pairs with this MoE module. The pairing: extract `layer_idx` from each FusedMoE via `module.layer_id`; find the attention layer with matching index (models with MoE have exactly one attention per layer index). Use that layer's raw byte tensor as the `pool_buffer`.
   - Build `PerLayerPool` objects. Staging tensors were pre-allocated in `_init_unified_pool_metadata` (step 2); attach them to the `PerLayerPool` here — do **not** allocate again.
   - Build `UnifiedPagePoolManager`, `block_pool.register_on_alloc_callback(manager.on_block_allocated)`.
   - For each FusedMoE module, set `module._unified_pool = manager`.
   - Call `manager.fill_staging_all_layers()` — one-shot CPU→GPU copy of every expert into every layer's staging. This is the big initial transfer (`num_experts × (w13+w2) × num_moe_layers` over PCIe). Staging is read-only after this point.
   - Call `manager.warm_pool_all_layers(expert_cache_size)` — populate pool pages with the first `expert_cache_size` experts per layer (for LRU residency tracking; kernel does not read these).
   - Assert `expert_cache_size * len(moe_modules) <= block_pool.num_gpu_blocks - 1`. If it fails, reduce `expert_cache_size` and log a warning.

4. Expose `get_unified_pool_block_count(self) -> int | None` that returns `self._unified_pool_manager.block_pool.num_gpu_blocks` if set, else `None`. This is the sanity-check hook (§7.9).

### 7.8 `vllm/v1/worker/gpu_worker.py` (new RPC passthrough)

Add two thin methods on the worker:

```python
def setup_unified_pool(self, block_pool):
    self.model_runner._setup_unified_pool_manager(block_pool)

def get_unified_pool_block_count(self):
    return self.model_runner.get_unified_pool_block_count()
```

They are dispatched via `collective_rpc` from EngineCore.

### 7.9 `vllm/v1/engine/core.py` (wiring)

In `_initialize_kv_caches`, after `self.model_executor.initialize_from_config(kv_cache_configs)` (around line 278):

```python
if self.vllm_config.offload_config.expert_unified_pool:
    bp = self.scheduler_kv_cache_manager.block_pool  # or however you reach it
    self.model_executor.collective_rpc("setup_unified_pool", args=(bp,))
```

Implementation note on the BlockPool reference: in uniprocess, `self.scheduler` is built later in `EngineCore.__init__` (not inside `_initialize_kv_caches`). Safer pattern: move the `setup_unified_pool` call to right after `self.scheduler = Scheduler(...)` is constructed (same `__init__`, later line). At that point `self.scheduler.kv_cache_manager.block_pool` is valid.

No multi-process serialization needed because the assumption is `UniProcExecutor`; assert that early:

```python
assert isinstance(self.model_executor, UniProcExecutor), (
    "--expert-unified-pool requires uniprocess execution (default when no TP/PP set)."
)
```

### 7.10 Config validation — `vllm/engine/arg_utils.py` or `vllm/config/vllm.py`

When `expert_unified_pool=True`, assert:
- `offload_config.expert_offload=True` (otherwise no CPU copy of experts exists).
- `cache_config.enable_prefix_caching=True`.
- `parallel_config.tensor_parallel_size == 1`.
- `parallel_config.pipeline_parallel_size == 1`.
- The model uses a standard attention (no MLA, no mamba). For MVP, check `hf_config.model_type in {"olmoe", "mixtral", ...}` or similar allow-list; otherwise raise.

---

## 8. Known Sharp Edges & How to Handle Them

1. **Block size vs kernel block size**: the manager's `block_size` (~1500 tokens for OLMoE, ~86K for Mixtral) is much larger than any attention kernel block size. vLLM already handles this in `prepare_kernel_block_sizes` (`vllm/v1/worker/utils.py:329`) by splitting manager blocks into kernel-sized subblocks. You don't need to intervene as long as the manager `block_size` is divisible by the kernel block size (typically 16). Validate this in `_init_unified_pool_metadata`.
2. **Prefix caching granularity**: with a 1500-token manager block, prefix caching only reuses prefixes in 1500-token chunks. The dissertation's Scenario A uses a 70K shared prefix, so `70000 // 1536 ≈ 45` blocks worth of hits per request. Still meaningful. Confirm this during §12.
3. **No GEMM sync needed (Design D)**: the kernel reads the static staging tensor, never pool pages. Pool pages are written to only on miss by the transfer stream, and `wait_stream` inside `ensure_experts_loaded` guarantees the DMA is complete before any subsequent allocator op. `release_after_forward` can run immediately after the kernel call. This is the major correctness simplification Design D buys over Design B.
4. **Staging buffer**: fixed `[num_experts, *w13_shape]` per layer, filled once at setup, never modified during the run. No sizing assertion. See §14 for how its memory is excluded from the simulated budget.
5. **Out-of-memory eviction**: if every block has `ref_cnt > 0`, `get_new_blocks(1)` raises `ValueError`. In the target uniprocess + single-concurrency benchmarks this cannot happen. Log and re-raise; do not fall back.
6. **Scheduler vs worker ordering**: vLLM V1 runs scheduler step and worker execute_model sequentially in uniprocess. No lock needed for BlockPool access. If future work uses an async engine loop, this assumption must be revisited.
7. **Attention backends**: only need to work with the backend used by the benchmark (`--enforce-eager` implies eager attention kernels, and FlashAttention v2 is the default on L40). The plan requires no changes in `vllm/v1/attention/`.
8. **Hybrid models / Mamba / MLA / sliding window**: explicitly refused by the config validator.
9. **Pool page bytes are "simulation-only"**: the kernel never reads them. They exist so the memory profiler correctly accounts for pool capacity (real bytes consumed per page) and so CPU→GPU DMA on miss has a real physical destination (real PCIe latency cost). If we ever want to drop even that, we could skip the DMA entirely and just record miss stats — but then miss latency disappears from the benchmark and the TPOT numbers lose meaning.

---

## 9. Data Structures Summary

| Object | Lives where | Purpose |
|---|---|---|
| `pool_buffer` (int8, one per layer) | Worker's `PerLayerPool` | Shared byte pool for KV+experts. Aliased to the existing per-layer KV raw tensor; no new allocation. |
| `staging_w13`, `staging_w2` (bf16, one pair per layer) | Worker's `PerLayerPool` | **Static** full-expert GEMM input. Size `num_experts × one expert`. Filled once at init from CPU, never modified. Excluded from simulated GPU budget (§14). |
| `expert_to_page`, `page_to_expert` | Per layer | Where each expert currently lives in this layer's pool. |
| `expert_layers_at_page` | Manager | Cross-layer view for the invalidation callback. |
| `FreeKVCacheBlockQueue` (HEAD→TAIL LRU) | Scheduler's `BlockPool` | Single source of truth for "who is coldest". |
| `cpu_w13[num_experts, ...]`, `cpu_w2[num_experts, ...]` | Per layer, pinned RAM | Source of truth for expert weights. |

---

## 10. End-to-End Walkthrough (Single Decode Step, Layer L, top-k=8, Design D)

1. Scheduler prepares `SchedulerOutput` (may allocate new KV blocks via `get_new_blocks` → callback fires → worker's expert mapping cleaned up for those pages).
2. Worker runs forward.
3. Layer L attention: reads KV from `pool_buffer` using the standard block table. Shared (synced) placement → nothing special.
4. Layer L FusedMoE:
   a. Router picks 8 unique experts `[e1..e8]`.
   b. `manager.ensure_experts_loaded(L, [e1..e8])`: 6 hits, 2 misses.
   c. Hits: `touch([P_hits])` (pins the 6 pool pages; protects them from scheduler eviction this step).
   d. Misses: `get_new_blocks(1)` × 2 → pages `P_new`. Callback invalidates any other layer that held an expert at those pages. DMA expert weights CPU→GPU into those pool pages (real PCIe cost — this is the miss latency the dissertation measures).
   e. `wait_stream` on transfer stream so no late DMA can race future allocator ops.
   f. `ensure_experts_loaded` returns (no staging work).
5. FusedMoE swaps `w13_weight.data = staging_w13`, `w2_weight.data = staging_w2` (full static tensors), runs GEMM without remapping `topk_ids`.
6. `manager.release_after_forward(L)` → all 8 pinned pool pages freed, ref_cnt→0, appended at TAIL. (No `cuda.synchronize()` — kernel doesn't read pool pages.)
7. Loop for other layers.

Invariant after each MoE layer: every pool page "used" (pinned-then-released) this step is at TAIL. Pages untouched this step drift toward HEAD. Next `get_new_blocks` call pops the globally coldest page (KV or expert).

Note: the static staging is what the kernel actually reads, so layer L's compute is identical regardless of pool composition. The pool's state affects the *cost* structure (miss DMAs, evictions) visible in the benchmark, not the numeric output of the forward.

---

## 11. Open Defaults

| Question | MVP answer |
|---|---|
| Executor | `UniProcExecutor` only. Assert at startup. |
| OOM fallback | None. Raise and let the benchmark fail loudly. |
| Warm-up | Sequential `eid in range(expert_cache_size)` per layer, identical to current `ExpertCache`. |
| Stats logging | `PerLayerPool` keeps `hits` / `misses`; `manager.log_stats()` prints per-layer aggregates every 100 steps and at shutdown. Log the HEAD-of-queue composition (expert vs KV) once per 100 steps to make the rebalancing dynamic observable in the dissertation. |
| BlockPool `bump` method | Not introduced. `free_blocks` appends at TAIL already. |
| Quantization | BF16 only. |
| Dtype handling in staging | Match `expert_dtype` (bf16). |
| Callback fired by `touch` | No — only by `get_new_blocks`. |
| Staging write after init | No. Static full-expert copy, one-shot at setup. |
| Kernel `topk_ids` remap | No. Staging is full-width; native indexing works. |
| Post-GEMM `cuda.synchronize()` | No. Kernel reads staging, not pool pages. |

---

## 12. Verification Procedure

Work through these in order; do not move on until each passes.

1. **Imports compile**:
   `python -c "from vllm.model_executor.layers.fused_moe.unified_pool import UnifiedPagePoolManager, PerLayerPool"`.
2. **Sync venv**: copy every modified source file into the matching path under `venv/lib/python3.12/site-packages/vllm/` (mirror the source layout). Re-run the import test against the venv Python: `venv/bin/python -c "..."`.
3. **Server boots**:
   ```
   venv/bin/python -m vllm.entrypoints.openai.api_server \
       --model allenai/OLMoE-1B-7B-0924 \
       --expert-offload --expert-unified-pool --expert-cache-size 12 \
       --enforce-eager --enable-prefix-caching --trust-remote-code \
       --gpu-memory-utilization 0.40 --max-model-len 131072
   ```
   Confirm startup log shows:
   - `# GPU blocks: <N_pages>` and the computed `block_size` is the large tokens-per-expert value.
   - `Staging overhead: X.XX GB (num_experts=64 × <bytes> × <num_moe_layers>)` and `Adjusted available memory: <Y> GB` so the §14 accounting is visible.
4. **Correctness smoke test**: issue one short completion request; compare tokens to a baseline run (`--expert-offload` without `--expert-unified-pool`). They should match modulo BF16 nondeterminism.
5. **Scenario A run** (KV-hot, `motivation-benchmarks.md` §A): with `--expert-cache-size 64 --expert-unified-pool`, verify TTFT drops to within ~15% of the `A-12` baseline.
6. **Scenario D run** (expert-hot, `motivation-benchmarks.md` §D): with `--expert-cache-size 12 --expert-unified-pool`, verify TPOT drops to within ~15% of the `D-64` baseline.
7. **Stats**: `manager.log_stats()` output should show hits + misses across all MoE layers, with `cpu_fallbacks == 0` throughout.
8. **Dynamics**: per-100-step HEAD composition log should show KV-dominated HEADs during Scenario A and expert-dominated HEADs during Scenario D. This is the core dissertation evidence.

---

## 13. Out of Scope (Phase 2+)

- True per-layer expert eviction (no cross-layer collateral).
- Modify `fused_moe_kernel` at `vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:315` to take a `page_table_ptr` argument and index `b_ptr + page_table[expert_id] * page_stride` instead of `b_ptr + expert_id * stride_be`. This eliminates staging entirely.
- Sub-page KV granularity (shrinks `block_size` while keeping unified pool).
- Multi-process / TP > 1.
- Quantized experts (FP8, INT8, AWQ). Would additionally require modifying `fused_moe_kernel_gptq_awq` at line 81 of the same file.
- Heuristic / data-driven warm-up.
- Runtime-adaptive `expert_cache_size`.

---

## 14. Simulation Framing (Dissertation Methodology)

This section is not a coding instruction; it's the experimental framing that the code must support.

### What the MVP actually simulates

The dissertation claim is: *a unified KV+expert pool with a single LRU dynamically rebalances memory between KV cache and expert cache based on workload pressure, outperforming static partitioning.*

Phase 2's steady-state design:
- One pool buffer per layer holding KV blocks and expert pages at arbitrary positions.
- A modified `fused_moe_kernel` (at `vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:315`) that takes an extra page-table argument and replaces `b_ptr + expert_id * stride_be` with `b_ptr + page_table[expert_id] * page_stride`. No expert duplication on GPU.

Phase 1 (this MVP, "Design D"):
- Implements the pool + LRU + cross-layer invalidation (the research-interesting part).
- Does real CPU→GPU DMA on pool expert misses (so miss latency is measured honestly).
- Keeps a static full-expert `staging` tensor per layer that the *unmodified* kernel reads from every step. This is how we avoid modifying `fused_moe_kernel` in Phase 1.

The physical latency profile of a Design D forward matches what Phase 2 would produce:

| | Phase 2 (real) | Phase 1 Design D (sim) | Naive "gather" scaffolding |
|---|---|---|---|
| CPU→GPU DMA on pool miss | ✓ real cost | ✓ real cost | ✓ real cost |
| GPU→GPU gather before GEMM | ✗ none | ✗ none | ✓ scaffolding tax |
| GEMM reads staging or pool? | pool (direct) | static staging (faked) | compacted staging slices |
| Per-layer `cuda.synchronize()` | no | no | yes (scaffolding tax) |

So Design D is latency-accurate for Phase 2's steady state. Only the GPU memory footprint differs (Phase 1 has a redundant full-expert staging; Phase 2 does not).

### Memory accounting

To keep baseline-vs-unified comparisons honest, the simulated GPU budget (what `--gpu-memory-utilization` controls, what `determine_available_memory` returns) must be identical for:

- **Baseline**: static partition of `B` bytes between `ExpertCache.cache_w13` region and KV cache region.
- **Unified**: dynamic partition of `B` bytes within the pool (KV blocks + expert pages).

Because the unified run pays an extra fixed cost for staging (`num_experts × (w13_bytes + w2_bytes) × num_moe_layers`), that cost is allocated **before** the memory profile runs. The profile then sees staging as already-consumed GPU memory and reports `(free_memory - staging)` as available. The scheduler/pool then get the same `B` bytes the baseline gets. Staging is a tax on the MVP implementation, not on the idea.

The dissertation text states this explicitly: *"The staging buffer is a Phase 1 artifact. Phase 2 eliminates it by modifying `fused_moe_kernel` to gather expert weights directly from pool pages via a page-table argument. For experimental comparisons, staging is excluded from the GPU budget so that baseline and unified runs operate on equal dynamic memory."*

### What this buys us

- **No assertion edge cases.** `len(needed) ≤ num_experts` is tautological.
- **No per-step copies.** Static staging means zero GPU→GPU traffic during a forward.
- **No GEMM sync.** The kernel doesn't read pool pages.
- **No `topk_ids` remap.** Staging is full-width, so global expert IDs index correctly.
- **Latency-accurate.** Pool miss costs are real PCIe time; kernel cost is identical to Phase 2.
- **Fair benchmarks.** Scenario A and D comparisons are apples-to-apples on pool/KV capacity.

### What this does not claim

- Phase 1 is **not** a production system. Staging overhead (~12.9 GB for OLMoE, 64 experts × 16 layers) is prohibitive at real scale. Phase 2 is required for a shippable system.
- The absolute GPU memory footprint of a unified-pool run is *larger* than baseline. The research claim is about behavior under equal dynamic budget.
- We are not measuring kernel-modification overhead itself (there is none in Phase 1). We claim Phase 2 preserves Phase 1's latency profile modulo one extra indirection per `b_ptr` load, which is a safe assumption for a hand-written Triton kernel.

### Required hooks in code

- `_init_unified_pool_metadata` allocates staging before `determine_available_memory` (§7.7 step 2) and fills it with all experts from CPU.
- Startup log prints the staging overhead and the adjusted available budget so reviewers can audit the accounting.
- Benchmark scripts in `motivation-benchmarks.md` pass the same `--gpu-memory-utilization` to baseline and unified runs.
