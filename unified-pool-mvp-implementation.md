# Unified Per-Layer Page Pool — MVP Implementation

> Companion to `unified-pool-mvp-plan.md`. The plan describes intent; this doc describes what's actually built on the `unified-pool-mvp` branch.

This implementation lives across 8 files in `vllm/vllm/`. One new module (`unified_pool.py`), 7 modified files.

```
vllm/vllm/config/offload.py                              ~185 lines total (config + validator added)
vllm/vllm/engine/arg_utils.py                            +3 hook lines (field, CLI arg, kwarg pass-through)
vllm/vllm/model_executor/layers/fused_moe/unified_pool.py 712 lines (new)
vllm/vllm/model_executor/layers/fused_moe/layer.py       1836 lines total
vllm/vllm/v1/core/block_pool.py                          612 lines total
vllm/vllm/v1/engine/core.py                              +13 lines (Stage-2 RPC dispatch)
vllm/vllm/v1/worker/gpu_model_runner.py                  ~285 lines added (Stage 1, Stage 2, raw-KV capture)
vllm/vllm/v1/worker/gpu_worker.py                        +3 lines (RPC passthrough)
```

vLLM is installed as a regular (non-`-e`) package. Every edit must be mirrored to `venv/lib/python3.12/site-packages/vllm/...` for the running server to see it. (As of 2026-05-01 the venv is in sync with `vllm/vllm/...`; verify with `diff -q` before re-syncing.)

Line numbers cited below were accurate at write-time; later edits (notably trace-logging additions in `unified_pool.py`) have caused drift. Treat them as orientation hints, not guarantees — search for the symbol if a number looks off.

---

## 1. Architectural Shape

```
┌──────────────────────────────────────────────────────────────┐
│  EngineCore (engine/core.py)                                  │
│   ├─ Scheduler                                                │
│   │   └─ KVCacheManager                                       │
│   │       └─ BlockPool ──┐                                    │
│   └─ Executor (UniProcExecutor)                               │
│       └─ Worker          │                                    │
│           └─ GPUModelRunner                                   │
│               ├─ FusedMoE × N    (one per MoE layer)          │
│               │   └─ UnifiedPool (per-layer state)            │
│               └─ UnifiedPoolManager  ◄── BlockPool reference  │
└──────────────────────────────────────────────────────────────┘
```

**One `UnifiedPoolManager`** owns the cross-layer state: the BlockPool reference, the shared prefix-recency LRU, the cross-layer `block_id → set(layer_idx)` index, the dedicated CUDA transfer stream, and the global `step` counter.

**One `UnifiedPool` per MoE layer** owns the per-layer state: this layer's expert-recency LRU, its `block_id ↔ expert_id` mappings, its CPU-pinned source-of-truth tensors, its full-width staging tensors, its alias slice into the layer's KV byte tensor, and its hit/miss counters.

**Block IDs are global; physical buffers are per-layer.** A single `block_id` denotes the same offset in every layer's pool buffer. Two consequences:
- An expert pinned at `block_id 5` in layer L only consumes physical memory in L's buffer; layer L'ʼs offset 5 is independently usable.
- A KV write at `block_id 5` is synced — it physically overwrites every layer's bytes at offset 5 simultaneously, so it must invalidate every layer's expert mapping at 5.

---

## 2. File-by-File

### 2.1 `config/offload.py`

Adds two things to `OffloadConfig`:

- `expert_unified_pool: bool` (line 113) — the new flag.
- A `model_validator` clause (line 164) requiring `expert_offload=True` when `expert_unified_pool=True`. Other constraint validation (TP=1, prefix caching, eager) is deferred to Stage 1 where the relevant configs are accessible.
- The existing `expert_cache_size` docstring is updated (line 109) to note that under the unified pool it controls only initial warm occupancy.

### 2.2 `engine/arg_utils.py`

Three lines: register the field default, add the CLI argument, and forward the value into `OffloadConfig` when constructing `VllmConfig`.

### 2.3 `model_executor/layers/fused_moe/unified_pool.py` (new)

The data plane. 712 lines, organized into:

**Module-level helpers** (lines 24-47):
- `_trace_enabled()` — reads `VLLM_UNIFIED_POOL_TRACE`. Toggles the per-forward `CACHE`/`EXPERT_LRU`/`PREFIX_LRU`/`REQUEST`/`CLAIM`/`RESULT`/`EVICT` lines (and the `PREFIX_ADDED`/`PREFIX_REMOVED` callback lines) used by the §6.6 architectural acceptance test. See §8 for the full format.
- `move_experts_to_cpu()` — extracts the CPU-pin logic that used to live inline in `_maybe_init_expert_cache`. Reused by both the plain expert-cache path and the unified Stage 1.

**`UnifiedPool`** (class, ~lines 50-153) — per-layer state. Fields:
- `staging_w13`, `staging_w2` — full-width GPU tensors `[num_experts, …]`, filled once at startup. The unmodified Triton kernel reads these every forward (Phase 1 simulation).
- `cpu_w13`, `cpu_w2` — CPU-pinned source-of-truth, DMA source on miss.
- `pool_buffer` — flat int8 view, narrowed to `num_gpu_blocks × page_size_bytes`, aliased onto the layer's KV byte tensor (no separate allocation).
- `expert_at_block: dict[int, int]`, `block_at_expert: dict[int, int]` — bidirectional mapping for this layer.
- `expert_lru: OrderedDict[int, int]` — *the* per-layer expert-recency structure consulted at eviction time. Bumped on every hit and miss.
- `pinned_blocks: set[int]` — blocks held off the eviction-candidate list for the duration of one forward.
- `hits`, `misses`, `forward_count` — stats.
- `manager` — back-reference to the owning `UnifiedPoolManager`. **Set externally by `setup_unified_pool` after construction** (gpu_model_runner.py: `pool.manager = manager`); not a constructor argument. Used by `_forward_with_unified_pool` to call `manager.ensure_loaded` / `release_pinned` / `end_forward_step` without threading the manager through every call site.

Key methods:
- `assign(block_id, expert_id, step)` — install a new mapping, MRU on `expert_lru`. Asserts both directions of the mapping were free.
- `drop(block_id) → expert_id | None` — remove the mapping.
- `bump_expert(expert_id, step)` — `move_to_end` on the LRU. Called for hits *and* misses inside `ensure_loaded`.

**`UnifiedPoolManager`** (class, ~lines 156-712) — cross-layer state.

Constructor (lines 175-195):
- Asserts the passed-in object is a `BlockPool`.
- Initializes `layers: dict[int, UnifiedPool]`, `block_holder: dict[int, set[int]]` (cross-layer "which layers hold an expert at this block"), `transfer_stream` (a dedicated `torch.cuda.Stream` for DMAs), `step`, and the shared `prefix_lru: OrderedDict[int, int]`.
- Registers three callbacks on the BlockPool: `_on_kv_allocation`, `_on_prefix_added`, `_on_prefix_removed`.

Mapping helpers (lines 207-281) — `_add_holder`, `_remove_holder`, `_drop_layer_mapping`, `_broadcast_drop_all_layers`, `_evict_prefix_globally`. The two collateral paths the plan calls out (§2 paragraph 7) live here:
- `_drop_layer_mapping` is the **per-layer-only** path — used when an expert miss in L overwrites L's bytes at a block.
- `_broadcast_drop_all_layers` is the **across-all-layers** path — used only when KV writes overwrite every layer's bytes at a block (i.e. KV-allocation).

Callbacks (lines 287-325):
- `_on_kv_allocation(block_ids)` — fired by BlockPool at the end of `get_new_blocks`. Calls `_broadcast_drop_all_layers` for each block. The KV writes are about to land; every layer's expert mapping there is now invalid.
- `_on_prefix_added(block_id)` — fired when a block re-enters the free queue with a hash. Bumps to MRU on `prefix_lru`. **This is the bump-on-hit signal for prefix recency** (a request just finished using these bytes — that's the freshest last-used timestamp we have without an attention-side hook).
- `_on_prefix_removed(block_id)` — fired when a block leaves the cached-prefix state (either pinned via `touch` or hash cleared). Drops from `prefix_lru`.

`warm_up(warm_count)` (lines 331-382) — Stage 2 pre-fill. **Shares one `block_id` per warmed expert across all layers.** The outer loop is over experts (not layers): for each `expert_id` in `range(warm_count)`, pick one block via `_select_victim_block` from layer 0's perspective (queue head), then iterate every layer and DMA `expert_id` into that same block_id's slot in each layer's per-layer pool buffer. Total block_ids consumed = `warm_count`, regardless of `num_layers`. After all DMAs, re-append the block to `free_block_queue` so the BlockPool still sees it as available for KV allocation. Synchronous DMAs (`_dma_expert_into_block_sync`) — warm-up is a one-shot startup step, not on the hot path.

Forward-path API (lines 388-479):

- `ensure_loaded(layer, needed_expert_ids)` — the hot path. Three passes:
  1. **Classify** hits vs misses (no mutation; trace snapshot captured here, *before* any state change).
  2. **Claim a block per miss** via `_select_victim_block` (Tier 1 free → Tier 2 cold tail of mixed LRU). For each victim: clear any cached-prefix hash globally, drop this layer's stale mapping if any, install the new mapping, add to `block_holder`, add to `pinned_blocks`, re-append the block to `free_block_queue` so KV allocation can still see it.
  3. **Batch DMAs** on `transfer_stream`, then `wait_stream` so the kernel can't launch before DMAs complete.
  Hits are pinned and bumped to MRU on `expert_lru`. Misses are pinned and recorded as MRU at assign time.

- `release_pinned(layer)` — clears `pinned_blocks` and increments `forward_count`.

- `end_forward_step()` — increments the global `step` counter. Logs stats every 100 steps. **Called once per forward (after all layers' MoE blocks have run)** — this is what the layer's `_forward_with_unified_pool` invokes after `release_pinned`.

`_select_victim_block(layer, needed_set) → (KVCacheBlock, tier_str)` (~lines 485-585) — three-tier victim search. Returns a tuple — the `tier_str` label is used for trace output and is one of:
- `"free-pure"` — Tier 1 hit on a slot with no holders.
- `"free-cross-layer-expert"` — Tier 1 hit on a slot that holds another layer's expert (safe to claim; pool buffers are per-layer).
- `"expert-local"` — Tier 2 chose the cold end of L's own `expert_lru`.
- `"prefix-global"` — Tier 2 chose the cold end of the shared `prefix_lru`.

1. **Tier 1 — free space from L's view.** Walk `block_pool.free_block_queue` head→tail. Return the first block that is (a) not in L's `pinned_blocks`, (b) has no expert mapping in *L* (other layers' mappings don't matter — physical bytes are independent), and (c) has no prefix hash. This is what lets two layers independently use the same `block_id` for different experts.

2. **Tier 2 — cold tail of L's mixed LRU.** Compare the front (oldest) of `layer.expert_lru` to the front of the shared `prefix_lru`. Pick whichever has the smaller `step` (= used longer ago). Tie-break: prefix wins (`<=` favours prefix-global on equal step). Skip pinned and currently-needed entries. The chosen block is removed from `free_block_queue` so the KV path can't pop it before the miss-DMA lands.

3. **Tier 3 — fail loudly.** Raise `RuntimeError`. The pool is exhausted; warn the operator to lower `--max-num-batched-tokens` or raise `--num-gpu-blocks-override`.

DMA helpers (lines 591-612) — `_dma_expert_into_block_async` does the two `narrow().copy_()` calls, one for w13 and one for w2, into the layer's `pool_buffer`. The `_sync` variant wraps it in the transfer stream for warm-up.

Trace helper (lines 618-687) — `_trace_pre_mutation` builds and prints the per-step header, `CACHE` composition counts, `EXPERT_LRU`, `PREFIX_LRU` (top 8), and `REQUEST` lines from `ensure_loaded` Pass 1, before any state mutation.

Stats (lines 693-712) — `log_stats()` and `shutdown_log()` print per-layer hits/misses/hit-rate, expert page count, and shared kv-prefix page count.

### 2.4 `model_executor/layers/fused_moe/layer.py`

Three blocks of changes inside `FusedMoE`:

- `__init__` (around line 641) sets `self._unified_pool = None` and `self._unified_pool_enabled` based on the offload config. The pool object itself is attached later by Stage 2. **Note:** `_unified_pool_enabled` is currently set but never read — `forward_native`'s dispatch checks `self._unified_pool is not None` directly. Safe to delete in a cleanup pass.

- `_maybe_init_expert_cache` (around line 732) — extracted to use `move_experts_to_cpu` instead of inlining. **Should be guarded so it's a no-op when `_unified_pool_enabled`** — currently the worker (gpu_model_runner.py:4592-4599) handles the branch externally, so this method isn't called when the unified pool is on.

- `unified_pool_stage1()` (lines 765-823) — Stage-1 entry point. Pins experts to CPU (`move_experts_to_cpu`), allocates the per-layer `staging_w13` and `staging_w2` GPU tensors, fills them once with all experts, stashes them and the CPU-pinned tensors on the module as `_unified_pool_staging_*` and `_unified_pool_cpu_*`, returns a metadata dict with `layer_idx`, `num_experts`, `w13_dtype`/`w2_dtype`, byte sizes per expert, and staging nbytes.

- `attach_unified_pool(pool)` (line 825) — Stage-2 entry point. Stores the per-layer `UnifiedPool` so `forward_native` dispatches to `_forward_with_unified_pool`.

- `forward_native` (line 1624) — adds a branch at the top: if `self._unified_pool is not None`, dispatch to `_forward_with_unified_pool` before the existing `_expert_cache` path.

- `_forward_with_unified_pool` (lines 1640-1689) — the forward. Steps:
  1. Run the router to get `topk_weights`, `topk_ids`.
  2. `needed_expert_ids = topk_ids.unique().tolist()`.
  3. `manager.ensure_loaded(pool, needed_expert_ids)`.
  4. Inside try/finally, swap `self.w13_weight.data` and `self.w2_weight.data` to point at `pool.staging_w13` and `pool.staging_w2`. Call `self.quant_method.apply(...)`. Restore originals in finally.
  5. `manager.release_pinned(pool)` then `manager.end_forward_step()`.

`global_num_experts` is left unchanged (staging is full-width); `topk_ids` is **not** remapped (global ids index correctly into full-width staging).

### 2.5 `v1/core/block_pool.py`

Three callback lists added in `BlockPool.__init__` (lines 188-198):
- `_on_allocation_callbacks: list[Callable[[list[int]], None]]`
- `_on_prefix_added_callbacks: list[Callable[[int], None]]`
- `_on_prefix_removed_callbacks: list[Callable[[int], None]]`

Three registration methods: `register_on_allocation_callback`, `register_on_prefix_added_callback`, `register_on_prefix_removed_callback`.

Fan-out points:

- `get_new_blocks` (around line 367) — at the **end**, after `_maybe_evict_cached_block` has already cleared any hashes, fires `_on_allocation_callbacks` with the list of new block IDs. The unified pool's `_on_kv_allocation` then broadcasts expert-mapping invalidation across layers.

- `_maybe_evict_cached_block` (around line 405) — after `block.reset_hash()`, fires `_on_prefix_removed_callbacks(block_id)`. This handles both KV-allocation evictions (via `get_new_blocks`) and explicit hash clears (via the new `evict_prefix_hash`).

- `touch` (around line 440) — when a block transitions out of the free queue with a hash, fires `_on_prefix_removed_callbacks`. (When all refs release, `free_blocks` re-adds it to the prefix LRU.)

- `free_blocks` (around line 477) — for every block re-entering the queue with a hash set, fires `_on_prefix_added_callbacks(block_id)`. **This is the bump-on-hit signal for prefix recency** — a request just finished, so this is the freshest last-used timestamp.

New method `evict_prefix_hash(block_id) → bool` (lines 516-525) — public API for the unified pool to clear a cached-prefix hash globally when an expert miss reuses a previously cached block.

### 2.6 `v1/engine/core.py`

After the scheduler is constructed (around line 146), if `expert_unified_pool` is set: assert the executor is `UniProcExecutor`, grab `scheduler.kv_cache_manager.block_pool`, and `collective_rpc("setup_unified_pool", args=(block_pool,), single_value=True)`.

The `UniProcExecutor` assertion is intentional — multi-process serialization of a `BlockPool` reference would require pickling the live block array and the free-queue linked list, which the MVP does not implement.

### 2.7 `v1/worker/gpu_worker.py`

Three-line passthrough (lines 518-520):

```python
def setup_unified_pool(self, block_pool) -> int:
    return self.model_runner.setup_unified_pool(block_pool)
```

### 2.8 `v1/worker/gpu_model_runner.py`

Three new methods on `GPUModelRunner`:

- `_unified_pool_stage1()` (line 6557) — see §3.1 below.
- `_unified_pool_capture_raw_kv(kv_cache_raw_tensors)` (line 6715) — small helper called from `_allocate_kv_cache_tensors` (line 6267) to snapshot the per-layer KV byte tensor dict for Stage 2.
- `setup_unified_pool(block_pool) → int` (line 6724) — see §3.4 below.

Plus one branch in `load_model` (line 4592-4599):

```python
if self.vllm_config.offload_config.expert_offload:
    if self.vllm_config.offload_config.expert_unified_pool:
        self._unified_pool_stage1()
    else:
        for module in self.model.modules():
            if isinstance(module, FusedMoE):
                module._maybe_init_expert_cache()
```

---

## 3. Initialization Sequence

The init splits into two stages, separated by vLLM's memory-profile and KV-cache-allocation flow.

### 3.1 Stage 1 — `_unified_pool_stage1`

Runs inside the worker's `load_model`, immediately after weights are loaded and before `profile_run`. Order:

1. **Validate flags** (lines 6571-6601). Asserts:
   - `tensor_parallel_size == 1` (no multi-rank coordination in MVP).
   - `pipeline_parallel_size == 1`.
   - `enable_prefix_caching == True`.
   - `enforce_eager == True` (variable-length DMAs aren't capturable in CUDA graphs).
   - `max_num_batched_tokens == 1` (a long prefill batch could route to hundreds of distinct experts in one forward and exhaust pool pages mid-call).

2. **Walk all FusedMoE modules** (lines 6603-6614). For each, call `module.unified_pool_stage1()` to (a) pin its experts to CPU, (b) allocate and fill its full-width staging tensors, and (c) return metadata.

3. **Verify uniform expert slot size across layers** (lines 6618-6625) — required for a single uniform page size.

4. **Read the first attention layer's KV spec** (lines 6627-6652). Asserts every attention layer has the same `AttentionSpec` subclass (no MLA/Mamba/sliding-window in the MVP).

5. **Derive `block_size_tokens`** (lines 6654-6681):
   ```
   bytes_per_token = 2 (K+V) × num_kv_heads × head_size × dtype_size
   block_size_tokens = expert_slot_bytes // bytes_per_token
   ```
   Asserts `expert_slot_bytes % bytes_per_token == 0` and `block_size_tokens % 16 == 0` (the kernel block size). For OLMoE-1B-7B-0924-Instruct: `expert_slot_bytes = 12,582,912`, `bytes_per_token = 8,192`, → `block_size_tokens = 1,536`.

6. **Mutate `cache_config.block_size`** (lines 6683-6690):
   ```python
   self.cache_config.block_size = block_size_tokens
   self.block_size = block_size_tokens
   self.cache_config.user_specified_block_size = True
   ```
   The `user_specified_block_size = True` flag is what stops `update_block_size_for_backend` (called from `UniProcExecutor._init_executor` after `load_model`) from clobbering this back to the backend's preferred 16. *That bug — using the wrong attribute name `block_size_user_specified` — was the cause of the boot failure where `num_gpu_blocks` came out as 13787 (= 16-token blocks) instead of 143 (= 1536-token blocks).*

7. **Stash MoE module list and metadata** on the runner (lines 6707-6713) for Stage 2.

After Stage 1 returns:
- `_init_executor` calls `update_block_size_for_backend` — short-circuits because of the user-specified flag.
- `determine_available_memory` runs `profile_run`. The staging tensors allocated in Stage 1 show up in the profile's weight footprint, so the KV budget reported back to the engine is **already net of staging overhead**. This is the §3 fairness invariant: baseline and unified runs compete on the same dynamic memory budget.

### 3.2 Engine — `_initialize_kv_caches`

Standard vLLM flow:
1. `get_kv_cache_specs()` — each attention layer's `get_kv_cache_spec` reads `cache_config.block_size` (now 1536) at call time and returns specs with that block size.
2. `determine_available_memory()` — profile result minus staging.
3. `get_kv_cache_configs(...)` — derives `num_gpu_blocks` from `available_memory // page_size_bytes // num_layers`. For the OLMoE setup: `26.93 GiB / 12.58 MiB / 16 layers ≈ 143 blocks`.
4. `model_executor.initialize_from_config(kv_cache_configs)` — workers allocate the per-layer KV byte tensors. The runner's `_allocate_kv_cache_tensors` snapshots them via `_unified_pool_capture_raw_kv`.

### 3.3 Scheduler construction

The engine then constructs the scheduler, which constructs `KVCacheManager`, which constructs the `BlockPool`. `BlockPool.num_gpu_blocks = 143` (matches `cache_config.num_gpu_blocks`).

### 3.4 Stage 2 — `setup_unified_pool(block_pool)`

Triggered by `engine/core.py:160`'s collective RPC. Order (gpu_model_runner.py:6724):

1. **Build `attn_layers: dict[layer_idx → torch.Tensor]`** by walking `compilation_config.static_forward_context`, filtering to `AttentionLayerBase`, matching against the captured raw-KV tensor dict, extracting `layer_idx` from the layer name, and storing the int8 view of the raw tensor.

2. **Construct the `UnifiedPoolManager`** with the block_pool reference. Constructor registers the three BlockPool callbacks.

3. **For each FusedMoE module**, look up its layer's KV byte tensor, **narrow it to `num_gpu_blocks × page_size_bytes`** (`raw_kv.narrow(0, 0, required_bytes)`), construct a `UnifiedPool` aliasing that slice, register it with the manager, and call `module.attach_unified_pool(pool)`.

   The narrow is necessary because vLLM sizes per-layer KV tensors to their share of the available KV budget, which is generally larger than `num_gpu_blocks × page_size_bytes` (block IDs only address the first `num_gpu_blocks` pages). Without the narrow, the alias's last byte would land past the addressable region and the math would not balance. *This was the second boot failure — "KV byte tensor size 1807089664 is not a multiple of page_size_bytes 12582912".* The slack at the tail is unaddressable in the baseline too, so dropping it is fair-benchmark-neutral.

4. **Warm-up sanity check** (gpu_model_runner.py lines 6817-6831). The check is on `expert_cache_size` alone (not `× num_moe_layers`) because warm-up shares one `block_id` per warmed expert across **all** layers (`UnifiedPoolManager.warm_up`, unified_pool.py lines 331-382):
   ```python
   warm_count = expert_cache_size
   num_blocks_available = block_pool.num_gpu_blocks - 1   # minus null block 0
   if warm_count > num_blocks_available:
       raise RuntimeError(...)
   ```
   So the minimum viable `--num-gpu-blocks-override` is `expert_cache_size + 1`, regardless of `num_moe_layers`. For OLMoE's 16-layer setup with `--expert-cache-size 8`, the floor is 9 blocks; the disjoint-id design that this replaced had a floor of `cache_size × num_layers + 1 = 129` (see `problems.md` for the rationale). The operator-facing failure for over-provisioned `--expert-cache-size` is "Reduce --expert-cache-size or increase memory budget."

5. **`manager.warm_up(expert_cache_size)`** — pre-loads the first `expert_cache_size` experts into the pool, one block_id shared across every layer per expert. This is for LRU residency tracking and realistic startup byte counts; the staging tensors already have every expert.

   Trade-off: every warmed `block_id` has all `num_moe_layers` layers as holders. If KV-allocation later claims one of those ids, the kv-broadcast invalidates expert mappings in **every** layer at once — accepted because warm-up is just startup seeding and the LRU reshapes from there.

Returns `block_pool.num_gpu_blocks` (the engine doesn't currently use the return value but the RPC signature reserves it).

---

## 4. Forward Path

```
FusedMoE.forward_native(hidden_states, router_logits)
  └─ if self._unified_pool: _forward_with_unified_pool(...)

_forward_with_unified_pool:
  1. router_logits → topk_weights, topk_ids
  2. needed = topk_ids.unique().tolist()
  3. manager.ensure_loaded(pool, needed):
       Pass 1: classify hits vs misses
       Pass 2: for each miss, _select_victim_block + assign + DMA-async
       Pass 3: wait_stream barrier
     Hits and misses both bump expert_lru to MRU
  4. swap w13_weight.data, w2_weight.data → staging tensors
  5. quant_method.apply(...)  (Triton kernel, full-width, unmodified)
  6. restore w13/w2 originals (try/finally)
  7. manager.release_pinned(pool)
  8. manager.end_forward_step()
```

`end_forward_step()` is called once per forward per layer. The `step` counter therefore increments by N (number of MoE layers) per token. That's harmless — recency comparisons in `_select_victim_block` are pairwise within the same `step` namespace, so the absolute rate doesn't matter.

---

## 5. Eviction Logic

`_select_victim_block(layer, needed_set)` — the heart of the unified pool. Three tiers:

**Tier 1 — free space from L's view.** Walks `free_block_queue` head→tail. Returns the first block where:
- `block_id not in layer.pinned_blocks`
- `layer.expert_of_block(block_id) is None` (no mapping in *this* layer)
- `cursor.block_hash is None` (no cached prefix)

The "no mapping in this layer" check (vs "no mapping in any layer") is the key insight. If layer 0 holds expert E_a at block 5, and layer 1 needs to claim block 5 for its own expert E_b, layer 1 is free to do so — layer 0's mapping at block 5 stays valid because layer 0's pool buffer at offset 5 isn't physically modified.

**Tier 2 — cold tail of L's mixed LRU.** Take the head (oldest) of `layer.expert_lru` and the head of `manager.prefix_lru`, skipping pinned/needed entries on each side. Compare their `step` values:
- If both exist: pick whichever has the smaller step. **Tie-break: prefix wins** (`oldest_prefix_step <= oldest_expert_step` favours the prefix entry on equal recency, ~line 561).
- If only one exists: pick it.
- If neither: raise (Tier 3 fail-loud).

The picked block is removed from `free_block_queue` and returned. The caller will (a) clear any prefix hash globally via `evict_prefix_hash`, (b) drop this layer's mapping if any via `_drop_layer_mapping`, (c) install the new mapping via `assign`, and (d) re-append to `free_block_queue` at the end so the KV path can still see the block.

**Tier 3 — fail loudly.** Pool exhausted. Tells the operator to lower `--max-num-batched-tokens` or raise `--num-gpu-blocks-override`.

---

## 6. Cross-Layer Coordination

Three places where state crosses the layer boundary:

**KV allocation broadcast** (`_on_kv_allocation`). Fires from `BlockPool.get_new_blocks`. KV writes are about to physically overwrite every layer's bytes at the chosen `block_id`s, so every layer's expert mapping at those blocks is now stale. Calls `_broadcast_drop_all_layers` for each block.

**Cached-prefix global eviction** (`_evict_prefix_globally`). Fires from `ensure_loaded` Pass 2 when an expert miss reuses a block that held a cached prefix hash. The hash is global (same hash at every layer), so once *any* layer overwrites its bytes the prefix is logically broken everywhere — regardless of whether other layers' bytes at that block changed. Calls `block_pool.evict_prefix_hash(block_id)`, which fires `_on_prefix_removed` and clears the hash.

**Expert-driven invalidation in L only** (`_drop_layer_mapping`). Fires from `ensure_loaded` Pass 2 when L's expert miss reuses a block that L itself was holding for a different expert. **Only L's mapping is dropped.** Other layers' mappings at the same block are untouched — their bytes haven't changed.

The `block_holder: dict[block_id → set[layer_idx]]` index tracks both *membership* ("which layers currently hold an expert at block B") and *eviction driver* ("which layers must be invalidated when B is reused for KV"). KV-driven invalidation iterates the full set; expert-driven invalidation only mutates the calling layer's entry. This distinction is the §7.5 / §7.1 lesson from the prior iteration.

---

## 7. Constraints and Required Flags

Validated at Stage 1, fail-fast with descriptive errors:

| Flag                          | Why                                                                 |
|-------------------------------|---------------------------------------------------------------------|
| `--expert-offload`            | The unified pool is built on top of expert-offload semantics.       |
| `--expert-unified-pool`       | The flag itself.                                                    |
| `--enable-prefix-caching`     | Without it, freed KV blocks never become prefix-LRU entries; KV-hot scenario can't hit. |
| `--enforce-eager`             | Variable-length miss DMAs cannot be captured in a CUDA graph.       |
| `--max-num-batched-tokens 1`  | Caps per-forward unique experts at top-k per layer; otherwise a long prefill batch could route to hundreds of distinct experts and exhaust pool pages mid-call. |
| TP=1, PP=1                    | Multi-rank coordination is not implemented in the MVP.              |

Plus the implicit constraint: BF16 attention with uniform `AttentionSpec` across layers (no MLA, no Mamba, no sliding-window).

---

## 8. Diagnostics and Observability

**Per-100-step stats** (auto). `manager.log_stats()` is called from `end_forward_step` every 100 steps. Per-layer one-liner:
```
UnifiedPool L0: hits=32296 misses=56 hit_rate=99.8% expert_pages=64 kv_prefix_pages=0
```
- `expert_pages = len(layer.expert_at_block)` — experts pinned in *this layer's* buffer.
- `kv_prefix_pages = len(manager.prefix_lru)` — global, same value on every layer's line.

**Trace mode** (`VLLM_UNIFIED_POOL_TRACE=1`). The actual trace format diverges from the plan §6.6 spec — what is emitted now, per forward per layer, all snapshots captured **before** mutation in `ensure_loaded`:

```
=== STEP <step> L<n> need=[E…] ===
UNIFIED CACHE L<n> occ <expert-ours>/<capacity> ours (expert-ours=…, expert-other=…, prefix=…, alloc-kv=…, pinned=…, free-pure=…)
UNIFIED EXPERT_LRU L<n> MRU→LRU [<count>]: E<eid>@p<page>#step<step>, …
UNIFIED PREFIX_LRU MRU→LRU [top 8 of <count>]: p<page>#step<step>, …
UNIFIED REQUEST L<n>: E<eid>,…
UNIFIED CLAIM page=<id> L<n> E<eid> cause=expert-L<n> tier=free-…           # only for Tier-1 free claims
UNIFIED EVICT page=<id> L<n> kind=expert E<eid> cause=expert-L<n> tier=…    # per Tier-2 eviction
UNIFIED EVICT page=<id> L=all kind=kv-prefix cause=… tier=…                 # per global prefix eviction
UNIFIED EVICT page=<id> L<m> kind=expert E<eid> cause=kv-alloc tier=kv-broadcast  # per KV-allocation invalidation
UNIFIED RESULT L<n> hits=[E<eid>@p<page>,…] misses=[E<eid>->p<page>(tier),…]
--- end L<n> ---
```

Plus prefix-LRU lifecycle lines (fired from BlockPool callbacks, not per-layer):
```
UNIFIED PREFIX_ADDED p<id> step=<step> size=<count>
UNIFIED PREFIX_REMOVED p<id> was_present=<yes|no> size=<count>
```

Notable divergences from plan §6.6:
- The plan said `CACHE` would be a "page→expert listing"; in practice `UNIFIED CACHE` emits only composition counts. The page→expert listing lives in `UNIFIED EXPERT_LRU` instead.
- The plan called the request line `NEED`; the implementation emits `UNIFIED REQUEST`.
- `EVICT` lines carry an extra `tier=…` field (Tier-1 `free-cross-layer-expert`, Tier-2 `expert-local`/`prefix-global`, KV-broadcast `kv-broadcast`).
- The plan §6.6 "hit/miss self-consistency" grep check therefore needs to compare `RESULT` HITs against `EXPERT_LRU`, not `CACHE`.

The variable is unregistered in `vllm/envs.py`, so vLLM warns "Unknown vLLM environment variable detected: VLLM_UNIFIED_POOL_TRACE" at startup — cosmetic only, the read uses `os.environ.get` directly.

**Shutdown summary**. `manager.shutdown_log()` exists but is not currently wired to a shutdown hook.

---

## 9. Known Limitations / Followups

These are tracked separately from boot bugs — they are gaps relative to the plan, not failures.

**Prefix-hit attention hook (plan §5, §7.6).** The plan calls for an attention-side hook (`manager.note_prefix_hits(layer_idx, block_ids)`) so prefix entries get bumped to MRU on every read. **Not yet implemented.** The current implementation only bumps prefix entries on `free_blocks` (i.e. when a request finishes using them). Without the read-time hook, the prefix LRU tracks claim-recency rather than use-recency for blocks held by long-lived requests. For the §6.6 acceptance test this is not yet caught because the prefix LRU is currently empty in single-request smoke tests.

**`VLLM_UNIFIED_POOL_TRACE` registration.** Add an entry in `vllm/envs.py` to silence the unknown-env warning. Cosmetic.

**`_maybe_init_expert_cache` early-return guard (plan §5).** The plan calls for `_maybe_init_expert_cache` to be a no-op when `_unified_pool_enabled`, so `prepare_communication_buffer_for_model` can't allocate a useless `ExpertCache`. The branch currently lives in `gpu_model_runner.py` instead, so this works in practice — but a defensive guard inside the method itself would be more robust against future caller order changes.

**Shutdown-log hook.** `manager.shutdown_log()` is defined but never called. Wire to engine shutdown to dump final stats.

**Multi-process serialization.** `engine/core.py` asserts `UniProcExecutor`. Going to multi-process would require pickling the live `BlockPool` (block array + linked-list free queue) across the executor boundary, or refactoring to a manager-on-each-worker design with cross-process sync.

---

## 10. Boot Bugs Encountered During Implementation

For posterity:

1. **Wrong attribute name on the user-specified flag.** Stage 1 was setting `cache_config.block_size_user_specified = True`; the actual field is `cache_config.user_specified_block_size`. The `hasattr(...)` guard returned False and the line was a no-op. `update_block_size_for_backend` then reset `block_size` to 16, and `num_gpu_blocks` came out as 13787 (= 1.8 GiB / 131072 bytes, a 16-token page) instead of 143 (= 1.8 GiB / 12.58 MiB, a 1536-token page). Fixed: use the correct attribute name unconditionally.

2. **Per-layer KV byte tensor not a multiple of page size.** vLLM sizes KV tensors to the per-layer share of available KV memory; for the OLMoE setup that was 1,807,089,664 bytes = 143.62 × 12,582,912. The original aliasing `raw_kv.view(torch.int8).reshape(-1, page_size_bytes)` failed on the non-multiple. Fixed: `raw_kv.narrow(0, 0, num_gpu_blocks × page_size_bytes)` slices to the addressable region; the slack at the tail is unaddressable by block IDs anyway and is unaddressable in the baseline too.

3. **`mark_recently_used` method that didn't exist.** `_forward_with_unified_pool` originally called `pool.mark_recently_used(needed_expert_ids)` after `release_pinned`. `UnifiedPool` has no such method — the bumps happen inside `ensure_loaded`. Fixed: replaced with `manager.end_forward_step()` to advance the global step counter.

4. **Venv sync gaps.** Six of the eight modified files weren't synced into `venv/lib/python3.12/site-packages/vllm/...`, so the boot saw a half-modified vLLM. Fixed by re-syncing all eight files.
