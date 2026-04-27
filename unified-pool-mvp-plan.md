# Unified Per-Layer Page Pool — MVP Plan

> **Scope: Phase 1 only.** This plan covers the unified-pool MVP that proves the LRU dynamics on benchmarks. Phase 2 (modifying the Triton `fused_moe_kernel` to read expert weights directly from pool pages, eliminating the staging tensor) is **not in scope** and is not a follow-up the agent should attempt or stub for. See §3 for the Phase 1 vs Phase 2 distinction and §4 for the explicit out-of-scope list.
>
> Starting point: `origin/main`, which already has working `--expert-offload` / `--expert-cache-size`. Implement on a fresh branch off `origin/main`. Sync edits into the venv (`venv/lib/python3.12/site-packages/vllm/...`) after every change — `vllm` is installed as a regular package, not `-e`. The agent should design data structures, function signatures, and sync code itself; this plan describes intent and constraints, not implementation.

---

## 1. Goal

Static GPU VRAM partitioning between KV cache and expert cache is suboptimal: the right split depends on the workload, which the operator doesn't know in advance. The unified pool replaces both regions with **one byte pool per layer**, governed by a single LRU that decides — across KV blocks *and* expert pages — what to evict under memory pressure. Regardless of how the operator sets the initial expert allocation, the LRU should converge on the split a workload-tuned static config would have picked.

Two scenarios prove dynamic rebalancing in both directions:

- **KV-hot.** Long chat with shared prefix, hits few experts. A balanced static partition can't fit the full conversation KV, so prefix reuse is partially lost to recompute. A workload-tuned static config gives more space to KV, fits everything, and saves the recompute. The unified pool — starting from a balanced or expert-heavy split — should evict cold experts to free KV space and match the tuned TTFT.
- **Expert-hot.** Diverse prompts with no shared prefix, each hits most/all experts. A balanced static partition fits only a fraction of experts, so every forward swaps between CPU and GPU. A workload-tuned static config gives more space to experts, fits all of them, and avoids the swaps. The unified pool — starting from a balanced or KV-heavy split — should evict cold KV to keep all hot experts resident and match the tuned TPOT.

In both cases the comparison is **unified-pool-with-arbitrary-initial-split vs. static-with-workload-tuned-split**: the win is achieving the tuned-config performance without the operator having to know the workload. Benchmark commands live in `motivation-benchmarks.md` (may need updating to match this framing).

---

## 2. Core Mechanic — One Mixed LRU Per Layer Over a Shared Page Namespace

The `block_id` namespace is **global** — one numbering shared across all layers, locked in by vLLM's single block table. The physical buffers it indexes into are **per-layer**: a `block_id` is a logical lease pointing at the same offset in every layer's pool buffer. KV writes are synced — a `block_id` allocated to active KV commits the same logical token's bytes at that offset in every layer at once. Expert writes are independent — an expert DMA at layer `L`'s `block_id 47` touches only `L`'s offset 47.

On top of this namespace, **each layer owns one LRU containing expert pages and KV blocks as peers**. The LRU ranks every evictable thing layer `L` currently holds at its offsets — `L`'s expert pages and `L`'s view of cached-prefix KV blocks. KV and expert entries compete on equal footing, ordered by `L`'s recency alone. There is no fixed expert/KV split inside a layer: the recency-driven eviction *is* the dynamic rebalance the unified pool exists to provide. Active-sequence KV is pinned by the scheduler, lives outside the LRU, and is never an eviction candidate.

Cached-prefix recency is naturally **global**, not per-layer — attention reads `block_id B` at every layer in lockstep within one forward, so per-layer timestamps for prefix entries are identical. The "per-layer mixed LRU" can therefore be implemented as a per-layer expert-recency map merged at query time with one shared prefix-recency map, rather than N redundant copies of the prefix ordering.

When `L` misses on a new expert, `L` walks its own LRU from cold to hot and picks the coldest non-pinned entry as victim:

- **Victim is one of `L`'s experts.** Drop `L`'s expert map entry at that `block_id`; DMA the new expert into `L`'s slot. Other layers' state at the same `block_id` is untouched — physical bytes elsewhere don't change, mappings elsewhere stay valid.
- **Victim is a cached-prefix KV block.** Clear the prefix hash from `FreeKVCacheBlockQueue` so future requests can't match against now-stale data, remove the corresponding entry from every layer's LRU, and DMA `L`'s new expert into `L`'s slot at that `block_id`. The drop is global by necessity — the cached prefix is a global concept (same hash at every layer), so once `L` overwrites its bytes the prefix is broken everywhere; from `L`'s perspective the prefix was the coldest thing it held, and dropping it costs the same prefix-cache miss everywhere.

When the **scheduler** needs an active KV block, it pops HEAD from `FreeKVCacheBlockQueue` as today. If the chosen `block_id` held an expert in any layer or sat as a cached-prefix entry in any layer's LRU, **every** affected layer's LRU and mappings are updated, because the impending KV writes physically overwrite every layer's bytes at that `block_id`. This is the only path that broadcasts across layers.

Cross-layer side effects come in two narrow forms: (a) scheduler KV-allocation invalidates expert mappings and prefix entries at the chosen `block_id` across every layer, because impending KV writes physically overwrite all layers' bytes there; (b) expert-side cached-prefix eviction clears the prefix hash globally and removes the prefix entry from every layer's LRU, because the cached prefix is itself a global concept. Everything else — `L`'s expert map updates, `L`'s expert-LRU entries — is local to `L`. The scheduler-side broadcast is implemented via a callback on `BlockPool.get_new_blocks` (fires after `_maybe_evict_cached_block`, before return). **Expert-side allocation does not flow through `get_new_blocks`** — the manager picks the victim from `L`'s LRU and mutates state in place; if the victim is a cached-prefix block, the manager invokes a separate `BlockPool` API to clear the hash and broadcast the prefix-entry removal across layers. The previous-iteration bug (§7.1) was making *every* expert miss broadcast — collapsing per-layer recency into one global LRU.

---

## 3. Phase 1 Simulation — Why There's a Staging Tensor

Phase 2 would modify the Triton `fused_moe_kernel` to take a page-table argument and read expert weights directly from pool pages, with no duplication. **Phase 2 is out of scope.** Phase 1 (this MVP) keeps the kernel unmodified and fakes the same effect with a static staging tensor: each layer keeps `staging_w13[num_experts, *w13_shape]` and `staging_w2[num_experts, *w2_shape]` on GPU, filled **once at startup** with all experts from CPU and never modified. Every forward swaps the layer's `w13_weight.data`/`w2_weight.data` to point at staging and runs the unmodified kernel — full-width, no `topk_ids` remap, no per-step gather.

The pool still does **real CPU→GPU DMA on expert misses** so PCIe miss latency is measured honestly. Those DMAs land in pool pages the kernel never reads.

Two sync rules, not symmetric:

- **`wait_stream` after miss DMAs is mandatory.** The kernel doesn't physically depend on the DMAs (it reads staging), but a Phase-2-style kernel *would*. Without the barrier, a miss-heavy forward launches the GEMM artificially early and the simulation lies. The barrier forces GEMM start time to `max(DMA completion times)`, matching the cost a real pool-reading kernel would pay.
- **No post-GEMM sync needed before freeing pool pages.** The kernel never reads pool pages, so there's no kernel-vs-free race. Release immediately after the kernel returns.

Latency fidelity:

|                              | Phase 2 (out of scope)  | Phase 1 (this MVP) |
|------------------------------|-------------------------|--------------------|
| CPU→GPU DMA on miss          | yes                     | yes                |
| GPU→GPU gather before GEMM   | no                      | no                 |
| GEMM input                   | pool pages              | static staging     |
| Per-layer post-GEMM sync     | no                      | no                 |

Only the GPU memory footprint differs (Phase 1 has redundant staging; Phase 2 wouldn't).

**Memory accounting is essential for fair benchmarks.** Staging is allocated *before* `determine_available_memory` runs, so the profiler subtracts it from the budget it reports to the scheduler. Baseline and unified runs then compete on equal *dynamic* memory — the experimental variable is "static vs. dynamic split", not total footprint. Print staging overhead and adjusted budget at startup so reviewers can audit.

---

## 4. MVP Scope

In scope:
- BF16, MoE-only.
- Single process / `UniProcExecutor`. Assert at startup. No TP > 1 or PP > 1.
- Single attention layer group (no MLA, no Mamba, no sliding window).
- Page size = one expert slot. `block_size_tokens` is derived so a KV page's `page_size_bytes` exactly equals one expert's `w13_bytes + w2_bytes`. This makes the pool buffer aliasing work with no attention-backend changes.
- KV-driven cross-layer expert collateral is **accepted** (synced KV writes physically overwrite every layer's bytes at a page).

Required flags (validate at startup, fail loudly if missing):
- `--enable-prefix-caching` — without it, freed KV blocks don't stay in the free queue with their hashes and the KV-hot scenario can't hit.
- `--enforce-eager` — cache misses trigger variable-length DMAs that can't be captured in a CUDA graph.
- `--max-num-batched-tokens 1` — caps per-forward unique-expert demand at `top_k` per layer. Without it a long prefill batch can route to hundreds of distinct experts in one forward and exhaust available pool pages mid-call (every page pinned, no allocation possible).

Note on `--expert-cache-size` under the unified pool: it controls **only the initial warm-pool occupancy** (how many expert pages each layer pre-loads at startup, so the first forward isn't full miss DMAs). The LRU reshapes the split from there based on actual workload pressure. The agent should not enforce, recompute, or treat this number as steady-state — the benchmark scenarios deliberately set it "wrong" to show the LRU recovers.

Out of scope (Phase 2+):
- Quantization (FP8, INT8, AWQ, GPTQ) — needs separate scale handling and a different kernel.
- Hybrid models, MLA, Mamba, sliding-window attention.
- Multi-process, expert parallelism, runtime-adaptive cache sizing, heuristic warm-up.
- Eliminating the staging tensor (requires modifying `fused_moe_kernel`).
- Sub-page KV granularity.

---

## 5. Implementation Envelope

The agent owns concrete data structures, function signatures, and stream/sync code. This section names the touchpoints so the agent isn't grepping blindly.

### New module — `vllm/model_executor/layers/fused_moe/unified_pool.py`

A per-layer pool object holding: the aliased pool byte buffer (a view onto the layer's KV raw tensor — **do not allocate a second buffer**), the layer's mixed LRU state (expert pages + cached-prefix KV blocks), the static staging tensors, the CPU-pinned source-of-truth tensors, hit/miss counters.

A manager that owns: the `BlockPool` reference, the cross-layer "which layers hold an expert at page P" index, the dedicated transfer stream, the on-allocation callback, per-forward pin/release bookkeeping, log-stats output. The existing `expert_cache.py` stays unchanged for the non-unified path.

### Config and CLI — `vllm/config/offload.py`, `vllm/engine/arg_utils.py`

Add `expert_unified_pool: bool` and `--expert-unified-pool`, mirroring the existing `--expert-offload` plumbing. Validate at startup that `expert_offload`, `enable_prefix_caching`, `tensor_parallel_size==1`, `pipeline_parallel_size==1` are all satisfied, and that the model architecture is on an explicit allow-list.

### BlockPool hook — `vllm/v1/core/block_pool.py`

Add an on-allocation callback list and a register method. Fan out callbacks at the end of `get_new_blocks`, after `_maybe_evict_cached_block` (so the prefix-cache hash is already cleared) and before the block is returned. Run regardless of `enable_caching`. **Do not** fire from `touch()` — touched blocks are already valid KV with a prefix-cache hash; they cannot hold a current expert mapping. Don't add a `bump` method — `free_blocks` already appends at TAIL.

### FusedMoE integration — `vllm/model_executor/layers/fused_moe/layer.py`

Add a `_unified_pool` attribute alongside the existing `_expert_cache`. In `forward_native`, dispatch to a new `_forward_with_unified_pool` before the existing expert-cache path. The unified path:
1. Dedupe `topk_ids` to needed expert ids (`topk_ids.unique().tolist()`).
2. Ask the manager to ensure them loaded — pins hits, allocates+DMAs misses, applies the `wait_stream` barrier.
3. Swap `w13_weight.data`/`w2_weight.data` to point at the layer's static staging tensors. **Leave `global_num_experts` unchanged** (staging is full width). **Do not remap `topk_ids`** — global ids index correctly into full-width staging.
4. Run `quant_method.apply` inside try/finally that restores the originals.
5. Ask the manager to release pinned pages (no `cuda.synchronize()` needed).

The current `_maybe_init_expert_cache` does CPU-pinning inline. Extract a `move_experts_to_cpu()` helper from those lines so the unified setup can reuse the same logic.

### Worker setup — `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu_worker.py`

Branch the existing post-load expert-offload hook: when `expert_unified_pool` is on, run the new init path instead of `_maybe_init_expert_cache` per module. The init runs in **two stages**, separated by vLLM's existing memory-profile + KV-cache-allocation flow (this is initialization sequencing inside Phase 1 — unrelated to the project-level Phase 1 / Phase 2 distinction):

**Stage 1 — pre-profile (during/right after `load_model`):**
- For each `FusedMoE` module: move experts to CPU-pinned memory, record metadata (expert slot bytes, w13/w2 shapes, dtype, num_experts, layer index).
- Compute `block_size_tokens = expert_slot_bytes // bytes_per_token_per_layer` from the first attention layer's KV spec. Sanity-check: `expert_slot_bytes % bytes_per_token_per_layer == 0` and `block_size_tokens` is a multiple of the kernel's supported block size (typically 16). Round/pad if not.
- Mutate `vllm_config.cache_config.block_size = block_size_tokens` and mark it user-specified.
- Allocate the per-layer staging tensors **here**, before `determine_available_memory` runs, and fill them with all experts from CPU. This is what keeps the simulated GPU budget honest (§3).

**Stage 2 — post-pool-init (after the engine constructs the scheduler):**
- The engine hands the worker the scheduler's `BlockPool` via a new collective RPC.
- Construct per-layer pool objects, aliasing each layer's KV byte tensor as the pool buffer (no new allocation).
- Wire the manager into each `FusedMoE` module, register the cross-layer callback on `BlockPool`.
- Warm the first `expert_cache_size` experts per layer **in the pool** (not staging — staging already has every expert; this is for LRU residency tracking and realistic startup bytes).
- Assert `expert_cache_size * num_moe_layers <= num_gpu_blocks - 1` (account for `BlockPool`'s null block at index 0). If it fails, **abort with a clear error** — do not silently shrink.

Expose two thin RPC passthroughs on the worker (`setup_unified_pool`, `get_unified_pool_block_count`) for engine-side dispatch.

### Engine wiring — `vllm/v1/engine/core.py`

After `Scheduler(...)` is constructed (so `kv_cache_manager.block_pool` exists), call `collective_rpc("setup_unified_pool", block_pool)` if the unified pool is enabled. Assert `UniProcExecutor` here too — the design assumes uniprocess; multi-process serialization is not handled.

---

## 6. Verification

Work through these in order; do not declare done until every step passes.

1. **Imports compile** — both source-tree and venv-installed paths.
2. **Server boots** with OLMoE, `--expert-offload --expert-unified-pool --expert-cache-size 12 --enable-prefix-caching --enforce-eager --max-num-batched-tokens 1`. Startup log shows `# GPU blocks`, the derived `block_size`, staging overhead, and the adjusted available memory.
3. **Correctness smoke test.** One short completion. Compare tokens to the same model with `--expert-offload` only (no unified pool). Should match modulo BF16 nondeterminism.
4. **KV-hot scenario.** Run unified with an expert-heavy initial `--expert-cache-size` (the "wrong" static split for this workload). TTFT must drop to within ~15% of the static baseline configured with the workload-tuned (KV-heavy) split.
5. **Expert-hot scenario.** Run unified with a KV-heavy initial `--expert-cache-size` (again, the "wrong" static split). TPOT must drop to within ~15% of the static baseline configured with the workload-tuned (expert-heavy) split.
6. **Per-layer invariant — architectural acceptance test.** Env-gated trace logging (`VLLM_UNIFIED_POOL_TRACE=1`). Per layer-forward emit three lines, **all snapshots captured before any state mutation**:
   - `CACHE` — page→expert listing
   - `NEED` — unique experts requested
   - `RESULT` — hits as `E<eid>@p<page>`, misses as `E<eid>->p<page>`

   Per eviction emit one line:
   - `EVICT page=<id> L<n> kind={expert E<eid> | kv-prefix} cause={kv-alloc | expert-L<m>}`

   Two grep checks the agent **must** run:
   - **Hit/miss self-consistency.** Every HIT page in a `RESULT` must appear in the same forward's preceding `CACHE`; every MISS page must not. Failure means the trace was captured after mutation.
   - **No expert-driven cross-layer expert eviction.** For every `EVICT` line where `kind=expert E<…>` and `cause=expert-L<m>`, the evicted-layer field `L<n>` must equal `L<m>`. Any mismatch means an expert miss in one layer wiped another layer's expert — that's the previous-iteration bug (§7.1). Do not proceed. `kind=kv-prefix` lines are *allowed* to fan out across layers (cached prefix is global, so the eviction is by design global) and `cause=kv-alloc` lines are expected to wipe across layers.
7. **Stats and dynamics.** Manager prints per-layer hits/misses every ~100 steps and at shutdown. Pool composition (kv / expert / empty / pinned) logged at the same cadence — should shift toward KV-heavy under the KV-hot scenario and expert-heavy under the expert-hot scenario. **This is the dissertation evidence.**

---

## 7. Lessons From Prior Iteration

Failure modes that are not obvious from reading the design.

1. **Single-global-LRU drift.** Previous attempt routed every expert miss through `BlockPool.get_new_blocks` and let the cross-layer callback wipe every layer's mapping at the returned page. An expert miss in `L_m` then evicted experts in `L_n ≠ L_m`, even though `L_n`'s pool buffer was physically untouched. The §6 step 6 grep check is what catches this. The §2 callback gating is what prevents it.

2. **Trace captured post-mutation state.** `expert_to_page` was read at the *end* of the load function, after misses had inserted their entries; the displayed `CACHE` then included pages that had just been assigned. The hit/miss self-consistency check would have flagged this. Capture all trace snapshots *before* mutating state, not after.

3. **Warm-up under-sized.** With small `--num-gpu-blocks-override`, requesting `expert_cache_size × num_moe_layers` warm-up pages can exceed the pool. Earlier-warmed pages get popped to satisfy later warm-up calls and every layer ends up empty before serving starts. Assert and abort, don't silently shrink.

4. **Top-k vs needed-experts confusion.** `top_k=8` does not mean 8 unique experts per forward — multiple tokens routing to the same expert dedupe upstream. Hit/miss counts are over `topk_ids.unique()`, not over `top_k`. Don't treat "fewer than top-k entries in NEED" as a bug.

5. **The "page → which layers hold an expert here" index is two concerns, not one.** It tracks both *membership* ("which layers currently hold an expert at page P") and *eviction driver* ("which layers must be invalidated when P is reused"). KV-driven invalidation iterates the full set; expert-driven invalidation only touches the calling layer's entry. Keep this distinction explicit at every call site.

