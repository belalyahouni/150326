# Unified pool trace levels

`VLLM_UNIFIED_POOL_TRACE` now has **three levels** instead of two. Resolved once at module load (`vllm/model_executor/layers/fused_moe/unified_pool.py:46-56`); the gate is a single attribute read after that, so the off path is effectively free.

There is a **separate** `VLLM_EXPERT_CACHE_TRACE` env var that gates the static expert-offload path's per-step print (`ExpertCache: needed=... hits=... misses=...`). Off by default; set to `"1"` to enable. Same module-load-cached pattern. The two flags are independent because expert-cache and unified-pool are mutually exclusive code paths — only the relevant one is active at a time. **Leave both unset for any latency-sensitive run.**

| `VLLM_UNIFIED_POOL_TRACE` | Lines emitted | Use for |
|---|---|---|
| unset / `"0"` | none | **Latency-sensitive runs.** TTFT/TPOT measurement. |
| `"1"` | essential only — see table below | **Mechanism evidence runs.** Pool composition + eviction tier counts for dissertation figures. |
| `"2"` | essential + per-step verbose dumps | Debugging only. Do not use for any timed run. |

## What each level emits

### Level 1 — essential

| Line prefix | When it fires | What it tells you |
|---|---|---|
| `UNIFIED CACHE L<n> step=<S> occ A/B ours (expert-ours=A, expert-other=X, prefix=P, alloc-kv=K, pinned=Pn, free-pure=F)` | Once per layer per forward step | Per-layer pool composition. **The right-axis source for the overlay figure.** Now self-contained (`step=` lets you align to bench timestamps without grepping nearby lines). |
| `UNIFIED EVICT page=<bid> L<n> kind=expert E<eid> cause=<...> tier=<...>` | When an expert mapping is dropped (cross-layer reclaim, expert miss eviction, KV broadcast) | Eviction-tier counts. `tier` ∈ `prefix-global`, `expert-local`, `kv-broadcast`, etc. |
| `UNIFIED EVICT page=<bid> L=all kind=kv cause=...` | When a block is fully claimed for KV (all-layer broadcast) | KV-allocation events. |
| `UNIFIED KV_CLAIM page=<bid> tier=<...>` | When the new KV-side selector picks a victim page | `tier` ∈ `truly-free`, `kv-evicts-expert`, `kv-evicts-prefix`. The KV-side counterpart to expert-side EVICT lines. |
| `UNIFIED PREFIX_ADDED p<bid> step=<S> size=<N>` | When a released page enters the prefix LRU | Prefix lifecycle. |
| `UNIFIED PREFIX_REMOVED p<bid> was_present=<bool> size=<N>` | When a prefix entry leaves the LRU | Prefix lifecycle. |

### Level 2 — verbose (everything in level 1 plus)

| Line prefix | Why it's verbose-only |
|---|---|
| `=== STEP <S> L<n> need=[...] ===` | Per-step header. Step number is already on `UNIFIED CACHE`; this is just a separator. |
| `UNIFIED EXPERT_LRU L<n> MRU→LRU [N]: E0@p0#step12, ...` | Full per-layer expert LRU dump every forward step. Huge string ↔ measurable latency hit. Useful for debugging expert recency, not needed for figures. |
| `UNIFIED PREFIX_LRU MRU→LRU [top 8 of N]: p0#step12, ...` | Top-8 prefix LRU snapshot every forward step. Useful for debugging, not for figures. |
| `UNIFIED REQUEST L<n>: E1,E5,E12,...` | List of experts the router asked for this step. |
| `UNIFIED RESULT L<n> hits=[E1@p0,...] misses=[E5->p3(free-pure),...]` | Per-step hit/miss breakdown. Eviction tier counts are already captured by `UNIFIED EVICT` and `UNIFIED KV_CLAIM`; this is duplicate signal. |
| `UNIFIED CLAIM page=<bid> L<n> E<eid> cause=... tier=free*` | Trace for free-tier expert claims (where no eviction line fires). Useful for "where did this expert land", not for figures. |
| `--- end L<n> ---` | Per-layer separator. |

## Volume comparison

Single 3000-token prompt, OLMoE-1B-7B, 16 layers:

| Level | Lines emitted | Approximate size |
|---|---|---|
| 0 (off) | 0 | 0 bytes |
| 1 (essential) | ~48k (16 layers × 2992 steps × CACHE + a handful of EVICT/KV_CLAIM) | ~6 MB |
| 2 (verbose) | ~330k+ | ~40 MB+ |

Level 1 is ~7× smaller than the previous "trace=1" behaviour, with no loss of dissertation-relevant data.

## How the dissertation testing flow uses this

The plan (`dissertation_results_plan.md`) needs both clean latency numbers AND pool-composition timelines for the overlay figures. These have conflicting requirements: the former wants zero log overhead; the latter needs the CACHE/EVICT/KV_CLAIM lines.

**Two-pass workflow per unified cell:**

1. **Latency pass** — `VLLM_UNIFIED_POOL_TRACE` unset. Run the bench command; record TTFT/TPOT from the bench JSON. No trace overhead, no log file growth from unified-pool prints.
2. **Trace pass** — `VLLM_UNIFIED_POOL_TRACE=1`, **same workload, same seed**. Server log captures CACHE / EVICT / KV_CLAIM / PREFIX_*. Latency numbers from this pass are *contaminated* by I/O and discarded; only the trace lines are kept.

The two passes are deterministic at fixed seed, so the per-step pool composition from pass 2 aligns to per-request timestamps from pass 1 for the overlay figure.

**Static cells** never need trace — they have no unified pool to instrument. Single pass.

## Required artifacts per unified cell

From the latency pass:
- `results/<cell>_seed<N>.json` — bench output with per-request TTFT/TPOT.

From the trace pass:
- `logs/<cell>_seed<N>_trace.log` — server stderr with `VLLM_UNIFIED_POOL_TRACE=1`. Parse for:
  - `UNIFIED CACHE` lines → per-step `(expert-ours, prefix, free-pure)` per layer (overlay right-axis).
  - `UNIFIED EVICT ... tier=...` and `UNIFIED KV_CLAIM ... tier=...` lines → tier-count tables (appendix).

## Parsing pointers

Single regex `^.*UNIFIED CACHE L(\d+) step=(\d+) .*expert-ours=(\d+).*prefix=(\d+).*alloc-kv=(\d+).*free-pure=(\d+)` extracts everything needed for the overlay's right axis.

For tier counts, just `grep -c 'tier=<name>' <log>` per tier name — no regex needed.

## What was removed

Nothing was deleted. All level-2 (verbose) lines still emit when `VLLM_UNIFIED_POOL_TRACE=2`. The change is purely about gating: essential lines respond to `=1`, verbose lines respond to `=2`.

## Source

All changes are in `vllm/model_executor/layers/fused_moe/unified_pool.py`:
- Module-level constants: lines 33-56.
- `_trace_pre_mutation` split (CACHE essential, rest verbose): around line 920.
- `CLAIM` and `RESULT` gates demoted to `_TRACE_VERBOSE`: lines ~599 and ~617.

Block-pool (`vllm/v1/core/block_pool.py`) is unchanged — it has no trace prints of its own.
