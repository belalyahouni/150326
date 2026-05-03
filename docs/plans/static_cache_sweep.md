# Static cache-size sweep

## Goal

Establish that **no static cache size dominates the unified pool** on either
workload regime. Existing results compare unified against 3 hand-picked static
configs (cache=20, 40, 64). This sweep tests the entire static Pareto frontier
so reviewers can't say the comparison was cherry-picked.

## Method

Run static (no unified pool) at **8 cache sizes**: 8, 16, 24, 32, 40, 48, 56, 64.

Each cell uses the same server config as Test 1A/1B static cells:
`util=0.3105`, `block_size=1536`, `--expert-offload`. Only `--expert-cache-size`
varies.

For each cell, run two benches against the same server (clean state per cell):

- **Workload A (KV-hot)** — alternating prefixes, **5 prompts** (shortened from
  20: prefix-cache hits dominate after the first 2 cold prefills, so 5
  is enough to demonstrate the steady-state TTFT).
- **Workload B (Expert-hot)** — random, **8 prompts** (unchanged).

Each cell saves two JSONs: `sweep_static_cache${N}_1A_seed1.json` and
`sweep_static_cache${N}_1B_seed1.json`.

## Expected outcome

Plot TTFT vs cache size for both workloads:

- Workload A: TTFT should be *flat-ish* for cache=8..56, then catastrophic at
  cache=64 (KV starvation). Hypothesis: alternating workload only needs ~15
  hot experts/layer, so any cache ≥ 16 covers expert demand. Cache=64 starves
  KV → full prefill every request.
- Workload B: TTFT should *decrease monotonically* with cache size — random
  workload activates more experts, so larger cache = fewer DMA misses.

The Pareto frontier of static configurations: the best static for A is
*different* from the best static for B. **No single static cache size minimises
both.** Unified's headline numbers (1A: 1.81 s, 1B: 3.05 s) should sit *below*
the entire frontier on both axes — proving unified isn't tied for second on
either workload, but ahead on both.

## Run budget

- 8 cells × 2 benches = 16 bench calls
- 8 server boots (one per cache size)
- 2-GPU parallelism: 4 rounds, ~5 min each → **~20 min wall time**

## Comparison data

Existing unified-from-bad results (controlled workload, no suffix, blocks=68):

- 1A unified mean TTFT: **1.81 s** (cache=64 init, beats best static-good 2.03 s)
- 1B unified mean TTFT: **3.05 s** (cache=16 init, beats best static-good 3.32 s)

If the sweep shows static cells stay above these numbers across all 8 cache
sizes, the dominance claim is empirically airtight.
