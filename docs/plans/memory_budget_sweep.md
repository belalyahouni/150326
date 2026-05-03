# Memory-budget sweep

## Goal

For each fixed per-layer memory budget M, show that:

M (per-layer total memory in blocks) = C + K  for static
                                     = pool   for unified
```

Static-only flag layout per cell:
1. The operator's static-config choice is a **trilemma** — picking prefix-
   tuned, middle, or expert-tuned splits each fails one phase of a mixed
   workload.
2. **Unified pool dominates static across the entire budget range**, with
   the gap largest at tight budgets where every block matters.

This subsumes the standalone static cache sweep and unified pool sweep
into one coherent comparison the dissertation can hang a chapter on.

## Memory-budget framing

The unified pool's page invariant — `expert_page_size == kv_block_size` —
means a static config with `cache=C, kv=K` consumes exactly `C+K`
blocks per layer of GPU memory. So matching budgets is clean:

```

```
--expert-cache-size C
--num-gpu-blocks-override K
```

(no `--gpu-memory-utilization`; the override pins KV directly so the
budget is exact, not a function of util/profile gating).

Unified-only flag layout per cell:

```
--expert-unified-pool
--expert-cache-size <init>
--num-gpu-blocks-override M
```

## Budget sweep

| M (blocks/layer) | Static splits (cache, kv) | Unified |
|---:|:---|:---|
| 16 | (8, 8), (12, 4) | pool=16, init=8 |
| 24 | (8, 16), (12, 12), (16, 8) | pool=24, init=12 |
| 32 | (8, 24), (16, 16), (24, 8) | pool=32, init=16 |
| 48 | (16, 32), (24, 24), (40, 8) | pool=48, init=24 |
| 52 | (16, 36), (24, 28), (40, 12) | pool=52, init=26 |
| 56 | (16, 40), (24, 32), (40, 16) | pool=56, init=28 |
| 64 | (20, 44), (40, 24), (64, 4) | pool=64, init=40 |

**M=56 and M=52 added retrospectively** to localise the cliff observed
between M=48 (unified P1=26.31 s — fails) and M=64 (unified P1=7.15 s
— wins). M=56 was the first crossing point (P1=7.13 s ✓); M=52 added
to localise the cliff more precisely. All extra cells use the same
`init = 0.5 × pool` heuristic as the M ≤ 48 cells.

**Floor on cache size**: OLMoE's router selects `top_k=8` experts per token, so a
forward pass requires at least 8 expert slots in cache. cache=4 or smaller
fails at engine init with `StopIteration` in `_initialize_kv_caches`. M=16
therefore only has 2 viable static splits ((8,8) and (12,4)); larger budgets
all use 3 splits.

The 3 static splits per budget represent "prefix-tuned / middle /
expert-tuned" within that budget. The unified init is the middle
split — operator-neutral starting point. (Convergence is initial-
condition-independent per Test 2A ablation.)

The M=64 row matches existing Test 2A static cells (cache=20/40/64
with util=0.3105, which auto-allocates KV to total ~64). Re-running
under the explicit `--num-gpu-blocks-override` regime gives a clean
budget-pinned comparison.

## Apples-to-apples verification

The matched-memory claim rests on the unified pool's page-aliasing
invariant: `expert_page_size == kv_block_size_per_layer`. For OLMoE-1B-7B
at `block_size=1536`:

```
expert page  = w13 + w2 = (2048·2048 + 2048·1024) × 2 bytes = 12 MB
kv block/layer = 1536 tokens × 16 heads × 128 dim × 2 (k,v) × 2 bytes = 12 MB
```

→ `1 expert page == 1 KV block per layer == 12 MB`. Therefore
`static(C, K) memory == unified(C+K) memory` to the byte.

To verify empirically per cell, the runner captures:

1. **`nvidia-smi memory.used`** at idle (after server ready, before the
   first bench request). Logged to
   `logs/budget${M}_<cell>_idle_mem.txt`.
2. **vLLM startup log lines** identifying:
   - `Model loading took X GiB` (same across all cells — just weights).
   - `ExpertCache: warmed C/64 experts` for static cells.
   - `GPU KV cache size: T tokens` (T / 1536 = KV blocks).
   - For unified: `num_gpu_blocks` reported by the engine.

Post-hoc, for each budget M we'll show a small table of measured idle
memory across the 4 cells. They should be within ±200 MB. If they're
not, the comparison isn't truly matched and we'd need to investigate
(probably static expert cache being allocated separately rather than
sharing the pool).

## Workload

Each cell runs the **Test 2A two-phase workload** on one long-lived
server (pool state carries between phases — that's the whole point):

- **Phase 1**: alternating prefixes, **5 prompts**, `--num-warmups 1`,
  `--custom-output-len 20`.
- **Phase 2**: random, **6 prompts**, `--num-warmups 0` (preserve pool
  state from Phase 1), `--random-input-len 256 --random-output-len 80`.

Save: `budget${M}_<config>_phase{1,2}_seed1.json` per cell.
`<config>` ∈ {`static_C${C}`, `unified_init${C}`}.

## Run budget

- 5 budgets × 4 cells/budget = **20 cells**
- Each cell = boot + 2 phases ≈ 4–5 min
- 2-GPU parallelism: 10 rounds × ~5 min = **~50 min wall time**

Trim option: drop M=48 (4 budgets × 4 cells = 16 cells, ~40 min).

## Expected outcome

For each budget, plot Phase 1 mean TTFT and Phase 2 mean TTFT for the 4
cells (3 statics + unified). Hypotheses:

- **At every budget**: the 3 static cells form a Pareto frontier where
  none dominates — prefix-tuned wins P1 but loses P2, expert-tuned the
  reverse, middle is mediocre at both.
- **Unified beats *all 3* static cells on at least one phase** at every
  budget, and likely on both phases at most budgets.
- **Tight budgets (M=16, 24)**: the static trilemma is sharpest. Tiny
  cache → expert thrashing on P2; tiny KV → KV starvation on P1. Unified
  has the most to gain here. Hypothesised gap: 2–5× on at least one phase.
- **Comfortable budgets (M=48, 64)**: gap shrinks. Both static and unified
  have room to fit most of what the workload needs. Unified should still
  win marginally (per the matching argument from earlier tests).
- **Catastrophic point**: at M=16, even unified may struggle since the
  hot expert set + KV demand jointly exceed the pool. Characterising the
  breakdown point is itself a useful result.

## What this proves

A single 2D table of (memory budget × config) showing unified is the
dominant configuration at every operating point in the dissertation's
target regime. Combined with the existing tests:

- Test 0 → correctness
- Tests 1A/1B → convergence from wrong start
- Test 2A/2B → workload shift
- **Memory-budget sweep → robustness across budgets and beats every
  static config at matched memory**

That's a complete defensible chapter.
