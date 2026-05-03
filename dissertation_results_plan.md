# Dissertation Results — Test Plan

Two experiments comparing **static expert offloading** against the **unified pool**
on OLMoE, plus a single-seed correctness check that closes the "does the pool
corrupt outputs?" reviewer question. Each test uses the same hardware (1× L40,
46 GB), the same model (`allenai/OLMoE-1B-7B-0924-Instruct`), and the same shared
GPU budget — static cells use `--gpu-memory-utilization 0.3105`, unified cells
use `--num-gpu-blocks-override 68` (no `--gpu-memory-utilization`; the override
both pins the pool size and is sufficient for vLLM's allocator). Across all
tests, the only experimental variable between cells in the same workload is
`--expert-cache-size` (and whether `--expert-unified-pool` is set).

The chain of evidence:
0. **Output consistency** (Test 0): unified pool produces the same tokens as
   static for the same prompt — closes the correctness question.
1. **Convergence from wrong start** (Test 1): unified pool catches up to the
   workload-tuned static, even when initialised at the operator-bad value.
2. **One-shot workload shift** (Test 2): static can win one phase or the other but
   never both; unified wins both.

For Tests 1 and 2 every data point is reported as **mean ± std over 3 seeds**
(`--seed 1, 2, 3`). Single-seed numbers are noisy on this hardware; 3 seeds is the
floor for the dissertation results to be defensible.

---

## Mechanism evidence is essential, not appendix

Pool composition over time is the **direct evidence** that the unified pool's
split is reshaping. For every unified cell in Tests 1 and 2:

- Parse `UNIFIED CACHE L<n> step=<S> ...` lines from the server log (level-1
  trace) to extract per-step `(expert-ours, prefix, free-pure)` page counts
  for each of the 16 layers. Each line is self-contained (the explicit `step=`
  field is what enables alignment to per-request bench timestamps).
- Average across layers per step, or pick a representative layer (layer 0 is
  fine if averaging muddies the signal).
- **Overlay the page-count line chart on the per-request TTFT/TPOT timeline.**
  Same x-axis (request index, mapped from step number via the bench's
  per-request timestamps), TTFT/TPOT on left y-axis, page counts on right
  y-axis. The reader sees the *latency converging* and the *split shifting*
  on the same chart — visual proof that one causes the other.

This overlay is the headline figure for both Tests 1 and 2. It's not an
appendix; it's the central evidence.

### Two-pass workflow per unified cell (mandatory)

`VLLM_UNIFIED_POOL_TRACE=1` adds ~6 MB of log writes per 3000-token prompt —
not free, and **contaminates latency measurement**. So every unified cell is
run **twice** at the same seed:

1. **Latency pass** — `VLLM_UNIFIED_POOL_TRACE` **unset**. Bench JSON is the
   only artefact kept. TTFT/TPOT here is clean.
2. **Trace pass** — `VLLM_UNIFIED_POOL_TRACE=1`, same workload, **same seed**.
   Server log is the only artefact kept. Trace pass's bench JSON is discarded
   (latency contaminated by I/O).

Determinism at fixed seed is what makes the alignment work: pass 1's per-request
timestamps and pass 2's per-step `UNIFIED CACHE step=<S>` lines describe the
same forward sequence, so step → request mapping is exact.

Static cells need only one pass (no unified pool to instrument).

See `unified_pool_trace_levels.md` for what level 1 vs level 2 emits and why
level 2 (`=2`) is debug-only — never use for any timed run.

---

## Preconditions

Two things must be in place before running anything:

1. **Code fix to `layer.py`.** The `_maybe_init_expert_cache` method was
   allocating a 12.3 GB ExpertCache on GPU even when `--expert-unified-pool`
   was active (the MVP plan §5 had specified this guard but it never made it
   into the code). Without the fix, unified runs are silently double-charged
   for memory. The fix is a 2-line early-return in
   `vllm/model_executor/layers/fused_moe/layer.py` (and synced into
   `venv-phase-2/`):
   ```python
   if self._expert_cache is not None or self._expert_cache_size <= 0:
       return
   if self._unified_pool_enabled:    # added
       return
   ```

2. **Prompts regenerated at `PREFIX_TOKENS = 3072`, no random suffix.** With
   block_size=1536, prefixes must be a clean multiple of the block size:
   3,072 = 2 × 1,536 keeps the prefix block-aligned. Each prompt is the
   raw repeated-character prefix (Candidate C from `expert_variety_test.md`)
   with **no random suffix**. Why no suffix:
   - The prior random-suffix variant produced cumulative expert variety
     across 20 requests (decode + suffix tokens activate novel experts)
     that pushed every cached expert past every prefix block on the
     mixed-LRU step counter — yielding `tier=prefix-global` evictions in
     the unified pool's blocks=65 stress test (see "Risks" §6 below).
   - Stripping the suffix keeps each prefill at 11–20 hot experts/layer
     (per the variety probe), leaving the rest of the cached experts
     genuinely cold (step ≈ 0). Tier 2 LRU comparison then cleanly picks
     `expert-local` over `prefix-global` because cold experts ≪ prefix step.
   - This is the *controlled* workload variant: it isolates the unified
     pool's reshape behaviour from the LRU-policy artefact. A future
     "realistic-workload" variant can re-introduce decode/suffix variety
     when comparing against a value-weighted LRU (future work).

   Each prompt is therefore one of two strings (10× "a"-prefix, 10× "b"-prefix
   alternated). Update `make_alternating_prompts.py` and re-run it before any
   benchmark. Per-request KV footprint stays at 3 blocks (2 prefix + 1 active
   decode block); the static-bad budget of 4 KV blocks (`util=0.3105`)
   preserves 1 block of scheduler slack as before.

## Shared server config

Common flags for **every** run (static and unified):

```
--expert-offload
--enable-prefix-caching
--enforce-eager
--trust-remote-code
--max-model-len 4096
--max-num-batched-tokens 1
--no-async-scheduling
--attention-backend TRITON_ATTN
--block-size 1536
```

`--block-size 1536` is mandatory: the unified pool's block size is dictated by
the architecture (must equal expert-slot bytes per layer for pool aliasing),
so static cells must match it for apples-to-apples comparison. At this
granularity prefix-cache hits work cleanly only when prefix length is a clean
multiple of 1,536 — see Preconditions above.

`--expert-cache-size` and the unified vs. static config flags differ per
cell — see the per-test sections below.

### Static-only flags

```
--gpu-memory-utilization 0.3105
```

Why 0.3105 (not 0.305): at block_size=1,536 with `cache=64`, util=0.305
auto-allocates only 3 KV blocks. A max-len request needs exactly 3 blocks
→ zero scheduler slack → vLLM deadlocks during prefix-cache eviction on
alternating workloads. util=0.3105 gives 4 blocks (= 6,144 KV tokens), 1
block of slack — enough for the scheduler to operate. 6,144 < 7,800
two-prefix demand, so static-bad is still genuinely starved (the
experimental property is preserved). Static-good (cache=20) at the same
util gets ~72k KV tokens, plenty of room.

### Unified-only flags

```
--expert-unified-pool
--num-gpu-blocks-override 68
```

**`--gpu-memory-utilization` is not set for unified cells.** The pool size
is pinned by `--num-gpu-blocks-override 68`, which both fixes the pool and
satisfies vLLM's allocator without needing a util value.

**Pool size is pinned at 68 blocks for every unified cell, in every test.**
This is the comparison fairness invariant: a fixed pool size across cells
keeps the only experimental variable as `--expert-cache-size` (initial split)
and isolates the reshape behaviour from any auto-allocator drift.

68 blocks at block_size=1,536 = 104,448 KV tokens of *headroom* if the entire
pool went to KV. In practice the pool is shared between expert pages and KV
pages — the split is the reshape we're measuring. 68 also satisfies the
warm-up floor `num_gpu_blocks ≥ expert_cache_size + 1` for every cache size
used in this plan (max is cache=64 → floor 65).

nvidia-smi at idle should read ~14 GB for unified cells — matching static's
~14 GB at util=0.3105. **Steady-state GPU memory consumed is the comparison
fairness invariant** for static vs. unified comparisons.

**Trace env var per pass:**
- Latency pass (static cells; pass 1 of unified cells): `VLLM_UNIFIED_POOL_TRACE`
  **unset**.
- Trace pass (pass 2 of unified cells only): `VLLM_UNIFIED_POOL_TRACE=1`. Never
  use `=2` in any timed run — that's debug-only and emits ~330k lines per prompt.

Each cell is **one server boot per seed**. Within that server, multiple
`vllm bench serve` invocations carry pool state forward. Restart the server between
cells (different `--expert-cache-size`) and between seeds (clean LRU state).

---

## Parallel execution across two GPUs

The machine has 2× L40 (both 46 GB, both idle). All cells in this plan are
independent — different cells, different seeds, both directions of Test 2 — so
we can run **two cells concurrently**, one per GPU, for ~2× wall-time speedup.

**Pinning scheme:**

| GPU | `CUDA_VISIBLE_DEVICES` | Port | Log dir suffix |
|---|---|---|---|
| 0 | `0` | `8000` | `_g0` |
| 1 | `1` | `8001` | `_g1` |

**Server launch — static cells (per GPU, latency pass):**

```bash
CUDA_VISIBLE_DEVICES=0 \
    /home/belal/150326/venv-phase-2/bin/vllm serve \
    allenai/OLMoE-1B-7B-0924-Instruct \
    --port 8000 \
    --expert-offload \
    --expert-cache-size <N> \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 \
    --gpu-memory-utilization 0.3105 \
    &> logs/<cell>_seed<N>_g0.log
```

**Server launch — unified cells (per GPU, latency pass):**

```bash
CUDA_VISIBLE_DEVICES=0 \
    /home/belal/150326/venv-phase-2/bin/vllm serve \
    allenai/OLMoE-1B-7B-0924-Instruct \
    --port 8000 \
    --expert-offload --expert-unified-pool \
    --expert-cache-size <N> \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 \
    --num-gpu-blocks-override 68 \
    &> logs/<cell>_seed<N>_g0.log
```

For the **trace pass** of a unified cell, prepend `VLLM_UNIFIED_POOL_TRACE=1`
to the unified launch command and write to `logs/<cell>_seed<N>_g0_trace.log`.

(Replace `0` and `8000` with `1` and `8001` for GPU 1.)

**Bench client (target the right server):**

```bash
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port 8000 \   # 8001 for GPU 1
    --endpoint /v1/completions \
    ...
```

**Pairing rules:**

- Within a cell, all bench calls (e.g., Test 2's two phases) must hit the
  *same* server — they share pool state. So a Test 2 cell occupies one GPU for
  the full duration of both phases. **Don't split a cell across GPUs.**
- Across cells, anything goes. Pair Test 1A cell on GPU 0 with Test 1B cell on
  GPU 1. Pair seed 1 of Test 2A on GPU 0 with seed 1 of Test 2B on GPU 1. Etc.
- Don't run two servers on the same GPU — both static (`util=0.3105`) and
  unified (`--num-gpu-blocks-override 68`) memory budgets are per-GPU, and
  two servers would collide on the residual budget when KV scales up.

**Driver script:** a small `run_sweep.sh` that maintains a queue of `(cell,
seed)` pairs and dispatches the next pair to whichever GPU finishes first.
Trivial to write; can be a `xargs -P 2` over a list of cell-seed pairs, or a
loop with a wait-for-first-free pattern.

Every server log filename includes both `_seed<N>` (already unique per seed)
and `_g{0,1}` (the GPU it ran on, for traceability if a result looks
suspicious — lets us correlate with `nvidia-smi` history). The `_g{0,1}`
suffix is **not** in result JSON filenames, since the bench output for
seed 1 is the same regardless of which GPU produced it.

---

## Test 0 — Output consistency check

**Goal:** demonstrate that the unified pool produces the same tokens as the static
baseline for the same prompt, modulo BF16 nondeterminism. Single seed, two
cells, two short bench calls. Adds a one-sentence claim to the results
("output consistency was verified against static baseline; tokens matched") with
the appendix evidence behind it.

### Cells

| Cell | `--expert-cache-size` | `--expert-unified-pool` |
|---|---|---|
| 0-static | 64 | no |
| 0-unified | 40 | **yes** |

### Workload

Three short deterministic prompts (no random workload — comparing tokens needs
determinism):
1. `"The capital of France is"` (~6 tokens)
2. `"def fibonacci(n):\n    "` (~10 tokens)
3. First 100 tokens of one of the long alternating prefixes from
   `alternating_prompts.jsonl`

For each prompt, request `max_tokens=32, temperature=0, seed=1`. Issue the same
three requests against each server via `curl`, save the returned token IDs (or
the decoded text — same comparison either way for greedy decoding).

### Reporting

- For each prompt, list the static-output and unified-output side by side.
  **Pass criterion:** identical tokens on at least 2/3 prompts; the third is
  allowed to diverge after the first ~10 tokens (BF16 reductions in the MoE
  GEMM are not bitwise-deterministic across cache layouts, so long completions
  *should* eventually diverge — that's expected).
- Single sentence in the results: "Output consistency between unified and
  static was verified on a deterministic test set; first-token agreement was
  exact, with cumulative divergence after ~10 tokens consistent with BF16
  reduction nondeterminism." Evidence in appendix.

### Total runs
2 server boots, 3 prompts each × 1 seed = **6 curl calls**. ~5 min wall time.

---

## Test 1 — Convergence from wrong start

**Anchored to:** `justification_scenarioA.md` and `justification_scenarioB.md`.

Reuse the existing workloads. Bump request counts so the convergence segment is
visible in addition to the warm-up segment. Add one new cell per scenario: unified
pool starting from the operator-bad cache size.

### 1A — KV-hot (alternating prefixes)

Workload: `alternating_prompts.jsonl` (already generated, **no random suffix**
— see Preconditions §2), 20 requests instead of 10 (rerun even the existing
static cells at 20 for apples-to-apples). The 20 prompts alternate between
two pure repeated-character prefixes ("a"-prefix and "b"-prefix), each
exactly 3,072 tokens = 2 × 1,536 blocks. There is no per-prompt suffix; the
only request-level variation is which of the two prefixes is sent.

| Cell | `--expert-cache-size` | `--expert-unified-pool` | Output |
|---|---|---|---|
| 1A-static-bad | 64 | no | `results/test1A_static_bad_seed{N}.json` |
| 1A-static-good | 20 | no | `results/test1A_static_good_seed{N}.json` |
| 1A-unified-from-bad | 64 (initial only) | **yes** | `results/test1A_unified_from_bad_seed{N}.json` |

Bench command (same for all three cells, only output filename differs):

```bash
vllm bench serve \
    --backend vllm --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name custom --dataset-path alternating_prompts.jsonl \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 \
    --num-prompts 20 \
    --max-concurrency 1 --num-warmups 1 \
    --seed ${SEED} \
    --result-filename results/test1A_<cell>_seed${SEED}.json \
    --save-result --trust-remote-code
```

**Reporting:**
- Mean and median TTFT, mean TPOT, prefix-cache hit rate (overall, all 20 requests).
- **Headline figure: per-request TTFT timeline overlaid with pool composition.**
  All three cells' TTFT on left axis. Unified cell's `expert-ours` and `prefix`
  page counts on right axis (one line each, dashed). The reader sees TTFT
  dropping and `expert-ours` shrinking on the same chart — the reshape *causes*
  the convergence.
- **Eviction-tier counts** for the unified cell (table in appendix):
  `prefix-global`, `kv-evicts-expert`, `kv-evicts-prefix`, `truly-free`. Confirms
  what kind of evictions did the work.

**Expected:** unified-from-bad's TTFT in the **last 10 requests** lands within ~15%
of static-good's. The first 5–10 requests pay the reshape cost. On the overlay,
`expert-ours` should fall from ~64 toward ~20 over those reshape requests; `prefix`
should rise correspondingly.

### 1B — Expert-hot (random)

Workload: `--dataset-name random`, 8 prompts (up from 3) so we have a longer
steady-state segment.

| Cell | `--expert-cache-size` | `--expert-unified-pool` | Output |
|---|---|---|---|
| 1B-static-bad | 16 | no | `results/test1B_static_bad_seed{N}.json` |
| 1B-static-good | 64 | no | `results/test1B_static_good_seed{N}.json` |
| 1B-unified-from-bad | 16 (initial only) | **yes** | `results/test1B_unified_from_bad_seed{N}.json` |

Bench:

```bash
vllm bench serve \
    --backend vllm --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 \
    --num-prompts 8 \
    --max-concurrency 1 --num-warmups 1 \
    --seed ${SEED} \
    --result-filename results/test1B_<cell>_seed${SEED}.json \
    --save-result --trust-remote-code
```

**Reporting:**
- Mean TTFT, **mean TPOT** (headline metric here — decode pays expert DMAs every
  token), output throughput.
- **Headline figure: per-request TPOT timeline overlaid with pool composition.**
  Same shape as 1A's overlay but mirrored: `expert-ours` should *grow* from 16
  toward ~60+, KV side shrinks. TPOT drops on the same chart.

**Expected:** unified-from-bad's TPOT in the **last 4 requests** lands within ~15%
of static-good's. The pool grows expert-side as random routing keeps missing
into cold experts; KV is dead weight under this workload and gets evicted.

### Test 1 total runs
- Static cells (4 of 6) × 3 seeds × 1 pass = 12 server boots.
- Unified cells (2 of 6) × 3 seeds × 2 passes (latency + trace) = 12 server boots.
- **Total: 24 server boots, 24 bench calls** (only latency-pass bench JSONs are
  retained; trace-pass bench output is discarded).

---

## Test 2 — One-shot workload shift

Two directions; both phases run against the same long-lived server. **Pool state
carries between phases** — that's the whole point.

**Static configs span the operator-choice space:**
- **expert-tuned (cache=64):** wins the random phase
- **prefix-tuned (cache=20):** wins the alternating phase
- **middle (cache=40):** the operator-hedge

**Unified config (headline):** initial cache=40 — neutral starting point, no
operator knowledge. **Ablation cells** start unified from cache=20 and cache=64 in
direction 1 only, to show convergence is initial-condition-independent.

### 2A — Direction: prefix-heavy → expert-heavy

Phase 1 (10 req): alternating prefixes — same bench command as 1A but
`--num-prompts 10`. Output: `results/test2A_<cell>_phase1_seed{N}.json`.

Phase 2 (6 req): random — same bench command as 1B but `--num-prompts 6` and
`--num-warmups 0` (the server has already done Phase 1 work; warming would
mis-reset the LRU). Output: `results/test2A_<cell>_phase2_seed{N}.json`.

Cells:
| Cell | `--expert-cache-size` | unified |
|---|---|---|
| 2A-static-prefix-tuned | 20 | no |
| 2A-static-middle | 40 | no |
| 2A-static-expert-tuned | 64 | no |
| 2A-unified-from-middle | 40 | **yes** |
| 2A-unified-from-prefix | 20 | yes (ablation) |
| 2A-unified-from-expert | 64 | yes (ablation) |

### 2B — Direction: expert-heavy → prefix-heavy

Phase 1 (6 req): random.
Phase 2 (10 req): alternating prefixes, `--num-warmups 0`.

Cells: same six as 2A but only the four headline cells (no ablation in this
direction — ablation in 2A is enough to make the convergence point).

### Reporting (both directions)

For each cell, report **per-phase** numbers:
- Phase 1: mean TTFT (if alternating) or mean TPOT (if random).
- Phase 2: same.

**Headline table per direction:**

| Cell | Phase 1 metric | Phase 2 metric | Both phases good? |
|---|---|---|---|
| static-prefix-tuned | (good, bad) | | one only |
| static-middle | (mediocre, mediocre) | | neither |
| static-expert-tuned | (bad, good) | | one only |
| **unified-from-middle** | (good, good) | | **both** ✓ |

**Headline figure: per-request TTFT/TPOT timeline across both phases overlaid
with pool composition for the unified cell.** Static lines flatline at their
bad value during their bad phase; unified rides the workload through both
phases. The pool composition trace shows the split shifting *at the phase
boundary* — that's the visible "moment of adaptation" the dissertation is
arguing for. This is the strongest single figure in the chapter.

**Eviction-tier counts** (appendix table) for each unified cell — the mix
should change at the phase boundary (e.g., Phase 1 dominated by
`kv-evicts-expert` as KV grows; Phase 2 by `prefix-global` as the expert side
reclaims those blocks).

### Test 2 total runs

Each Test 2 cell runs **two phases** per pass (Phase 1 bench call + Phase 2
bench call against the same long-lived server). Unified cells run two passes
(latency + trace), so 4 bench calls per unified cell per seed.

- Direction 2A: 3 static × 3 seeds × 1 pass + 3 unified × 3 seeds × 2 passes
  = 9 + 18 = **27 server boots**, 54 bench calls.
- Direction 2B: 3 static × 3 seeds × 1 pass + 1 unified × 3 seeds × 2 passes
  = 9 + 6 = **15 server boots**, 30 bench calls.
- **Total: 42 boots, 84 bench calls** (only latency-pass bench JSONs retained;
  trace-pass JSONs discarded).

---

## Total experimental budget

Wall-time figures are **with two-GPU parallelism**, accounting for the
two-pass workflow on unified cells. Single-GPU times in parentheses.

| Test | Server boots | Bench calls (kept) | Wall time (2-GPU) | Wall time (1-GPU) |
|---|---|---|---|---|
| 0 | 2 | (6 curl calls) | ~3 min (parallel, single seed) | ~5 min |
| 1 | 24 | 24 (12 trace-pass discarded) | ~16 min/seed | ~32 min/seed |
| 2 | 42 | 84 (28 trace-pass discarded) | ~70 min/seed | ~140 min/seed |
| **Total** | **68** | **108 + 6 curl** | **~4.5 hours for full 3-seed sweep** | **~9 hours** |

If wall-time becomes prohibitive even with both GPUs:
- Drop ablation cells in Test 2 (saves 4 unified-cell boots × 3 seeds = 12 boots
  / 24 bench calls / ~15 min per seed at 2-GPU).
- Consider running Test 1 at 1 seed if its contrast is large enough to be obvious.
- Skip the trace pass on seeds 2 and 3 of any unified cell if the seed-1 trace
  already makes the mechanism point cleanly. The *latency* numbers still come
  from 3 seeds; the overlay figure only needs one trace seed.

---

## Output organisation

```
results/
├── test0_static_outputs.txt          # token comparison evidence for appendix
├── test0_unified_outputs.txt
├── test1A_static_bad_seed1.json      # ... seed2, seed3
├── test1A_static_good_seedN.json
├── test1A_unified_from_bad_seedN.json
├── test1B_*_seedN.json
├── test2A_<cell>_phase1_seedN.json   # 6 cells × phase × seed
├── test2A_<cell>_phase2_seedN.json
├── test2B_*_phase{1,2}_seedN.json    # 4 cells × phase × seed
└── (existing: scenarioA_*.json, scenarioB_*.json reused as background)

logs/
├── test0_static_seed1_g{0,1}.log
├── test0_unified_seed1_g{0,1}.log
├── test1A_<cell>_seedN_g{0,1}.log           # latency-pass server stderr (no trace)
├── test1A_<cell>_seedN_g{0,1}_trace.log     # trace-pass server stderr (level-1 only) — unified cells only
├── test1B_*.log / test1B_*_trace.log
├── test2A_*.log / test2A_*_trace.log
└── test2B_*.log / test2B_*_trace.log
```

`*_trace.log` files exist only for unified cells (the trace pass). They are
the source of the pool composition overlay. Latency-pass `*.log` files for
unified and all logs for static cells are kept for sanity / boot-time
diagnostics but aren't parsed for figures.

---

## Aggregation / reporting helpers (to be written)

- `aggregate_results.py` — walks `results/`, joins seeds, emits CSV with
  mean ± std per cell per metric. One CSV per test.
- `plot_overlay.py` — reads per-request metrics from the bench JSONs **and**
  per-step pool composition from the parsed server logs, produces the
  TTFT/TPOT-overlaid-with-pool-composition figure. **This is the headline
  figure generator** for both Test 1 and Test 2.
- `parse_pool_trace.py` — parses level-1 `UNIFIED CACHE L<n> step=<S> ...`,
  `UNIFIED EVICT ... tier=...`, and `UNIFIED KV_CLAIM ... tier=...` lines from
  the **trace-pass** server logs only. Emits per-step pool composition
  (averaged across layers) and eviction-tier counts. Single regex for the
  CACHE line: `^.*UNIFIED CACHE L(\d+) step=(\d+) .*expert-ours=(\d+).*prefix=(\d+).*alloc-kv=(\d+).*free-pure=(\d+)`. Tier counts are
  `grep -c 'tier=<name>'`. Feeds `plot_overlay.py` and the appendix tier-count
  tables.

These are post-processing only — keep the bench cells themselves vanilla so we
can rerun any cell independently.

---

## Risks / things to watch during pilot

1. **Unified-pool memory accounting.** The MVP plan (§3) notes Phase 1 keeps a
   static staging tensor on GPU. Static cells get the full residual; unified
   cells lose that staging footprint. Verify at startup — printed staging
   overhead should be subtracted from KV budget so the comparison is fair on
   *dynamic* memory. If unified's KV budget is materially smaller than static's,
   the comparison is biased *against* unified and any win is conservative.
2. **Pool size under unified.** The auto-allocator should pick a sensible
   `num_gpu_blocks` after staging + initial expert cache are accounted for. If
   the pool is too generous, Tier 1 (truly-free) absorbs everything and the
   reshape never fires. Watch the eviction-tier counts in pilot — at least some
   `kv-evicts-expert` and `prefix-global` should appear in the mid-shift cells.
3. **`--num-warmups` discipline.** Phase 2 of Test 2 must use `--num-warmups 0`
   so the server's existing pool state isn't reset. Easy footgun.
4. **Per-request metrics format.** Verify `vllm bench serve --save-result`
   actually emits per-request TTFT/TPOT and not just summary stats. If not, we
   need to instrument or post-process the server's response stream.
5. **Aligning per-request timestamps to per-step pool snapshots.** Resolved by
   the trace-level redesign: every `UNIFIED CACHE` line in level-1 carries an
   explicit `step=<S>` field, and the latency-pass and trace-pass runs share
   a seed → both passes deterministically execute the same forward sequence.
   `plot_overlay.py` maps step → request index via the bench JSON's per-request
   step ranges (or, if the bench JSON doesn't expose step boundaries, by
   piecewise-linearly mapping step number to request index over the run's
   total step count, which is good enough for the visual).
6. **Mixed-LRU step-count bias against prefix blocks (FINDING from pilot).**
   The unified pool's Tier 2 eviction compares `oldest_prefix.step` vs
   `oldest_expert.step` and evicts whichever is older. Step counters bump
   on every touch regardless of value: a prefix is bumped *once per request*
   (when its blocks re-enter the free queue with hashes), while experts are
   bumped on *every forward pass* (8 top-k touches per layer per token). On
   workloads with even modest expert variety (decode + variable suffixes
   across multiple requests), every cached expert acquires a step ≥ the
   prefix's step within ~2 forward passes after the prefix's last bump,
   making the prefix the LRU and the eviction victim — even when the prefix
   block represents thousands of tokens of saved prefill (~10s of work) and
   each evicted expert represents ~1ms of saved DMA. **Mitigation in this
   plan:** strip the random suffix from `make_alternating_prompts.py` (see
   Preconditions §2) so each prefill stays at the variety-doc's "Candidate
   C" footprint (~11-20 hot experts/layer); ~44 cached experts then stay at
   step ≈ 0 and lose the LRU race to the prefix. **Future work:** replace
   the step-counter LRU with a value-weighted scoring policy (e.g.
   recompute-cost vs. probability-of-reuse), which would make the policy
   robust to realistic-variety workloads without needing the controlled
   prompt construction. Documented as a limitation of the current
   prototype, not a result.

---

## Next step

Pilot run: one seed of Test 1A only — 4 server boots (1A-static-bad latency,
1A-static-good latency, 1A-unified-from-bad latency, 1A-unified-from-bad
trace), paired across both GPUs. ~10 min wall time. Confirms:
- Server boots cleanly with all configs at the shared budget.
- The two-GPU pinning works (no cross-GPU contention, both servers reach
  steady-state KV size matching solo runs).
- vllm bench JSON contains per-request metrics.
- The trace pass log contains level-1 `UNIFIED CACHE L<n> step=<S> ...` lines
  parsable by the regex above.
- Latency pass and trace pass at the same seed produce the same per-request
  TTFT trajectory (sanity — confirms determinism for the alignment).
- Convergence direction is correct (unified-from-bad's later requests faster
  than its earlier ones).
- The overlay figure can be generated end-to-end from the artefacts.

If pilot looks clean, queue the full sweep. Run Test 0 alongside the pilot
since it's only ~3 minutes (also parallel across GPUs) — gets the correctness
claim out of the way early.
