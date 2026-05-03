# 1.2.4 Empirical Study — Benchmark Runbook

Four runs in order. Each run = (1) start server in terminal A, (2) run client in terminal B,
(3) Ctrl-C the server before moving to the next run (server-side flags differ between runs).

Hardware: 1× L40 (46 GB). Model: `allenai/OLMoE-1B-7B-0924-Instruct`. Branch: `main`
(static expert-LRU). Pool budget held fixed at ~14 GB across all four runs.

```bash
# one-time prep — create results dir
mkdir -p results
```

---

## Run A1 — Scenario A baseline (50/50 split: cache=37, KV=3500 blocks)

**Terminal A — server:**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 37 \
    --num-gpu-blocks-override 3500 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.40
```

**Terminal B — client (Scenario A workload: 25 prefixes × 2500 tokens, KV-pressured):**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 2500 \
    --prefix-repetition-suffix-len 100 \
    --prefix-repetition-num-prefixes 25 \
    --prefix-repetition-output-len 20 \
    --num-prompts 50 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioA_baseline.json \
    --save-result \
    --trust-remote-code
```

Expected: high mean TTFT — prefixes don't all fit (62 500 prefix tokens vs 56 000 KV
budget) → re-prefill on every prefix re-use.

Then **Ctrl-C the server** before Run A2.

---

## Run A2 — Scenario A workload-aware (KV-heavy 25/75: cache=18, KV=5300 blocks)

**Terminal A — server:**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 64\
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.35
```

**Terminal B — client (same Scenario A workload, only output filename differs):**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 2500 \
    --prefix-repetition-suffix-len 100 \
    --prefix-repetition-num-prefixes 25 \
    --prefix-repetition-output-len 20 \
    --num-prompts 50 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioA_oracle.json \
    --save-result \
    --trust-remote-code
```

Expected: low mean TTFT — all 25 prefixes fit (84 800 KV tokens > 62 500 needed) →
near-100% prefix-cache hits after warm-up.

Then **Ctrl-C the server** before Run B1.

---

## Run B1 — Scenario B baseline (50/50 split: cache=37, KV=3500 blocks)

**Terminal A — server (identical to Run A1's server):**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 37 \
    --num-gpu-blocks-override 3500 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests
```

**Terminal B — client (Scenario B workload: random unrelated long prompts):**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-prefix-len 0 \
    --random-input-len 2048 \
    --random-output-len 80 \
    --random-range-ratio 0.3 \
    --num-prompts 20 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioB_baseline.json \
    --save-result \
    --trust-remote-code
```

Expected: high TTFT + high TPOT — every forward needs ~27 experts not in cache, each a
CPU→GPU PCIe DMA.

Then **Ctrl-C the server** before Run B2.

---

## Run B2 — Scenario B workload-aware (Expert-heavy 87.5/12.5: cache=64, KV=850 blocks)

**Terminal A — server:**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 64 \
    --num-gpu-blocks-override 850 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests
```

**Terminal B — client (same Scenario B workload, only output filename differs):**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-prefix-len 0 \
    --random-input-len 2048 \
    --random-output-len 80 \
    --random-range-ratio 0.3 \
    --num-prompts 20 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioB_oracle.json \
    --save-result \
    --trust-remote-code
```

Expected: low TTFT + low TPOT — all 64 experts permanently resident, zero PCIe traffic.

Ctrl-C the server.

---

## Result table to populate

| Run | Cache slots | KV blocks | Mean TTFT (ms) | Mean E2EL (ms) | Throughput (req/s) | Notes |
|---|---|---|---|---|---|---|
| A1 baseline 50/50 | 37 | 3 500 | _fill_ | _fill_ | _fill_ | Prefix cache thrashes |
| A2 oracle KV-heavy | 18 | 5 300 | _fill_ | _fill_ | _fill_ | All 25 prefixes resident |
| B1 baseline 50/50 | 37 | 3 500 | _fill_ | _fill_ | _fill_ | ~27 expert DMAs per forward |
| B2 oracle expert-heavy | 64 | 850 | _fill_ | _fill_ | _fill_ | 0 expert DMAs |

JSON results land in `results/scenarioA_baseline.json`, `scenarioA_oracle.json`,
`scenarioB_baseline.json`, `scenarioB_oracle.json`.

---

## Why these numbers (memory math)

- **Per cache slot:** 16 layers × 3 mats × 2048 × 1024 × 2 B = **192 MB**
- **Per KV block:** 16 tokens × 16 layers × 2 × 16 heads × 128 dim × 2 B = **2 MB**
- **Dense weights** (attn + router + embeddings, BF16): ~2 GB, lives outside the pool

| Split | Cache slots × 192 MB | KV blocks × 2 MB | Pool total | KV tokens |
|---|---|---|---|---|
| 50/50 (baseline) | 37 → 7.1 GB | 3 500 → 7.0 GB | 14.1 GB | 56 000 |
| KV-heavy 25/75 (A oracle) | 18 → 3.5 GB | 5 300 → 10.6 GB | 14.1 GB | 84 800 |
| Expert-heavy 87.5/12.5 (B oracle) | 64 → 12.3 GB | 850 → 1.7 GB | 14.0 GB | 13 600 |

Same pool size, different splits. Scenario A's prefix demand (62 500 tokens) sits *above*
the 50/50 KV (56 000) but *below* the oracle KV (84 800) — that's why baseline thrashes
and oracle doesn't. Scenario B has 64 experts active and 50/50 caches only 37 → 27 DMAs
per forward; oracle caches all 64 → 0 DMAs.

`--enforce-eager` is mandatory: cache misses involve variable-length DMA which can't be
captured in a CUDA graph (see `expert-cache-implementation.md`). `--expert-unified-pool`
is **not** set — this study measures the *static* baseline that the dissertation will
later improve upon.

## Visuals to produce from the four JSON files

1. Two-bar chart per scenario: mean E2E latency, baseline vs oracle, with speed-up annotation.
2. Stacked TTFT vs decode latency for all four runs — shows *which phase* is hurting in each baseline.
3. Pool-allocation diagram: four side-by-side stacked bars (cache GB / KV GB summing to
   the fixed 14 GB) — visually drives home that pool size is identical, only the split differs.
4. Optional: prefix-cache hit-rate from server `/metrics` (Prometheus endpoint) for A1 vs A2.

## Caveats to acknowledge in the write-up

- **OLMoE on one L40 is small relative to production MoE serving.** Pool sizes here are
  scaled so the trade-off is observable in 14 GB; the qualitative shape (a single static
  split is suboptimal) is what generalises.
- **`max-concurrency=1` understates KV pressure.** Real serving sees concurrent
  requests, which makes the KV-heavy oracle even more important. Sequential here keeps
  interpretation clean.
- **Random tokens are not natural-language traffic.** Routing patterns may differ.
  Acceptable for this feasibility study; the dissertation's later evaluation should rerun
  on a real trace.
- **Repeat each measurement 3× with different `--seed` (1, 2, 3)** and report mean ± std
  if any single run looks noisy.
