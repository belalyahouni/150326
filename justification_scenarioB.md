# Scenario B — Expert-hot Justification Benchmark

Two runs, single scenario. Same total GPU budget, only `--expert-cache-size` changes.
Demonstrates the **opposite** trade-off from Scenario A: when the workload has no
prefix reuse and broad expert activation, giving memory to the KV cache is wasted —
the unused KV blocks decay; the win comes from packing every expert into GPU memory
so the MoE kernel never pays a PCIe DMA.

Hardware: 1× L40 (46 GB). Model: `allenai/OLMoE-1B-7B-0924-Instruct`. Branch: `main`
(static expert-LRU + the `gpu_model_runner` fix that makes `--expert-cache-size`
actually affect the auto-allocated KV budget).

---

## The Idea

- **B1 — naive cache=16:** only 16 of 64 experts on GPU. Every forward needs up to
  8 unique experts per token, and across a chunked-prefill batch of 512 tokens the
  router touches most of the 64 experts. The 16-slot cache thrashes; ≥30 expert
  DMAs per layer per forward → high TTFT (prefill) **and** high TPOT (every decode
  forward also picks 8 routed experts that may not all be cached). Freed memory
  goes to KV — but the random workload has no prefix overlap, so that KV is dead
  weight.
- **B2 — workload-tuned cache=64:** all 64 experts permanently resident on GPU.
  Zero PCIe traffic regardless of routing. Fast prefill, fast decode. The KV cache
  is tiny, but the workload doesn't reuse prefixes anyway.

The dissertation point: **a static config that splits memory 50/50 (or any KV-leaning
ratio) loses to a workload-tuned all-experts config when the workload is expert-hot.**
The unified pool's job is to discover this ratio automatically without the operator
having to know the workload.

---

## Why this works (after the fix)

In the previous main-branch code, `model_memory_usage` was captured before
`_maybe_init_expert_cache` ran, so `--expert-cache-size` had no effect on the
auto-allocated KV cache. Scenario B's full memory rebalance therefore couldn't be
demonstrated without `--num-gpu-blocks-override`.

Commit `f74d38a89f` ("Re-measure GPU memory after expert cache init") fixes this:
the profiler now re-reads GPU usage after experts move to CPU and the cache is
allocated, so the auto KV budget grows when the cache shrinks. Scenario B now uses
the same one-knob design as Scenario A — only `--expert-cache-size` differs between
runs.

---

## Memory Math

Per slot / block (BF16, OLMoE, 16 layers):
- Expert slot: **192 MB**
- KV block (16 tokens): **2 MB**

GPU budget at `--gpu-memory-utilization 0.305` on L40 (46 GB) ≈ 14.0 GB:

| Run | Cache slots | Expert cache | KV cache (auto) | Where it goes |
|---|---|---|---|---|
| B1 cache=16 | 16 | 3.1 GB | ~10 GB | KV is huge but **unused** (no prefix reuse) |
| B2 cache=64 | 64 | 12.3 GB | ~0.5 GB | All experts resident, KV is irrelevant |

Dense weights + activation peak take the residual ~0.9 GB in both runs.

---

## Workload

Five random prompts, ~1024 tokens each, 80-token output, **no shared prefix**.
Sequential. The point is to exercise the router across diverse inputs so the union
of experts touched per chunked-prefill forward spans most/all of the 64. We use
vllm bench's built-in `random` dataset directly — unlike Scenario A, no custom JSONL
is needed because we *want* every prompt to be unrelated.

> **Why no `--max-num-batched-tokens 1`?** That flag was only required for the
> unified-pool MVP (variable-length DMAs can't be CUDA-graphed). On main + static
> expert cache, dropping it lets vLLM run normal chunked prefill, which packs
> multiple tokens into one forward and naturally activates many experts at once —
> exactly what an expert-hot scenario needs to surface DMA cost. Keeping it would
> stretch each cache=16 prefill into a many-minutes affair without changing the
> qualitative outcome.

```bash
mkdir -p results
```

---

## Run B1 — Bad config (cache=16)

**Terminal A — server:**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 16 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.305
```

Confirm in startup log: `Available KV cache memory` should be ≥ 9 GiB and
`GPU KV cache size` should report tens of thousands of tokens (the freed memory
flowing into KV thanks to the fix).

**Terminal B — client:**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 80 \
    --random-range-ratio 0 \
    --random-prefix-len 0 \
    --num-prompts 5 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioB_bad.json \
    --save-result \
    --trust-remote-code
```

Expected: high TTFT (prefill thrashes the cache) and high TPOT (every decode
forward needs 8 experts that may not be in the 16-slot cache).

Then **Ctrl-C the server** before B2.

---

## Run B2 — Good config (cache=64)

**Terminal A — server** (only `--expert-cache-size` differs):
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 64 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.305
```

Confirm: `Available KV cache memory` should drop to roughly 0.5 GiB and
`GPU KV cache size` should be on the order of 4,000 tokens — that's exactly the
"KV is starved but the workload doesn't care" state we want.

**Terminal B — client** (only output filename differs):
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 80 \
    --random-range-ratio 0 \
    --random-prefix-len 0 \
    --num-prompts 5 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioB_good.json \
    --save-result \
    --trust-remote-code
```

Expected: low TTFT and low TPOT — every routed expert is already on GPU; zero PCIe
DMA traffic.

Ctrl-C the server when done.

---

## Result table to populate

| Run | Cache slots | KV tokens | Mean TTFT (ms) | Mean TPOT (ms) | Output throughput (tok/s) | Notes |
|---|---|---|---|---|---|---|
| B1 cache=16 | 16 | huge / unused | _fill_ | _fill_ | _fill_ | Constant expert DMA |
| B2 cache=64 | 64 | tiny / unused | _fill_ | _fill_ | _fill_ | All experts resident |

JSON results land in `results/scenarioB_bad.json` and `results/scenarioB_good.json`.

The headline metrics for Scenario B are **TPOT** and **output throughput** (decode
phase, where the static cache penalty bites every single token). In Scenario A, TTFT
told the story; here, TPOT does.

---

## Caveats

- **B1's freed KV is genuinely wasted** in this benchmark — there's no prefix
  reuse, no concurrent requests competing for KV, and each random prompt fits in
  the model's 4k context easily. This is *intentional*: we're showing the operator
  who blindly gave KV "all the leftover memory" what they're losing.
- **`--max-concurrency 1`** keeps interpretation clean. With concurrent requests,
  the small-cache config's compounding misses would slow throughput further; the
  qualitative result holds either way.
- **Run 3× with `--seed 1, 2, 3`** if a single run looks noisy. Random workloads
  have more run-to-run variance than fixed prefixes.
- **`--expert-unified-pool` is *not* set.** This benchmark establishes the
  static-config problem the unified pool is designed to solve, mirroring Scenario A.
