# Scenario A — KV-hot Justification Benchmark

Two runs, single scenario. Same total GPU budget, only `--expert-cache-size` changes.
Demonstrates that a config that caches all experts starves the prefix cache, while a
config that caches only the hot subset frees enough memory for prefix-cache hits to
work — at the cost of cheap occasional expert DMAs.

Hardware: 1× L40 (46 GB). Model: `allenai/OLMoE-1B-7B-0924-Instruct`. Branch: `main`
(static expert-LRU, no unified pool).

---

## The Idea

- **A1 — naive default (cache=64):** every expert lives on GPU. ~12.3 GB used by
  expert cache. KV cache squeezed to ~4,224 tokens — barely one request worth.
  Alternating prefixes evict each other → **~0% prefix-cache hit rate** → every
  request recomputes the full prefix.

- **A2 — workload-tuned (cache=20):** only the hot ~⅓ of experts on GPU. Frees
  ~8.5 GB back to KV. KV grows to ~72,000 tokens — fits 20+ long prefixes. All
  prefixes stay resident → **~100% prefix-cache hit rate** after warmup.

The choice of cache=20 is workload-driven: the alternating prompts (see Workload
below) are built from `expert_variety_test.md`'s Candidate C (single repeated
character), which touches only 11–20 distinct experts per layer. A 20-slot expert
cache covers that footprint with negligible cold-expert DMA.

Cold expert misses in A2 cost a few ms of PCIe DMA each. Prefix recompute in A1
costs hundreds of ms per request. The trade is heavily in A2's favour.

---

## Memory Math

Per slot / block (BF16, OLMoE, 16 layers):
- Expert slot: **192 MB** (16 × 3 mats × 2048 × 1024 × 2 B)
- KV block (16 tokens): **2 MB**

GPU budget at `--gpu-memory-utilization 0.305` on L40 (46 GB) ≈ 14.0 GB:

| Run | Cache slots | Expert cache | KV cache | KV tokens | Long prefixes that fit |
|---|---|---|---|---|---|
| A1 cache=64 | 64 | 12.3 GB | 0.52 GB | 4,224  | ~1  |
| A2 cache=20 | 20 |  3.75 GB | ~9.0 GB | ~72,000 | ~18 |

Dense weights + activations take the residual ~1.2 GB in both runs.

---

## Workload

Two distinct prefixes of ~3,900 tokens each, 60-token random suffix per request,
10 sequential requests **strictly alternating** between the two prefixes (A, B, A,
B, …). Alternation is the load-bearing property: it forces A's prefix-cache entry
to be evicted before A's next reuse in any KV that fits only one prefix.

The two prefixes are built by tokenizing `"a" * 50000` and `"b" * 50000` and
truncating to PREFIX_TOKENS — single-character repetition, following Candidate C
of `expert_variety_test.md`. This keeps each prefix's expert footprint to ~11–20
distinct experts per layer (vs ~64/64 for random tokens), which is what makes the
cache=20 A2 config viable. With a random-token workload, A2 would thrash the
small expert cache and the picture would be muddied by expert-DMA latency on top
of the prefix-cache effect we want to isolate.

PREFIX_TOKENS is set to **3,900** (not e.g. 3,000) deliberately. A1's KV holds
264 16-token blocks (4,224 tokens). At 3,900-token prefixes, each request occupies
~248 blocks (244 prefix + 4 suffix), leaving only 16 blocks free. When the next
alternating request loads, it must evict ≥232 of the previous prefix's blocks,
which guarantees block 0 of the previous prefix is evicted — breaking the
contiguous-prefix-hit chain on the next same-prefix request. Smaller prefixes
(e.g. 3,000 tokens / 192 blocks) leave 72 KV blocks of headroom, so the previous
prefix retains residue at its tail and the hit-rate metric reports partial
contiguous hits instead of the clean ~0% the experiment is meant to show.

Total prefix demand: 2 × 3,900 = 7,800 tokens. **Above** A1's KV capacity (4,224),
**well below** A2's (~72,000). That's the dividing line — no further tuning needed.

> **Why not vllm bench's built-in `prefix_repetition`?** That dataset emits prompts
> in *clusters* (5× A, then 5× B), not alternating. With clustered ordering, the
> previous request's prefix blocks remain in the free queue with their hash, so the
> next request hits regardless of total KV size — both runs converge on ~80% hit
> rate and the experiment fails to differentiate. We use vllm bench's `custom`
> dataset with `--disable-shuffle` instead, fed a JSONL we generate ourselves.

**One-time prep — generate the alternating prompts file and the results dir:**
```bash
mkdir -p results
python make_alternating_prompts.py    # writes alternating_prompts.jsonl
```

---

## Run A1 — Bad config (cache=64)

**Terminal A — server:**
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 64 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.305
```

Confirm in startup log: `Available KV cache memory: ~0.5 GiB`, `GPU KV cache size:
~4,224 tokens`. If KV is much larger, the bad config isn't biting — lower util by 0.005.

**Terminal B — client:**
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name custom \
    --dataset-path alternating_prompts.jsonl \
    --disable-shuffle \
    --skip-chat-template \
    --custom-output-len 20 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioA_bad.json \
    --save-result \
    --trust-remote-code
```

Expected: high mean TTFT — every request re-prefills the full ~3,900-token prefix
because the previous request's prefix was just evicted to fit the alternating one.

Then **Ctrl-C the server** before A2.

---

## Run A2 — Good config (cache=20)

**Terminal A — server** (only `--expert-cache-size` differs):
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 20 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.305
```

Confirm: `GPU KV cache size: ~72,000 tokens`.

**Terminal B — client** (only output filename differs):
```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name custom \
    --dataset-path alternating_prompts.jsonl \
    --disable-shuffle \
    --skip-chat-template \
    --custom-output-len 20 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --num-warmups 1 \
    --seed 1 \
    --result-filename results/scenarioA_good.json \
    --save-result \
    --trust-remote-code
```

Expected: low mean TTFT after the first two requests — prefix hits, only the
60-token suffix recomputes.

Ctrl-C the server when done.

---

## Results — single seed=1 run, 2026-05-03

| Run | Cache slots | KV tokens | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Prefix hit rate | Wall time (s) |
|---|---|---|---|---|---|---|---|
| A1 cache=64 | 64 | 4,224  | **41,305** | 45,335 | 12.20 | **13.1%** (5696/43624) | 415.4 |
| A2 cache=20 | 20 | 71,808 | **5,999**  | 1,408  | 17.14 | **80.4%** (35056/43624) | 63.3 |

**Speedup A2/A1:** Mean TTFT **6.9×**, median TTFT **32.2×**, wall time **6.6×**.

JSON results land in `results/scenarioA_bad.json` and `results/scenarioA_good.json`.

### Why A1 isn't strictly 0%, and A2 isn't strictly 100%

The hit-rate metric counts the longest contiguous prefix-cache match starting at
block 0 (`kv_cache_manager.py:find_longest_cache_hit`). Two effects float A1 above
0% and pull A2 below 100%:

- **Warmup primes `prompt[0]` for free.** vllm bench warmup runs the first prompt
  (a-prefix); the main run also opens with `prompt[0]`. That single request is a
  full ~3,904-token hit no matter how small KV is. Across 11 total requests
  (1 warmup + 10 main, ~3,963 tok each ≈ 43,624 token queries), that one free hit
  alone is ~9% — i.e., the floor of A1's hit rate, regardless of alternation.
- **A2's ceiling is ~80% by construction.** Warmup hits 0% (empty cache); the
  first "b" request hits 0% (b never seen); the first "a" request after warmup
  hits ~99% but its random suffix is fresh. After steady state, every request
  hits its 244-block prefix but always misses the 4-block random suffix and the
  output. Sum over 11 requests: ~3,904 hits × ~9 cache-warm requests / 43,624
  total ≈ 80%. The TTFT median (1,408 ms vs A1's 45,335 ms) is the metric that
  shows the steady-state benefit clearly.

---

## Caveats

- **A2 incurs some cold-expert DMAs.** With 20 cache slots and the model's top_k=8,
  forwards occasionally hit experts not currently cached. The Candidate-C prompts
  hold this to a handful of misses per layer at most; cost is a few ms per miss,
  and the savings from avoided prefix recompute (~hundreds of ms per request)
  dominate. The dissertation point is "even with that cost, freeing memory for
  prefix cache wins."
- **The cache=20 sizing is workload-coupled.** It works because the prompts touch
  ~11–20 experts/layer. A random-token workload at the same cache size would
  thrash. This is exactly the brittleness the unified pool is designed to remove.
- **`--max-concurrency 1`** keeps the experiment sequential and easy to interpret.
  Concurrent requests would stress KV further in A1, sharpening the contrast.
- **Run 3× with `--seed 1, 2, 3`** if a single run looks noisy. Report mean ± std.
- **`--expert-unified-pool` is *not* set.** This benchmark establishes the static-config
  problem the unified pool is designed to solve.
