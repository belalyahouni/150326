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

- **A2 — workload-tuned (cache=32):** only the hot half of experts on GPU. Frees
  ~6 GB back to KV. KV grows to ~54,000 tokens — fits 13+ long prefixes. All
  prefixes stay resident → **~100% prefix-cache hit rate** after warmup.

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
| A2 cache=32 | 32 |  6.1 GB | ~6.6 GB | ~54,000 | ~13 |

Dense weights + activations take the residual ~1.2 GB in both runs.

---

## Workload

Two distinct prefixes of ~3,900 tokens each, 100-token unique suffix per request,
10 sequential requests **strictly alternating** between the two prefixes (A, B, A,
B, …). Alternation is the load-bearing property: it forces A's prefix-cache entry
to be evicted before A's next reuse in any KV that fits only one prefix.

Total prefix demand: 2 × 3,900 = 7,800 tokens. **Above** A1's KV capacity (4,224),
**well below** A2's (~54,000). That's the dividing line — no further tuning needed.

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

## Run A2 — Good config (cache=32)

**Terminal A — server** (only `--expert-cache-size` differs):
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-cache-size 60 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-batched-tokens 1 \
    --gpu-memory-utilization 0.305
```

Confirm: `GPU KV cache size: ~54,000 tokens`.

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



Expected: low mean TTFT after the first two requests — prefix hits, only the 100-token
suffix recomputes.

Ctrl-C the server when done.

---

## Result table to populate

| Run | Cache slots | KV tokens | Mean TTFT (ms) | Mean E2EL (ms) | Prefix hit rate | Notes |
|---|---|---|---|---|---|---|
| A1 cache=64 | 64 | ~4,224  | _fill_ | _fill_ | ~0%   | Prefixes evict each other |
| A2 cache=32 | 32 | ~54,000 | _fill_ | _fill_ | ~100% | Both prefixes stay resident |

Hit rate from server `/metrics` (Prometheus, `gpu_prefix_cache_hit_rate` or similar).

---

## Caveats

- **A2 incurs some cold-expert DMAs.** With 32 cache slots and the model's top_k=8,
  forwards may hit experts not currently cached. Cost is a few ms per miss; the savings
  from avoided prefix recompute (~hundreds of ms per request) dominate. The
  dissertation point is "even with that cost, freeing memory for prefix cache wins."
- **`--max-concurrency 1`** keeps the experiment sequential and easy to interpret.
  Concurrent requests would stress KV further in A1, sharpening the contrast.
- **Run 3× with `--seed 1, 2, 3`** if a single run looks noisy. Report mean ± std.
- **`--expert-unified-pool` is *not* set.** This benchmark establishes the static-config
  problem the unified pool is designed to solve.
