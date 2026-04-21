# Motivation Benchmarks — Unified GPU VRAM Paging

**Goal**: Prove that static VRAM partitioning between KV cache and expert cache is suboptimal in two
distinct regimes, and that no single static split is simultaneously optimal for both. This is the
empirical motivation for a unified pool that dynamically rebalances at runtime.

---

## Setup

- **GPU**: NVIDIA L40 (48 GB VRAM)
- **Model**: `allenai/OLMoE-1B-7B-0924` (64 experts, 8 active per token, 16 layers)
- **Tool**: `vllm bench serve`

### How vLLM allocates memory

vLLM allocates in this order at startup:

1. Non-expert model weights (always on GPU, fixed)
2. Expert cache buffer: `expert-cache-size × expert_size` bytes (carved out before KV)
3. KV cache: whatever remains within `gpu_memory_utilization × total_VRAM`

Crucially, step 3 fills the remainder — KV automatically shrinks as expert cache grows. The total
used is always bounded by `gpu_memory_utilization × total_VRAM`. This means **varying
`--expert-cache-size` with a fixed `--gpu-memory-utilization` is the real tradeoff**: it shifts
VRAM between expert cache and KV within a fixed pool, exactly as a unified pool would do
dynamically. No VRAM is added or invented.

### Estimated VRAM breakdown

OLMoE KV size per token: 8 KV heads × 128 head_dim × 2 (K+V) × 16 layers × 2 bytes (fp16) = **64 KB/token**

Expert weights on GPU at each cache size (estimated ~184 MB per expert):

| expert-cache-size | Expert VRAM | Non-expert model | KV budget (at gpu-mem-util 0.40) | KV token capacity |
|---|---|---|---|---|
| 12 | ~2.2 GB | ~2 GB | ~15 GB | ~240K tokens |
| 32 | ~5.9 GB | ~2 GB | ~11.3 GB | ~181K tokens |
| 64 | ~11.8 GB | ~2 GB | ~5.4 GB | ~86K tokens |

**Why `--gpu-memory-utilization 0.40`**: the tradeoff is real at any utilization value, but at the
default (0.90) the KV budgets are 200K–600K tokens — so large that sequential single-user requests
never come close to filling them and Scenario A produces no observable KV starvation. Setting
utilization to 0.40 uniformly across all runs scales the pool down to ~19 GB (equivalent to
emulating a smaller GPU), bringing the KV budgets into a range where practical sequence lengths can
stress them. The same value is used for all six runs so the comparison is fair.

**Calibration step** (do this before running Scenario A): start each server config and read the
logged line:
```
INFO ... # GPU blocks: XXXX, # CPU blocks: YYYY
```
Multiply `# GPU blocks` by `block_size` (default 16 tokens) to get the exact KV token capacity.
Set `--prefix-len` to ~85% of the cache-size-64 KV capacity so that prefix + input overflows it
but fits easily in cache-size 12. The numbers below assume the estimates above hold; adjust if
logged values differ.

---

## The Motivation Table

| Config | Scenario A TTFT (high KV reuse) | Scenario D TPOT (high expert diversity) |
|---|---|---|
| cache-size 12 (KV-heavy) | **Good** — large KV budget holds prefix | **Bad** — frequent expert swaps |
| cache-size 32 (balanced) | Good — KV budget still ample | Mediocre — more swaps than cache-size 64 |
| cache-size 64 (expert-heavy) | **Bad** — KV budget overflows, prefix evicted | **Good** — all experts resident |
| **Unified pool** *(future)* | Good | Good |

No static split is simultaneously optimal. The unified pool automates what the oracle would do.

---

## Scenario A — High KV Reuse, Varying Expert Cache Size

**What we are showing**: when requests share a long prefix, a large expert cache starves the KV
cache, causing the prefix to be evicted and recomputed on every request. Reducing the expert cache
frees VRAM for KV, and the prefix stays resident — dramatically lowering TTFT.

**Why this is the unified pool's win condition**: the expert cache slots occupied by cache-size 64
are not all being used simultaneously (8 experts active per token, many slots idle), yet they
consume VRAM that would eliminate all prefix cache misses.

### Server startup (run one at a time)

```bash
# Shared flags for all Scenario A servers
SHARED="--model allenai/OLMoE-1B-7B-0924 \
        --expert-offload \
        --enforce-eager \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.40 \
        --max-model-len 131072 \
        --trust-remote-code"

# A-12: small expert cache → large KV budget
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 12

# A-32: balanced
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 32

# A-64: large expert cache → small KV budget
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 64
```

`--enable-prefix-caching` is required: without it vLLM does not reuse KV blocks across requests
regardless of how large the KV budget is.

`--max-num-batched-tokens` is **not** set to 1 here. With a ~70K-token sequence, processing 1 token
at a time would make a single cache-miss prefill take ~9 minutes (70K × 8ms). The cache hit/miss
contrast is equally clear with normal batching, just fast enough to actually run.

### Benchmark command

```bash
# Run against each server in turn
vllm bench serve \
  --backend vllm \
  --endpoint /v1/completions \
  --num-prompts 20 \
  --dataset-name random \
  --prefix-len 70000 \      # shared prefix across all requests — adjust per calibration
  --input-len 10000 \       # unique suffix per request
  --output-len 200 \
  --max-concurrency 1 \
  --trust-remote-code \
  --num-warmups 1
```

**Total tokens per request**: 70K (prefix) + 10K (input) + 200 (output) = **80,200 tokens**.

With the estimated budgets:
- cache-size 64 (~86K capacity): 80K fits but only barely — the prefix will be evicted to make room
  for the unique input + output on many requests → repeated cache misses → high TTFT
- cache-size 12 (~240K capacity): 80K is well within budget → prefix stays resident from request 2
  onwards → low TTFT

### Key metrics

| Metric | Interpretation |
|---|---|
| **Mean TTFT** | Primary signal. Low = prefix cache hit (skip prefix recompute). High = cache miss. |
| P99 TTFT | Tail behaviour under eviction pressure |
| Output throughput (tok/s) | Overall efficiency |

### Expected results

- **A-12** and **A-32**: TTFT drops sharply after request 1 (warmup). Subsequent requests hit the
  cached prefix and only compute the 10K unique suffix. TTFT ≈ cost of prefilling 10K tokens.
- **A-64**: TTFT stays high across all requests. Prefix is evicted to accommodate unique input +
  output, so every request recomputes the full 80K-token context.
- The TTFT gap between A-12 and A-64 is the quantified cost of static over-allocation to experts.

---

## Scenario D — High Expert Diversity, Varying Expert Cache Size

**What we are showing**: when requests have diverse, non-overlapping inputs, the KV cache provides
no reuse value — its VRAM allocation is wasted. At the same time, diverse inputs activate different
experts, thrashing a small expert cache. Increasing the expert cache (even at the cost of KV budget)
reduces per-token generation latency substantially.

**Why this is the unified pool's win condition**: the large KV budget of cache-size 12 sits mostly
empty (no prefix reuse, no concurrent requests), yet it forces frequent PCIe transfers that inflate
every token's latency.

### Server startup (run one at a time)

```bash
# Shared flags for all Scenario D servers
# Note: gpu-memory-utilization is default (0.90) — no artificial constraint needed here.
# The expert cache size is the variable; KV budget is irrelevant (no reuse).
SHARED="--model allenai/OLMoE-1B-7B-0924 \
        --expert-offload \
        --enforce-eager \
        --max-num-batched-tokens 1 \
        --trust-remote-code"

# D-12: small expert cache → frequent swaps
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 12

# D-32: balanced
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 32

# D-64: large expert cache → minimal swaps
python3 -m vllm.entrypoints.openai.api_server $SHARED --expert-cache-size 64
```

`--max-num-batched-tokens 1` is kept here for consistency with eval.txt and to isolate the
per-token expert swap cost without batching effects. `--enable-prefix-caching` is intentionally
omitted (diverse inputs, no prefix reuse).

### Benchmark command

```bash
vllm bench serve \
  --backend vllm \
  --endpoint /v1/completions \
  --num-prompts 50 \
  --dataset-name random \
  --input-len 100 \
  --output-len 200 \
  --max-concurrency 1 \
  --trust-remote-code \
  --num-warmups 2
```

No `--prefix-len`. Each request is a distinct random input, maximising expert activation diversity
across requests.

### Key metrics

| Metric | Interpretation |
|---|---|
| **Mean TPOT** | Primary signal. High = stalled waiting for expert swap from CPU. |
| Mean ITL | Correlated with TPOT; per-token stall from expert loading |
| Output throughput (tok/s) | Overall generation speed |

### Expected results

- **D-64**: TPOT near the eval.txt baseline (~14 ms). All experts on GPU; no PCIe transfers during
  generation.
- **D-12**: TPOT high (~32 ms, consistent with eval.txt). Only 19% of experts cached; most tokens
  trigger a swap.
- **D-32**: TPOT intermediate. Captures the gradient between the two extremes.
- The TPOT difference between D-12 and D-64 is the quantified cost of static over-allocation to KV.

---

## Summary: All Runs

| Run | expert-cache-size | gpu-mem-util | Workload | max-batched-tokens | Primary metric |
|---|---|---|---|---|---|
| A-12 | 12 | 0.40 | prefix-len 70K, input 10K, output 200, 20 prompts | default | TTFT |
| A-32 | 32 | 0.40 | same | default | TTFT |
| A-64 | 64 | 0.40 | same | default | TTFT |
| D-12 | 12 | 0.90 | input 100, output 200, 50 prompts, no prefix | 1 | TPOT / ITL |
| D-32 | 32 | 0.90 | same | 1 | TPOT / ITL |
| D-64 | 64 | 0.90 | same | 1 | TPOT / ITL |

---

## What the Unified Pool Claims to Achieve

Once implemented, a unified pool run against both workloads should match:
- **Scenario A**: TTFT of A-12 (pool recognises idle expert slots and shifts VRAM to KV)
- **Scenario D**: TPOT of D-64 (pool recognises cold KV blocks and shifts VRAM to expert cache)

A single pool with one static-equivalent starting split would adapt online to hit both targets,
without the operator knowing the workload type in advance.
