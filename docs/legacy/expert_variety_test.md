# Test: find a Phase 1 prompt that hits as few distinct experts as possible

Goal: discover a long prefix that fills KV without spreading expert activations across all 64 experts per layer. We need this so that during the real Phase 2 test, the small set of Phase-1-warmed experts stays comfortably in cache while the cold prefix blocks dominate the LRU tail and lose to Phase-2 expert misses (`tier=prefix-global`).

We use the existing `UNIFIED RESULT` trace lines at `vllm/model_executor/layers/fused_moe/unified_pool.py:584` — they already list every expert ID hit/missed per layer per forward step. No new logging needed; just grep + sort -u.

## 1. Server

Generous pool and full expert cache so nothing gets evicted during the experiment — that way the trace is a clean record of "which experts the router *chose*", not contaminated by cache-pressure reloads.

```bash
VLLM_UNIFIED_POOL_TRACE=1 vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-unified-pool \
    --expert-cache-size 64 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-num-batched-tokens 1 \
    --max-model-len 4096 \
    --num-gpu-blocks-override 96 \
    --attention-backend TRITON_ATTN \
    --no-async-scheduling \
    &> /tmp/expert_variety_test.log
```

## 2. Send one candidate prompt

`vllm bench serve --dataset-name random` is the worst possible choice here — random tokens maximize router spread. We construct deterministic, low-entropy prompts via `python3` + `jq` and post them with `curl`.

**Before each candidate**, restart the server (or at minimum bump a marker into the log so you can scope your grep):

```bash
echo "===== CANDIDATE: <name> =====" >> /tmp/expert_variety_test.log
```

### Candidate A — single token repeated

The most concentrated possible workload. Position embeddings still vary per token, so routing won't *fully* collapse, but this is the floor.

```bash
PROMPT=$(python3 -c 'print("the " * 3000, end="")')
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "$PROMPT" '{
    model: "allenai/OLMoE-1B-7B-0924-Instruct",
    prompt: $p,
    max_tokens: 4,
    temperature: 0,
    seed: 1
  }')"
```

### Candidate B — short phrase repeated

Slightly more natural; routing usually concentrates on a handful of experts per layer.

```bash
PROMPT=$(python3 -c 'print("The cat sat on the mat. " * 500, end="")')
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "$PROMPT" '{
    model: "allenai/OLMoE-1B-7B-0924-Instruct",
    prompt: $p,
    max_tokens: 4,
    temperature: 0,
    seed: 1
  }')"
```

### Candidate C — single character

```bash
PROMPT=$(python3 -c 'print("a" * 9000, end="")')
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "$PROMPT" '{
    model: "allenai/OLMoE-1B-7B-0924-Instruct",
    prompt: $p,
    max_tokens: 4,
    temperature: 0,
    seed: 1
  }')"
```

For the actual Phase 2 test you want **~3000 prompt tokens** (so each request occupies ~3 KV blocks at the unified pool's ~1024-tokens-per-block granularity). Use a tokenizer to confirm if you need a precise number; for screening, eyeball it.

## 3. Analyze: which experts did the router actually choose?

OLMoE has 16 layers × 64 experts. After running a candidate, run the snippets below scoped to that candidate's section of the log (e.g., `awk '/===== CANDIDATE: A =====/,0' /tmp/expert_variety_test.log` piped into the greps).

### a. Unique experts hit per layer

```bash
for L in $(seq 0 15); do
  count=$(grep "UNIFIED RESULT L${L} " /tmp/expert_variety_test.log \
          | grep -oE "E[0-9]+" | sort -u | wc -l)
  echo "L${L}: ${count}/64 unique experts hit"
done
```

### b. Which experts were NOT hit (per layer)

```bash
for L in $(seq 0 15); do
  hit=$(grep "UNIFIED RESULT L${L} " /tmp/expert_variety_test.log \
        | grep -oE "E[0-9]+" | sort -u | sed 's/E//')
  echo "L${L} unhit: $(comm -23 <(seq 0 63) <(echo "$hit" | sort -n) | tr '\n' ',' )"
done
```

### c. Hit-count distribution across all layers

Top experts overall (a tiny set dominating means Phase 1 is concentrated, which is what we want):

```bash
grep "UNIFIED RESULT" /tmp/expert_variety_test.log \
  | grep -oE "E[0-9]+" | sort | uniq -c | sort -rn | head -20
```

### d. Total expert activations summed across all layers

```bash
total_unique_layer_expert_pairs=$(
  grep "UNIFIED RESULT L" /tmp/expert_variety_test.log \
    | grep -oE "L[0-9]+ .*" \
    | awk '{
        layer=$1;
        for(i=1;i<=NF;i++) if(match($i, /E[0-9]+/)) print layer, substr($i, RSTART, RLENGTH);
      }' | sort -u | wc -l
)
echo "Total distinct (layer, expert) pairs touched: ${total_unique_layer_expert_pairs} / $((16*64))"
```

This is the most important single number. It tells you: across the whole model, how many expert slots will Phase 1 keep warm. If it's, say, 200/1024, your Phase 1 expert footprint is small and there's lots of room for `prefix_lru` to grow before Phase 2 evictions need to fight a packed expert cache.

## 4. Interpretation

| Result | Meaning |
|---|---|
| ~64/64 unique per layer for every layer | Workload still too diverse — routing is spreading across all experts. Try more aggressive repetition (Candidate A or C). |
| < 16/64 unique per layer for most layers | Excellent — use this prompt as Phase 1 of the unified-pool swap test. |
| Total `(layer, expert)` pairs ≪ 1024 | Good — small Phase 1 expert footprint. The full expert cache (64 per layer) won't actually be needed, leaving pool slack for prefix_lru. |
| Single expert dominates (one expert hit thousands of times in distribution) | Realistic-enough; routing collapse on repeated tokens is expected and fine for our purposes. |

## 5. Pick the winner

The candidate with the smallest `(layer, expert)` pair count is your Phase 1 prompt. Carry it into the eventual two-phase Phase 2 swap test (Phase 1 = N sequential calls of this prompt with different seeds → fills `prefix_lru`; Phase 2 = expert-diverse workload → drains it via `tier=prefix-global` evictions).

## 6. Results — run 2026-05-03

Server: `--expert-cache-size 64 --num-gpu-blocks-override 96 --enable-prefix-caching --max-num-batched-tokens 1` (full expert cache so no eviction-driven reloads contaminate the count).

Method note: the in-doc `>>` marker line was clobbered because vllm opens the log file with `>` (truncate) and tracks its own write offset, which overwrites appended marks. We used line-count snapshots before/after each curl as the boundary instead — works reliably.

### Per-candidate footprint

| Candidate | Prompt | Tokens sent | (layer, expert) pairs | Per-layer range |
|---|---|---|---|---|
| Baseline (random tokens, prefix_lru_test) | `--dataset-name random` ×3200-token prefix | ~3200 | **1024 / 1024** | 64/64 every layer |
| A | `"the " * 3000` | 3001 | 376 / 1024 | 17–29 / 64 |
| B | `"The cat sat on the mat. " * 500` | 3501 | 748 / 1024 | 34–55 / 64 |
| C (initial) | `"a" * 9000` | 1125 *(tokenized to chunks)* | 257 / 1024 | 12–19 / 64 |
| **C-scaled** | `"a" * 24000` | **3000** | **243 / 1024** | **11–20 / 64** |

### Top dominating experts (C-scaled)

| Expert | Routings |
|---|---|
| E6 | 18,014 |
| E59 | 18,012 |
| E51 | 15,015 |
| E1 | 15,007 |
| E43, E12, E58 | 12,005–12,008 each |
| E60, E9, E45 | ~9,000 each |

A handful of ~10 experts dominates routing across all layers — the most concentrated pattern of any candidate tried.

### Verdict

- **Candidate B is unsuitable.** Natural-language repetition spreads routing across most of every layer (34–55/64). Phase 1 with B would warm too many experts to leave room for `prefix_lru`.
- **Candidate A** (`"the " * 3000`) reaches the right length and is 63% smaller than the random baseline — usable but no longer the leader.
- **Candidate C-scaled is the Phase 1 winner.** `"a" * 24000` tokenizes to exactly 3000 tokens (right for ~3 KV blocks per request) and touches only 243 / 1024 (layer, expert) pairs — a 76% reduction vs. random and 35% smaller footprint than A. Per-layer range is 11–20 / 64. Use this as Phase 1 of the two-phase swap test.
