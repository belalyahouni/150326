# Realistic-workload test

## Goal

Show that the unified pool's headline result (M=64 budget) generalises beyond
the synthetic alternating workload to a **realistic LLM-serving pattern**:
shared system prompt + variable user query + decode. This is what production
chatbots, RAG systems, and code assistants actually look like.

## Workload structure

Each request:

```
[ 1,536-token system prompt — fixed, identical across all requests ]
[ 50-200 token user query — varies; drawn from a pool of 8 distinct queries ]
[ 50 decode tokens ]
```

- **System prompt**: 1,536 tokens = exactly 1 block at `block_size=1536`.
  Fixed text (a plausible instruction template). After the first request,
  this block is hashed and cached → every subsequent request gets a 1-block
  prefix-cache hit.
- **User queries**: 8 distinct natural-language queries, each tokenising to
  50-200 tokens. Bench cycles through them across 10 prompts → some queries
  repeat → tests whether the **second block** (which contains tail of system
  prompt + start of user query) caches when the same user query repeats.
- **Decode**: 50 tokens via `--custom-output-len 50`.
- **Order**: deterministic (cycling through the 8 queries).

`--num-prompts 10`, `--num-warmups 1`, `--seed 1`.

## Configurations to compare

At memory budget **M=64 only** (the headline budget where unified wins on
synthetic; this test asks "does it still win on realistic?"). Same operator-
trilemma framing as Test 2A:

| Cell | Cache | KV | Stance |
|---|---:|---:|---|
| static prefix-tuned | 20 | 44 | "operator expects KV-heavy workload" |
| static middle | 40 | 24 | hedge |
| static expert-tuned | 60 | 4 | "operator expects expert-heavy workload" |
| **unified** | init=40 | pool=64 | adaptive |

(Why C=60 not C=64: KV=0 doesn't boot. Same constraint as the budget sweep.)

## Apples-to-apples verification

Same approach as the budget sweep: capture `nvidia-smi memory.used` at idle
for each cell and confirm all 4 cells are within ±200 MiB. Per the page-
aliasing invariant (`expert_page_size == kv_block_size_per_layer == 12 MB`),
all 4 should sit at ~25 GB.

## Expected outcome

This workload has **less reshape opportunity** than alternating (no mid-run
phase shift, no two competing prefixes), so the unified-vs-static gap should
be **smaller** than what we measured on Test 1A/1B/2A. But unified should
still win because:

- Static dedicates separate physical memory to expert and KV. KV demand for
  this workload is small (~3 active blocks per request), so most of static's
  KV memory sits unused.
- Unified's pool space converts that wasted KV memory into expert-cache
  space → larger effective expert cache → better TPOT during decode.

**Concrete prediction**: unified beats best-static by ~5-10% on TTFT and
TPOT. Smaller gap than synthetic (where it was 20-50%), but a clean
real-workload win.

If unified loses on this realistic workload, that's a strong negative
finding worth flagging.

## Run budget

- 4 cells × 1 bench = 4 server boots
- 2-GPU parallelism: 2 rounds × ~5 min = **~10 min wall time**
- Each cell saves: `realistic_<config>_seed1.json` and idle-mem log

## Why this is in scope

- Same model, tokenizer, hardware, GPU memory budget as all earlier tests.
- Only changes the workload. No new variables (no concurrency, no multi-turn,
  no tool-use, no streaming).
- Tests the dissertation's central claim ("unified pool dominates static at
  matched memory") on a workload that approximates real deployment.

## What this gives the chapter

Adds **one paragraph** to the results: "the headline win at M=64 also holds
on a realistic system-prompt + variable-query workload." Closes the
"synthetic benchmark" objection without going down the rabbit hole of
ShareGPT trace replay or multi-tenant simulation.
