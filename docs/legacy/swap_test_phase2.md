# Test: Phase 2 unified pool — prefix-heavy → expert-heavy swap

## Goal

Demonstrate that the unified-pool manager reshapes its memory budget at runtime under workload pressure. Start with a workload that fills the pool with cold prefix KV blocks (low expert demand), then switch to an expert-hungry workload and watch the manager evict prefix blocks (`tier=prefix-global` in the trace) to grow the per-layer expert cache beyond its initial size.

This covers the **"Large KV budget / small expert budget" → "expert-hot workload"** cell of the dissertation 2x2 matrix. The reverse direction (prefix pressure evicting cold experts) goes through a different code path and gets its own follow-up document.

## Hypothesis

After **Phase 1**, the per-layer `UNIFIED CACHE` line should show:
- `prefix=` high (most of the pool)
- `expert-ours=` ~8 (the initial top-k warmup, untouched because Phase 1's expert footprint is small and fits in those 8 slots)
- `free-pure=` near 0

After **Phase 2**, the same line should show:
- `expert-ours=` much higher (most of the cache slots — the cache *grew*)
- `prefix=` much lower
- A non-zero count of `tier=prefix-global` lines emitted during Phase 2

If we see this, Phase 2 of the unified pool is working as designed.

## Workload design

Two phases against the same server. Use line-count snapshots to scope per-phase analysis (an in-log marker via `echo >>` would be clobbered — vllm opened the log with `>` and tracks its own offset, so its writes overwrite appended marks).

### Phase 1 — prefix-dominant fill

Goal: pile up cold prefix blocks in `prefix_lru` while keeping expert footprint tiny.

**Prompt template**: `"req <N>: " + "a" * 23900` — same low-variety body as the C-scaled winner from `expert_variety_test.md` (243/1024 pairs for plain `"a"`-spam), prepended with a short distinct tag per request. The tag is ~5 tokens so it won't shift routing materially, but it makes the first chunk's hash unique → prefix caching can't collapse later requests onto earlier ones, and each request deposits ~3 fresh blocks into `prefix_lru` after release.

Why not just `"<letter>" * 24000`? Tried it first; the OLMoE tokenizer is wildly inconsistent across single chars — `"a"*24000` and `"f"*24000` tokenize to 3000 tokens, but `"b"*24000` and most others tokenize to 4093 (over the 4096 model limit, request rejected). The numeric-tag scheme sidesteps the tokenizer quirk entirely.

### Phase 2 — expert-diverse drain

Goal: maximize expert misses, force the manager into the LRU-vs-LRU comparison at `unified_pool.py:661`, demonstrate cold prefix blocks lose to recently-used experts.

**Prompt**: `--dataset-name random --random-prefix-len 3200` — the workload we already measured at 64/64 experts per layer. Maximally diverse routing.

## Server config

```bash
VLLM_UNIFIED_POOL_TRACE=1 /home/belal/150326/venv-phase-2/bin/vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-unified-pool \
    --expert-cache-size 8 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-num-batched-tokens 1 \
    --max-model-len 4096 \
    --num-gpu-blocks-override 32 \
    --attention-backend TRITON_ATTN \
    --no-async-scheduling \
    &> /tmp/swap_test.log
```

### Why these specific values

- `--expert-cache-size 8` — top_k worth, the floor needed to process any token. Starting small leaves room for the cache to *grow* during Phase 2 (otherwise we wouldn't see growth, just constant churn within a fixed-size cache).
- `--num-gpu-blocks-override 32` per-layer — sized so:
  - Phase 1 fits comfortably: 8 expert + 1 null + 3 active KV + ~20 prefix slots = 32. We expect prefix to fill those 20 slots after enough requests.
  - Phase 2 *can't* fit all 64 experts in 32 slots, so expert misses must evict either prefix or expert. Cold prefix should lose first → that's the swap.

If the test fails to produce `tier=prefix-global` lines: pool is probably too generous. Halve to 24, retry. If the server crashes during Phase 1: pool too tight; raise to 40.

## 1. Phase 1 — fill the pool with prefix

12 sequential requests with distinct numeric tags. N=12 was chosen so 12 × 3 ≈ 36 candidate prefix blocks, more than enough to saturate the ~20 prefix slots a 32-block pool can hold per layer.

```bash
PHASE1_START=$(wc -l < /tmp/swap_test.log)
echo "phase1 start line: $PHASE1_START"

for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  PROMPT=$(python3 -c "print('req $n: ' + 'a' * 23900, end='')")
  curl -s http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{
      model: "allenai/OLMoE-1B-7B-0924-Instruct",
      prompt: $p,
      max_tokens: 4,
      temperature: 0
    }')" | jq -r '.choices[0].text'
done

PHASE1_END=$(wc -l < /tmp/swap_test.log)
echo "phase1 end line: $PHASE1_END"
echo "$PHASE1_START" > /tmp/swap_phase1_start.txt
echo "$PHASE1_END"   > /tmp/swap_phase1_end.txt
```

## 2. Mid-test snapshot — verify the pool is prefix-skewed

The `UNIFIED CACHE L<n>` line at the end of every layer's forward step prints occupancy in the format:

```
UNIFIED CACHE L<n> occ X/Y ours (expert-ours=A, expert-other=B, prefix=C, alloc-kv=D, pinned=E, free-pure=F)
```

Take the last such line per layer (its state right at end of Phase 1):

```bash
echo "Per-layer pool breakdown at end of Phase 1:"
for L in $(seq 0 15); do
  sed -n "$(cat /tmp/swap_phase1_start.txt),$(cat /tmp/swap_phase1_end.txt)p" /tmp/swap_test.log \
    | grep "UNIFIED CACHE L${L} " | tail -1
done
```

Also count Phase 1's expert footprint (sanity check that 12 distinct prefixes stayed concentrated):

```bash
sed -n "$(cat /tmp/swap_phase1_start.txt),$(cat /tmp/swap_phase1_end.txt)p" /tmp/swap_test.log \
  | grep "UNIFIED RESULT L" \
  | awk '{
      layer=$5;
      for (i=6; i<=NF; i++) {
        s=$i;
        while (match(s, /E[0-9]+/)) {
          print layer, substr(s, RSTART, RLENGTH);
          s=substr(s, RSTART+RLENGTH);
        }
      }
    }' | sort -u | wc -l
echo "(expect well below 1024; ideally under ~400)"
```

**Decision point.** If the per-layer breakdown shows `prefix` ≥ 15 and `free-pure` ≤ 3, Phase 1 succeeded — proceed to Phase 2. If `free-pure` is still large, the pool isn't pressured: either bump N higher (a–z instead of a–l) or shrink `--num-gpu-blocks-override` and restart.

## 3. Phase 2 — expert-diverse drain

```bash
PHASE2_START=$(wc -l < /tmp/swap_test.log)
echo "phase2 start line: $PHASE2_START"

vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-prefix-len 3200 \
    --random-input-len 1 \
    --random-output-len 4 \
    --random-range-ratio 0 \
    --num-prompts 6 \
    --max-concurrency 1 \
    --num-warmups 0 \
    --seed 42 \
    --trust-remote-code

PHASE2_END=$(wc -l < /tmp/swap_test.log)
echo "phase2 end line: $PHASE2_END"
echo "$PHASE2_START" > /tmp/swap_phase2_start.txt
echo "$PHASE2_END"   > /tmp/swap_phase2_end.txt
```

## 4. Final checks — did the swap actually happen?

### a. Pool breakdown after Phase 2 (compare against §2 above)

```bash
echo "Per-layer pool breakdown at end of Phase 2:"
for L in $(seq 0 15); do
  sed -n "$(cat /tmp/swap_phase2_start.txt),$(cat /tmp/swap_phase2_end.txt)p" /tmp/swap_test.log \
    | grep "UNIFIED CACHE L${L} " | tail -1
done
```

Expected delta vs Phase 1 end: `expert-ours` grew (e.g. 8 → 25), `prefix` shrunk toward 0.

### b. Count `tier=prefix-global` evictions during Phase 2

The single most important number — every prefix→expert swap is one of these.

```bash
sed -n "$(cat /tmp/swap_phase2_start.txt),$(cat /tmp/swap_phase2_end.txt)p" /tmp/swap_test.log \
  | grep -c "tier=prefix-global"
```

Expected: > 0, and ideally close to (Phase 1 prefix count) × 16 layers.

### c. For comparison, count other eviction tiers

```bash
for tier in free-pure free-cross-layer-expert expert-local prefix-global; do
  n=$(sed -n "$(cat /tmp/swap_phase2_start.txt),$(cat /tmp/swap_phase2_end.txt)p" /tmp/swap_test.log \
       | grep -c "tier=${tier}")
  echo "  tier=${tier}: ${n}"
done
```

Reading the mix: `free-pure` at the start of Phase 2 (one-shot — uses up remaining free), then `prefix-global` for a while (the swap proper), then `expert-local` once prefix is drained and the cache is just churning experts.

### d. Sanity: Phase 2 reached most of the expert space

```bash
sed -n "$(cat /tmp/swap_phase2_start.txt),$(cat /tmp/swap_phase2_end.txt)p" /tmp/swap_test.log \
  | grep "UNIFIED RESULT L" \
  | awk '{
      layer=$5;
      for (i=6; i<=NF; i++) {
        s=$i;
        while (match(s, /E[0-9]+/)) {
          print layer, substr(s, RSTART, RLENGTH);
          s=substr(s, RSTART+RLENGTH);
        }
      }
    }' | sort -u | wc -l
echo "(expect close to 1024 — random workload reaches all experts)"
```

## 5. Interpretation

| Result | Meaning |
|---|---|
| Phase 1 ends with `prefix` >> `expert-ours` and `free-pure ≈ 0`; Phase 2 produces many `tier=prefix-global` lines; final breakdown shows `expert-ours` >> Phase 1's value | **Working as designed.** Pool reshaped from prefix-dominant to expert-dominant under workload pressure. |
| Phase 1 ends with `prefix` small / `free-pure` large | Phase 1 didn't pressure the pool. Bump N (use a–z), or shrink `--num-gpu-blocks-override`, then restart. |
| Phase 2 produces zero `tier=prefix-global` despite many `UNIFIED RESULT L<n> ... misses=[...]` lines | Pool too generous — misses absorbed by free space. Shrink pool. |
| Phase 2 evictions are mostly `tier=expert-local` instead of `prefix-global` | Phase 1 experts aged out faster than prefix blocks (unexpected — prefix should be older). Likely cause: long gap between phases. Run Phase 2 immediately after Phase 1 finishes. |
| Phase 2 expert-footprint count is far below 1024 | Random workload did fewer requests than expected, or `vllm bench serve` failed silently. Check the bench output. |
| Server crashes during Phase 1 (`UnifiedPool ... pool exhausted`) | Pool too tight. Bump `--num-gpu-blocks-override` to 40 and retry. |

## 6. Tuning ladder (if first run is inconclusive)

In order of decreasing impact, try:

1. **Shrink pool first** (`--num-gpu-blocks-override 24`) — fastest way to force the swap to be visible.
2. **Bump Phase 1 N** to a–z (26 distinct prefixes) — saturates `prefix_lru` harder.
3. **Bump Phase 2 `--num-prompts`** to 12+ — drives more expert misses, sustains the LRU comparison longer.
4. **Drop `--expert-cache-size` to 4** — gives more starting room for prefix in Phase 1 (though may cause expert thrashing during Phase 1's own forward passes).

## 7. Reverse direction (future doc)

Mirror test: start with random / expert-heavy workload (cache fills with diverse experts, no prefix), then send long prefix-heavy traffic and watch experts get evicted in favor of KV. Structurally different because expert eviction in favor of KV happens via the KV allocation path, not the expert miss handler — different code path, separate test design.
