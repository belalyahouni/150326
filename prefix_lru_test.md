# Test: does `PREFIX_LRU` populate, and does the manager actually evict prefix in favor of experts?

Goal: verify (a) that `_on_prefix_added` populates `prefix_lru`, (b) that the snapshot during a forward step actually shows entries (not just `of 0`), and (c) that at least some evictions choose `tier=prefix-global`.

**Why two prompts.** With one repeated prompt, every cached prefix block gets re-claimed the moment the next request schedules — `_on_prefix_removed` fires before our trace ever sees the entry. Two alternating prompts (A, B, A, B, …) keep one prompt's prefix idle in the cache while the other is mid-forward, so the trace snapshot taken inside `ensure_loaded` finally has something to show.

## 1. Server

Pool size 16 = top_k(8) + 1 null + 3 active KV blocks (current request) + 3 idle KV blocks (other request's prefix sitting in the cache) + 1 free margin. Tight enough that holding both prefixes pressures the expert side.

```bash
VLLM_UNIFIED_POOL_TRACE=1 vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
    --expert-offload \
    --expert-unified-pool \
    --expert-cache-size 8 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --max-num-batched-tokens 1 \
    --max-model-len 4096 \
    --num-gpu-blocks-override 16 \
    &> /tmp/prefix_lru_test_2.log
```

The `2>&1 | tee /tmp/prefix_lru_test.log` tail merges stderr into stdout (vLLM logs to stderr) and writes the combined stream to a file while still printing it live, so you can grep the file after the test.

## 2. Send the requests

In a second terminal. Two `vllm bench serve` commands that differ only in `--seed` — different seed → different random prefix tokens, so each command sends a distinct shared-prefix workload. The `random` dataset generates the prefix once per invocation (`vllm/benchmarks/datasets.py:508`), so all `--num-prompts 3` requests within a single command share the same 3200-token prefix; the two commands give you two different prefixes.

**Run command A:**

```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-prefix-len 3200 \
    --random-input-len 1 \
    --random-output-len 4 \
    --random-range-ratio 0 \
    --num-prompts 3 \
    --max-concurrency 1 \
    --num-warmups 0 \
    --seed 1 \
    --trust-remote-code
```

**Then run command B (same command, `--seed 2`):**

```bash
vllm bench serve \
    --backend vllm \
    --endpoint /v1/completions \
    --model allenai/OLMoE-1B-7B-0924-Instruct \
    --dataset-name random \
    --random-prefix-len 3200 \
    --random-input-len 1 \
    --random-output-len 4 \
    --random-range-ratio 0 \
    --num-prompts 3 \
    --max-concurrency 1 \
    --num-warmups 0 \
    --seed 2 \
    --trust-remote-code
```
  grep -E "UNIFIED PREFIX_ADDED|UNIFIED PREFIX_REMOVED" /tmp/prefix_lru_test.log | head -80

**Optional — repeat A, then B again** to get more alternation cycles. The lifecycle:

1. Cmd A run 1: prefix A misses cache, populates 3 KV blocks. After it finishes → those 3 blocks land in `prefix_lru`.
2. Cmd A runs 2 & 3: hit cache, blocks pulled out and put back. `prefix_lru` ends empty during forward, populated between requests.
3. Cmd B run 1: prefix B misses cache. **Prefix A's 3 blocks are sitting idle in `prefix_lru` during B's forward** → this is the moment the trace finally shows non-zero entries.
4. Cmd B runs 2 & 3: similar, but now A's prefix is one of the candidates the eviction logic compares against.
5. If you re-run Cmd A, its prefix has likely been evicted (or not, depending on pressure), so this round may show eviction-tier decisions in the log.

## 3. What to look for in `/tmp/prefix_lru_test.log`

After both commands finish, grep:

### a. `prefix_lru` actually shows entries during a forward

```bash
grep "UNIFIED PREFIX_LRU" /tmp/prefix_lru_test.log | grep -v "of 0]" | head
```

Expected: lines like `UNIFIED PREFIX_LRU MRU→LRU [top 8 of 3]: s4#h..., s5#h..., s6#h...`. The trace snapshots during command B's first forward should show 3 entries (command A's idle prefix blocks). If every line still says `of 0`, either the callback isn't firing or the pool is too tight and command A's prefix got evicted before B's first forward ran — bump pool to 18 and re-run.

### b. `tier=prefix-global` appears in evictions

```bash
grep "tier=prefix-global" /tmp/prefix_lru_test.log | head
```

Expected: at least one eviction line during command B (or a re-run of A) where the manager chose a stale prefix block over an expert. If zero, expert LRU stayed colder than prefix LRU throughout — try a tighter pool (15) or longer prefixes (`--random-prefix-len 4000`) to push more KV pressure.

### c. Prefix cache is actually hitting

```bash
grep "Prefix cache hit rate" /tmp/prefix_lru_test.log | tail
```

Expected: > 0% by the end. Confirms vLLM matched the second+ requests' prompts.

## 4. Interpretation

| Result | Meaning |
|---|---|
| `PREFIX_LRU [of N]` with N > 0 **and** `tier=prefix-global` lines present | Working as designed. Release-recency is enough; no hook needed. |
| `PREFIX_LRU [of N]` with N > 0 **but** no `tier=prefix-global` lines | LRU populates but expert tail always wins the comparison — workload is expert-hot, not a bug. Try squeezing the pool harder. |
| `PREFIX_LRU` always `of 0` | `_on_prefix_added` isn't firing, or entries are removed before any trace fires. Real plumbing bug — investigate. |
| Server crashes mid-test | Pool floor too tight. Bump `--num-gpu-blocks-override` to 14. |
