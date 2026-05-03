# Unified pool — closing the KV-side allocation gap

## The problem

The unified pool's whole premise is that experts and KV blocks share one physical pool, with eviction decisions made by comparing the value of each candidate (cold expert vs cold prefix). The current implementation only does this **for expert misses**. KV allocation bypasses unified-pool decision-making entirely — vLLM's standard `BlockPool.get_new_blocks` (`block_pool.py:351`) does a pure FIFO `popleft_n` from the front of the free queue, regardless of what each block holds. The unified pool only learns about the choice *after the fact*, via the observational `_on_kv_allocation` callback (`unified_pool.py:376`).

Concretely:

| Allocation path | Goes through `_select_victim_block`? | Respects free-pure first? |
|---|---|---|
| Expert miss | ✅ Yes (`unified_pool.py:612`) | ✅ Yes — Tier 1 skips prefix-tagged blocks |
| KV allocation | ❌ No — vLLM standard FIFO `popleft_n` | ❌ No — pure popleft |

### Observed consequence

Phase 1 of the swap test (12 distinct long-prefix requests, pool=64 per layer) ended with `prefix=4 per layer` and `free-pure=5 per layer`. Twelve requests deposited 12 prefix blocks; only 4 survived. The 8 lost prefix blocks were silently cannibalized by KV allocation pulling them from the front of the LRU queue — even though free-pure blocks were sitting at the back. **`tier=prefix-global` evictions during Phase 1: 0** (the unified pool's swap logic never fired; the standard allocator made all the relevant decisions).

The mechanism (`block_pool.py:378` `_maybe_evict_cached_block`): every block returned by `popleft_n` gets its hash silently cleared, firing `_on_prefix_removed` which removes the block from `prefix_lru`. The unified pool can't intervene — it only gets notified.

## Why this matters for the dissertation

The dissertation idea is **bidirectional**: at any moment, evict whichever side is less valuable. The current implementation is one-directional:

- Expert misses can sacrifice cold prefix (works — `tier=prefix-global`).
- KV allocation cannot weigh prefix vs experts. It just grabs the front of the FIFO.

Without closing this gap, "the unified pool dynamically rebalances" is only half-true. Workloads with high KV churn (many short-lived requests with distinct prefixes) destroy prefix cache value even when the pool has slack, because the standard allocator has no concept of "prefix is worth keeping if there's an alternative."

## A deeper insight that collapses the design

Initial sketch was a two-phase fix: (A) make KV prefer "no-hash" blocks over prefix-tagged ones, (B) later add LRU comparison between cold experts and cold prefix. Phase A was implemented and tested against the swap workload — it eliminated prefix cannibalization (`PREFIX_REMOVED` dropped from 7 → 0 with pool=64, 12 prompts).

But a focused follow-up test (cache-size=64, pool=66, 4 prompts) revealed Phase A is incomplete. Trace:

- `PREFIX_ADDED=4, PREFIX_REMOVED=0` ✅ (prefix preserved)
- `EVICT kind=expert ... cause=kv-alloc tier=kv-broadcast`: 128 events ⚠ (KV silently dropped expert mappings)
- `expert-ours` per layer dropped from 64 → 58–60

**Root cause:** warmup (`unified_pool.py:469`) appends each just-warmed block back to the free queue. From the block pool's view, expert-occupied blocks and truly-free blocks both have `block_hash=None` — indistinguishable. Phase A grabs the first no-hash block at the queue front, which is often an expert-occupied block. The post-allocation broadcast (`_on_kv_allocation`) silently drops every layer's expert mapping at that block.

So Phase A made two errors in one:
1. **It treated expert-occupied blocks as if they were free.** The block-pool layer simply doesn't know what experts are.
2. **The expert evictions it caused were arbitrary.** Whichever expert happened to be at the queue front got dropped — independent of whether it was hot or cold. In our test it was E0–E3 (warmed earliest, never re-touched), but only by accident of warmup ordering.

Once you tighten Phase A to also skip expert-occupied blocks (preferring truly-free), you've already implemented half of Phase B. And once truly-free runs out, you must compare cold expert vs cold prefix to pick the right victim — which *is* Phase B. So the sensible design is a single consolidated solution.

## Solution: route KV victim selection through the unified pool

### Architecture

- `BlockPool.register_kv_victim_selector(callback)` — register a callable that returns the next N victim blocks (already dequeued). When set, `get_new_blocks(n)` calls `callback(n)` instead of `popleft_n(n)`. The post-selection housekeeping is unchanged: `_maybe_evict_cached_block` still clears prefix hashes for any returned blocks that have one, `_on_allocation_callbacks` still broadcasts expert eviction for any returned blocks that hold experts. The selector decides *which* blocks; the existing machinery handles the consequences correctly regardless.
- `UnifiedPoolManager` registers `_select_kv_victim_blocks` as the callback during init.

### Selector logic

For each of N blocks needed, do this once:

1. **Tier 1 — truly free.** Walk the free queue front-to-back. First block where:
   - `block.block_hash is None` (no prefix), AND
   - `block_holder.get(block_id, set())` is empty (no expert mapping in any layer), AND
   - `block_id` not in any layer's `pinned_blocks`.

   If found: dequeue and return. Cost = 0 (no eviction).

2. **Tier 2 — LRU compare.** Otherwise:
   - **Oldest expert (any layer):** scan each of the 16 `expert_lru` first entries (oldest per layer); take the min-step. Skip pinned. The block holding that expert is the candidate.
   - **Oldest prefix:** `prefix_lru` first entry. Skip pinned.
   
   Compare steps; take the colder. If expert: `_on_kv_allocation`'s broadcast (already wired into `get_new_blocks` post-selection) drops all layers' expert mappings at the block. If prefix: `_maybe_evict_cached_block` clears the hash and fires `_on_prefix_removed`. Either way, the existing post-selection housekeeping in `get_new_blocks` Just Works.

3. **Tier 3 — fail.** Neither side has a non-pinned candidate → raise. Caller (scheduler) handles preemption.

### Why this fixes everything

- **No more accidental expert eviction.** Tier 1 only takes truly-free blocks (no hash AND no expert mappings).
- **Expert eviction becomes deliberate.** Tier 2 picks the LRU-oldest expert across all layers, not whichever happens to be at the queue front.
- **Bidirectional.** Cold expert vs cold prefix is now a real comparison in both directions:
  - Expert miss path (`_select_victim_block`, exists today): when expert needs space, pick colder of cold-prefix vs cold-expert.
  - KV demand path (`_select_kv_victim_blocks`, this design): when KV needs space, pick colder of cold-prefix vs cold-expert.
- **Symmetric with the existing expert-side selector.** Both follow the same Tier 1 / Tier 2 / Tier 3 structure; the only difference is which "mine" the requestor counts as (KV doesn't have a layer; experts do).

### Phase A removal

- `BlockPool._prefer_no_hash_for_kv` flag — removed.
- `BlockPool._popleft_n_prefer_no_hash` method — removed.
- `BlockPool.enable_prefer_no_hash_for_kv` method — removed.
- `UnifiedPoolManager.__init__` call to `enable_prefer_no_hash_for_kv` — removed.

The new selector subsumes everything Phase A did, and does it correctly.

## Test plan

### Quick verification — focused test

**Server:** `--expert-cache-size 64 --num-gpu-blocks-override 66`. After warmup: 64 expert blocks, 1 null, ~1 truly-free.

**Workload:** 4 distinct `req $n: ` + `"a" * 23900` prompts.

**Expected:**
- `PREFIX_ADDED = 4, PREFIX_REMOVED = 0` (prefix never evicted — Tier 2 always picks step=0 expert as colder).
- Expert evictions occur, but they are the *oldest* experts (step=0), and the count is bounded by KV demand (1 KV block per request → 4 expert evictions per layer maximum).
- The eviction trace should now identify a victim by intentional choice, not by queue order.

If `PREFIX_REMOVED > 0`: the LRU comparison is wrong (prefix is being picked as colder than experts somehow). Investigate the step values.

If `EVICT kind=expert` count is ≪ 128 (the Phase A baseline) but > 0: Tier 1 is finding truly-free blocks for the first request or two, then Tier 2 starts evicting cold experts. That's correct behavior.

### Full swap test re-verification

After the focused test passes, re-run `swap_test_phase2.md` Phase 1 (pool=64, expert-cache-size=8, 12 distinct-prefix prompts) and Phase 2 (random workload). The Phase 1 expectation (prefix accumulates, no cannibalization) should still hold; Phase 2 should now show:
- `tier=prefix-global` evictions during expert miss path (existing behavior, unchanged).
- The new selector firing during KV allocation, with correct LRU choices.

## Results — focused test (run 2026-05-03)

Server: `--expert-cache-size 64 --num-gpu-blocks-override 66 --enable-prefix-caching --max-num-batched-tokens 1`. After warmup: 64 expert blocks per layer, 1 null, 1 truly-free.

Workload: 4 sequential `req $n: ` + `"a" * 23900` prompts (~3000 tokens each).

| Metric | Phase A baseline | New unified selector |
|---|---|---|
| `PREFIX_ADDED` | 4 | 4 |
| `PREFIX_REMOVED` | 0 | 0 |
| `EVICT kind=expert` (per-layer broadcast events) | 128 | 93 |
| KV claims labeled `tier=truly-free` | (not labeled) | **4** |
| KV claims labeled `tier=kv-evicts-expert` | (not labeled) | **4** |
| KV claims labeled `tier=kv-evicts-prefix` | (not labeled) | **0** |
| `expert-ours` per layer at end | 58–60 | 60 |

**Eviction-trace pattern.** Each request takes 2 blocks. The first claim of each request hits Tier 1 (`truly-free` — the released block from the previous request, hash-cleared); the second falls into Tier 2 and the LRU comparison correctly picks an expert at step=0 over the only available prefix at step ≫ 0. The actual `EVICT` lines confirm this: `page=1 ... E0 cause=kv-alloc tier=kv-broadcast` (E0 was warmed first, never re-touched).

**The eviction is now deliberate, not arbitrary.** Phase A also evicted E0–E3 in this scenario, but only by accident of warmup queue order. With the new selector, the chosen victim is the LRU-oldest expert by design — if hot experts were at the queue front and cold ones at the back, the new selector would still correctly pick the cold ones.

## Implementation summary

Two files changed (against the Phase A intermediate state):

**`vllm/v1/core/block_pool.py`:**
- Removed: `_prefer_no_hash_for_kv` flag, `_popleft_n_prefer_no_hash` helper, `enable_prefer_no_hash_for_kv` setter (Phase A scaffolding, now subsumed).
- Added: `_kv_victim_selector: Callable[[int], list[KVCacheBlock]] | None` and `register_kv_victim_selector(callback)`.
- Modified `get_new_blocks` to delegate to the registered selector when set, falling back to `popleft_n` otherwise.

**`vllm/model_executor/layers/fused_moe/unified_pool.py`:**
- Removed: the `enable_prefer_no_hash_for_kv()` call in `__init__`.
- Added: `register_kv_victim_selector(self._select_kv_victim_blocks)` call in `__init__`.
- Added: `_select_kv_victim_blocks(num_blocks)` (loop wrapper), `_pick_one_kv_victim()` (Tier 1 / Tier 2 / Tier 3 logic), `_oldest_global_expert()` and `_any_layer_pins(block_id)` (helpers).
- Added trace logging: `UNIFIED KV_CLAIM page=X tier={truly-free|kv-evicts-expert|kv-evicts-prefix}`.

Standard vLLM (no unified pool): no behavior change. The selector is only registered when the unified pool is active; otherwise `get_new_blocks` falls back to standard `popleft_n`.
