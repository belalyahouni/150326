# Problems

## Warm-up: disjoint vs. shared `block_id`s

**Before.** Stage-2 warm-up walked the pool once per layer and claimed a fresh `block_id` for each `(layer, expert)` pair, so the global namespace consumed at startup was `expert_cache_size × num_moe_layers`. For OLMoE with `--expert-cache-size 8` and 16 MoE layers, that is 128 ids before a single KV block can exist — the upfront capacity check at `gpu_model_runner.py:6822` enforced this exactly, making `--num-gpu-blocks-override 129` the minimum viable pool size.

**Problem.** The 129-block floor blocked the KV-hot scenario the dissertation depends on. To prove the LRU evicts cold experts to grow KV, we need to start from a small pool where KV pressure is real — ideally `~9` blocks, just enough to seat warm experts plus a single null block. The disjoint-id design also wasted the per-layer-buffer asymmetry: `block_id 47` already addresses 16 independent physical slots (one per layer), so giving each layer its own id during warm-up is a no-op in physical memory but an `N×` cost in namespace.

**Solution.** `UnifiedPoolManager.warm_up` now picks one `block_id` per warmed expert and DMAs that expert into the same id's slot in every layer. Total ids consumed at startup is `expert_cache_size`, the assertion at `gpu_model_runner.py:6822` is relaxed to `warm_count <= num_blocks_available`, and the minimum `--num-gpu-blocks-override` drops to `expert_cache_size + 1` (9 for the OLMoE config). The accepted trade-off: a kv-broadcast on a warm-shared id invalidates expert mappings in all 16 layers at once instead of one — fine, because warm-up is just startup seeding and the LRU reshapes from there.

choice:
1 mixed LRU (in blockpool)
1 mixed LRU (outside of blockpool)
2 lrus for expert and kv with steps (outside of blockpool)
1 expert LRU and adjsut existing vllm kv lru (in blockpool and outside of blockpool)

after phase 1 reaslied that k and v are actually split and so messes up the paging system

blockpool lru confused with unfiedpool so not evicting free pages by default
then only evicting free pages, doesnt consider experts, blockpool evicits expert instead of checjing unfieid pool lru.

if we do some prefix, then still have random recomputation it kicks prefix out. we cant for rprefix to stay, but we should weigh up the recomputation cost isntead of straight steps.