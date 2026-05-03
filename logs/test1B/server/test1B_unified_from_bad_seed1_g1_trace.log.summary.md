# Trace summary: `test1B_unified_from_bad_seed1_g1_trace.log`

- File size: 6.7 MB
- Trace level: 1 (essential)
- Pool capacity (pages): 68

## Run config

- 'model_tag': 'allenai/OLMoE-1B-7B-0924-Instruct'
- 'port': 8001
- 'model': 'allenai/OLMoE-1B-7B-0924-Instruct'
- 'trust_remote_code': True
- 'max_model_len': 4096
- 'enforce_eager': True
- 'attention_backend': 'TRITON_ATTN'
- 'block_size': 1536
- 'num_gpu_blocks_override': 68
- 'enable_prefix_caching': True
- 'expert_offload': True
- 'expert_cache_size': 16
- 'expert_unified_pool': True
- 'max_num_batched_tokens': 1
- 'async_scheduling': False

## Eviction & KV-claim totals

- `UNIFIED EVICT` total: **0**
- `UNIFIED KV_CLAIM` total: **9**
- `UNIFIED PREFIX_ADDED` total: 0
- `UNIFIED PREFIX_REMOVED` total: 0
- `UNIFIED CACHE` snapshots: 48240

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 0
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 0
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 9

## KV_CLAIM breakdown by tier

- `truly-free`: 9 (100.0%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=16, alloc-kv=0)
- **Most KV-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=16, alloc-kv=0)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L1 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L2 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L3 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L4 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L5 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L6 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L7 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L8 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L9 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L10 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L11 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L12 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L13 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L14 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |
| L15 | 1.000 (eo=16, kv=0) | 1.000 (eo=16, kv=0) | 0.000 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 48224 | 64/68 | 64 | 0 | 0 | 2 |
| L1 | 48225 | 64/68 | 64 | 0 | 0 | 2 |
| L2 | 48226 | 64/68 | 64 | 0 | 0 | 2 |
| L3 | 48227 | 64/68 | 64 | 0 | 0 | 2 |
| L4 | 48228 | 64/68 | 64 | 0 | 0 | 2 |
| L5 | 48229 | 64/68 | 64 | 0 | 0 | 2 |
| L6 | 48230 | 64/68 | 64 | 0 | 0 | 2 |
| L7 | 48231 | 60/68 | 60 | 0 | 0 | 2 |
| L8 | 48232 | 61/68 | 61 | 0 | 0 | 2 |
| L9 | 48233 | 59/68 | 59 | 0 | 0 | 2 |
| L10 | 48234 | 60/68 | 60 | 0 | 0 | 2 |
| L11 | 48235 | 60/68 | 60 | 0 | 0 | 2 |
| L12 | 48236 | 64/68 | 64 | 0 | 0 | 2 |
| L13 | 48237 | 63/68 | 63 | 0 | 0 | 2 |
| L14 | 48238 | 62/68 | 62 | 0 | 0 | 2 |
| L15 | 48239 | 61/68 | 61 | 0 | 0 | 2 |

Mean across layers: expert-ours=62.4, prefix=0.0, alloc-kv=0.0, free-pure=2.0.
