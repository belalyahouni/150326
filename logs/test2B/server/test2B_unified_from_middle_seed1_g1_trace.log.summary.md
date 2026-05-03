# Trace summary: `test2B_unified_from_middle_seed1_g1_trace.log`

- File size: 19.5 MB
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
- 'expert_cache_size': 40
- 'expert_unified_pool': True
- 'max_num_batched_tokens': 1
- 'async_scheduling': False

## Eviction & KV-claim totals

- `UNIFIED EVICT` total: **79**
- `UNIFIED KV_CLAIM` total: **21**
- `UNIFIED PREFIX_ADDED` total: 20
- `UNIFIED PREFIX_REMOVED` total: 16
- `UNIFIED CACHE` snapshots: 138992

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 64
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 4
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 17

## EVICT breakdown

By kind:
- `expert`: 79 (100.0%)

By cause:
- `kv-alloc`: 64 (81.0%)
- `expert-L1`: 5 (6.3%)
- `expert-L2`: 4 (5.1%)
- `expert-L0`: 4 (5.1%)
- `expert-L3`: 2 (2.5%)

By tier:
- `kv-broadcast`: 64 (81.0%)
- `expert-local`: 15 (19.0%)

## KV_CLAIM breakdown by tier

- `truly-free`: 17 (81.0%)
- `kv-evicts-expert`: 4 (19.0%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 8 |
| L1 | 9 |
| L2 | 8 |
| L3 | 6 |
| L4 | 4 |
| L5 | 4 |
| L6 | 4 |
| L7 | 4 |
| L8 | 4 |
| L9 | 4 |
| L10 | 4 |
| L11 | 4 |
| L12 | 4 |
| L13 | 4 |
| L14 | 4 |
| L15 | 4 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=40, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.966 at L7 step=136151 (expert-ours=56, alloc-kv=2)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L1 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L2 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L3 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L4 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L5 | 0.968 (eo=60, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L6 | 0.968 (eo=60, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L7 | 0.966 (eo=56, kv=2) | 1.000 (eo=40, kv=0) | 0.034 |
| L8 | 0.967 (eo=59, kv=2) | 1.000 (eo=40, kv=0) | 0.033 |
| L9 | 0.966 (eo=57, kv=2) | 1.000 (eo=40, kv=0) | 0.034 |
| L10 | 0.967 (eo=58, kv=2) | 1.000 (eo=40, kv=0) | 0.033 |
| L11 | 0.967 (eo=59, kv=2) | 1.000 (eo=40, kv=0) | 0.033 |
| L12 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L13 | 0.968 (eo=60, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L14 | 0.968 (eo=61, kv=2) | 1.000 (eo=40, kv=0) | 0.032 |
| L15 | 0.967 (eo=59, kv=2) | 1.000 (eo=40, kv=0) | 0.033 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 138976 | 62/68 | 62 | 2 | 2 | 2 |
| L1 | 138977 | 62/68 | 62 | 2 | 2 | 2 |
| L2 | 138978 | 62/68 | 62 | 2 | 2 | 2 |
| L3 | 138979 | 62/68 | 62 | 2 | 2 | 2 |
| L4 | 138980 | 62/68 | 62 | 2 | 2 | 2 |
| L5 | 138981 | 61/68 | 61 | 2 | 2 | 2 |
| L6 | 138982 | 61/68 | 61 | 2 | 2 | 2 |
| L7 | 138983 | 56/68 | 56 | 2 | 2 | 2 |
| L8 | 138984 | 59/68 | 59 | 2 | 2 | 2 |
| L9 | 138985 | 58/68 | 58 | 2 | 2 | 2 |
| L10 | 138986 | 60/68 | 60 | 2 | 2 | 2 |
| L11 | 138987 | 61/68 | 61 | 2 | 2 | 2 |
| L12 | 138988 | 62/68 | 62 | 2 | 2 | 2 |
| L13 | 138989 | 62/68 | 62 | 2 | 2 | 2 |
| L14 | 138990 | 61/68 | 61 | 2 | 2 | 2 |
| L15 | 138991 | 60/68 | 60 | 2 | 2 | 2 |

Mean across layers: expert-ours=60.7, prefix=2.0, alloc-kv=2.0, free-pure=2.0.
