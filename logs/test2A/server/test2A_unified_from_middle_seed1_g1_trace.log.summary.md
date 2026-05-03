# Trace summary: `test2A_unified_from_middle_seed1_g1_trace.log`

- File size: 18.8 MB
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

- `UNIFIED EVICT` total: **74**
- `UNIFIED KV_CLAIM` total: **21**
- `UNIFIED PREFIX_ADDED` total: 22
- `UNIFIED PREFIX_REMOVED` total: 20
- `UNIFIED CACHE` snapshots: 133888

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 64
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 4
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 17

## EVICT breakdown

By kind:
- `expert`: 72 (97.3%)
- `kv-prefix`: 2 (2.7%)

By cause:
- `kv-alloc`: 64 (86.5%)
- `expert-L1`: 7 (9.5%)
- `expert-L2`: 3 (4.1%)

By tier:
- `kv-broadcast`: 64 (86.5%)
- `expert-local`: 8 (10.8%)
- `prefix-global`: 2 (2.7%)

## KV_CLAIM breakdown by tier

- `truly-free`: 17 (81.0%)
- `kv-evicts-expert`: 4 (19.0%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 4 |
| L1 | 9 |
| L2 | 7 |
| L3 | 4 |
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
- **Most KV-heavy moment**: ratio=0.951 at L1 step=49153 (expert-ours=39, alloc-kv=2)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L1 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L2 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L3 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L4 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L5 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L6 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L7 | 0.952 (eo=40, kv=2) | 1.000 (eo=40, kv=0) | 0.048 |
| L8 | 0.952 (eo=40, kv=2) | 1.000 (eo=40, kv=0) | 0.048 |
| L9 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L10 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L11 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L12 | 0.956 (eo=43, kv=2) | 1.000 (eo=40, kv=0) | 0.044 |
| L13 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L14 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L15 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 133872 | 64/68 | 64 | 2 | 0 | 2 |
| L1 | 133873 | 64/68 | 64 | 2 | 0 | 2 |
| L2 | 133874 | 64/68 | 64 | 2 | 0 | 2 |
| L3 | 133875 | 64/68 | 64 | 2 | 0 | 2 |
| L4 | 133876 | 64/68 | 64 | 2 | 0 | 2 |
| L5 | 133877 | 64/68 | 64 | 2 | 0 | 2 |
| L6 | 133878 | 64/68 | 64 | 2 | 0 | 2 |
| L7 | 133879 | 61/68 | 61 | 2 | 0 | 2 |
| L8 | 133880 | 64/68 | 64 | 2 | 0 | 2 |
| L9 | 133881 | 64/68 | 64 | 2 | 0 | 2 |
| L10 | 133882 | 63/68 | 63 | 2 | 0 | 2 |
| L11 | 133883 | 64/68 | 64 | 2 | 0 | 2 |
| L12 | 133884 | 64/68 | 64 | 2 | 0 | 2 |
| L13 | 133885 | 64/68 | 64 | 2 | 0 | 2 |
| L14 | 133886 | 64/68 | 64 | 2 | 0 | 2 |
| L15 | 133887 | 64/68 | 64 | 2 | 0 | 2 |

Mean across layers: expert-ours=63.8, prefix=2.0, alloc-kv=0.0, free-pure=2.0.
