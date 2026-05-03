# Trace summary: `test2A_unified_from_expert_seed1_g1_trace.log`

- File size: 18.7 MB
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
- 'expert_cache_size': 64
- 'expert_unified_pool': True
- 'max_num_batched_tokens': 1
- 'async_scheduling': False

## Eviction & KV-claim totals

- `UNIFIED EVICT` total: **84**
- `UNIFIED KV_CLAIM` total: **21**
- `UNIFIED PREFIX_ADDED` total: 22
- `UNIFIED PREFIX_REMOVED` total: 20
- `UNIFIED CACHE` snapshots: 133888

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 32
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 2
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 19

## EVICT breakdown

By kind:
- `expert`: 82 (97.6%)
- `kv-prefix`: 2 (2.4%)

By cause:
- `kv-alloc`: 32 (38.1%)
- `expert-L1`: 11 (13.1%)
- `expert-L2`: 7 (8.3%)
- `expert-L3`: 7 (8.3%)
- `expert-L4`: 4 (4.8%)
- `expert-L11`: 4 (4.8%)
- `expert-L15`: 3 (3.6%)
- `expert-L7`: 3 (3.6%)
- `expert-L13`: 2 (2.4%)
- `expert-L0`: 2 (2.4%)
- `expert-L5`: 2 (2.4%)
- `expert-L6`: 2 (2.4%)
- `expert-L14`: 2 (2.4%)
- `expert-L10`: 1 (1.2%)
- `expert-L8`: 1 (1.2%)
- `expert-L12`: 1 (1.2%)

By tier:
- `expert-local`: 50 (59.5%)
- `kv-broadcast`: 32 (38.1%)
- `prefix-global`: 2 (2.4%)

## KV_CLAIM breakdown by tier

- `truly-free`: 19 (90.5%)
- `kv-evicts-expert`: 2 (9.5%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 4 |
| L1 | 10 |
| L2 | 9 |
| L3 | 9 |
| L4 | 6 |
| L5 | 4 |
| L6 | 4 |
| L7 | 5 |
| L8 | 3 |
| L9 | 2 |
| L10 | 3 |
| L11 | 6 |
| L12 | 3 |
| L13 | 4 |
| L14 | 4 |
| L15 | 4 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=64, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.969 at L0 step=98848 (expert-ours=62, alloc-kv=2)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L1 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L2 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L3 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L4 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L5 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L6 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L7 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L8 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L9 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L10 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L11 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L12 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L13 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L14 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |
| L15 | 0.969 (eo=62, kv=2) | 1.000 (eo=64, kv=0) | 0.031 |

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
| L7 | 133879 | 64/68 | 64 | 2 | 0 | 2 |
| L8 | 133880 | 64/68 | 64 | 2 | 0 | 2 |
| L9 | 133881 | 64/68 | 64 | 2 | 0 | 2 |
| L10 | 133882 | 64/68 | 64 | 2 | 0 | 2 |
| L11 | 133883 | 64/68 | 64 | 2 | 0 | 2 |
| L12 | 133884 | 64/68 | 64 | 2 | 0 | 2 |
| L13 | 133885 | 64/68 | 64 | 2 | 0 | 2 |
| L14 | 133886 | 64/68 | 64 | 2 | 0 | 2 |
| L15 | 133887 | 64/68 | 64 | 2 | 0 | 2 |

Mean across layers: expert-ours=64.0, prefix=2.0, alloc-kv=0.0, free-pure=2.0.
