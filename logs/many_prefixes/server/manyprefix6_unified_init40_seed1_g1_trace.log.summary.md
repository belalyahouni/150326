# Trace summary: `manyprefix6_unified_init40_seed1_g1_trace.log`

- File size: 41.8 MB
- Trace level: 1 (essential)
- Pool capacity (pages): 64

## Run config

- 'model_tag': 'allenai/OLMoE-1B-7B-0924-Instruct'
- 'port': 8001
- 'model': 'allenai/OLMoE-1B-7B-0924-Instruct'
- 'trust_remote_code': True
- 'max_model_len': 4096
- 'enforce_eager': True
- 'attention_backend': 'TRITON_ATTN'
- 'block_size': 1536
- 'num_gpu_blocks_override': 64
- 'enable_prefix_caching': True
- 'expert_offload': True
- 'expert_cache_size': 40
- 'expert_unified_pool': True
- 'max_num_batched_tokens': 1
- 'async_scheduling': False

## Eviction & KV-claim totals

- `UNIFIED EVICT` total: **232**
- `UNIFIED KV_CLAIM` total: **24**
- `UNIFIED PREFIX_ADDED` total: 24
- `UNIFIED PREFIX_REMOVED` total: 12
- `UNIFIED CACHE` snapshots: 295872

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 192
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 12
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 12

## EVICT breakdown

By kind:
- `expert`: 232 (100.0%)

By cause:
- `kv-alloc`: 192 (82.8%)
- `expert-L1`: 20 (8.6%)
- `expert-L3`: 9 (3.9%)
- `expert-L2`: 7 (3.0%)
- `expert-L0`: 4 (1.7%)

By tier:
- `kv-broadcast`: 192 (82.8%)
- `expert-local`: 40 (17.2%)

## KV_CLAIM breakdown by tier

- `truly-free`: 12 (50.0%)
- `kv-evicts-expert`: 12 (50.0%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 16 |
| L1 | 31 |
| L2 | 19 |
| L3 | 21 |
| L4 | 12 |
| L5 | 12 |
| L6 | 12 |
| L7 | 12 |
| L8 | 12 |
| L9 | 12 |
| L10 | 12 |
| L11 | 12 |
| L12 | 12 |
| L13 | 12 |
| L14 | 12 |
| L15 | 12 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=40, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.949 at L7 step=295319 (expert-ours=37, alloc-kv=2)

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
| L7 | 0.949 (eo=37, kv=2) | 1.000 (eo=40, kv=0) | 0.051 |
| L8 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L9 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L10 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L11 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L12 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L13 | 0.953 (eo=41, kv=2) | 1.000 (eo=40, kv=0) | 0.047 |
| L14 | 0.952 (eo=40, kv=2) | 1.000 (eo=40, kv=0) | 0.048 |
| L15 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 295856 | 50/64 | 50 | 10 | 2 | 2 |
| L1 | 295857 | 50/64 | 50 | 10 | 2 | 2 |
| L2 | 295858 | 50/64 | 50 | 10 | 2 | 2 |
| L3 | 295859 | 50/64 | 50 | 10 | 2 | 2 |
| L4 | 295860 | 40/64 | 40 | 10 | 2 | 2 |
| L5 | 295861 | 45/64 | 45 | 10 | 2 | 2 |
| L6 | 295862 | 41/64 | 41 | 10 | 2 | 2 |
| L7 | 295863 | 37/64 | 37 | 10 | 2 | 2 |
| L8 | 295864 | 39/64 | 39 | 10 | 2 | 2 |
| L9 | 295865 | 47/64 | 47 | 10 | 2 | 2 |
| L10 | 295866 | 43/64 | 43 | 10 | 2 | 2 |
| L11 | 295867 | 46/64 | 46 | 10 | 2 | 2 |
| L12 | 295868 | 45/64 | 45 | 10 | 2 | 2 |
| L13 | 295869 | 44/64 | 44 | 10 | 2 | 2 |
| L14 | 295870 | 44/64 | 44 | 10 | 2 | 2 |
| L15 | 295871 | 45/64 | 45 | 10 | 2 | 2 |

Mean across layers: expert-ours=44.8, prefix=10.0, alloc-kv=2.0, free-pure=2.0.
