# Trace summary: `test2B_unified_from_expert_seed1_g1_trace.log`

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
- 'expert_cache_size': 64
- 'expert_unified_pool': True
- 'max_num_batched_tokens': 1
- 'async_scheduling': False

## Eviction & KV-claim totals

- `UNIFIED EVICT` total: **63**
- `UNIFIED KV_CLAIM` total: **21**
- `UNIFIED PREFIX_ADDED` total: 20
- `UNIFIED PREFIX_REMOVED` total: 16
- `UNIFIED CACHE` snapshots: 138992

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 32
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 2
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 19

## EVICT breakdown

By kind:
- `expert`: 63 (100.0%)

By cause:
- `kv-alloc`: 32 (50.8%)
- `expert-L0`: 6 (9.5%)
- `expert-L1`: 6 (9.5%)
- `expert-L2`: 4 (6.3%)
- `expert-L14`: 2 (3.2%)
- `expert-L9`: 2 (3.2%)
- `expert-L3`: 2 (3.2%)
- `expert-L15`: 2 (3.2%)
- `expert-L11`: 2 (3.2%)
- `expert-L7`: 1 (1.6%)
- `expert-L12`: 1 (1.6%)
- `expert-L10`: 1 (1.6%)
- `expert-L4`: 1 (1.6%)
- `expert-L13`: 1 (1.6%)

By tier:
- `kv-broadcast`: 32 (50.8%)
- `expert-local`: 31 (49.2%)

## KV_CLAIM breakdown by tier

- `truly-free`: 19 (90.5%)
- `kv-evicts-expert`: 2 (9.5%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 7 |
| L1 | 8 |
| L2 | 6 |
| L3 | 4 |
| L4 | 3 |
| L5 | 2 |
| L6 | 2 |
| L7 | 3 |
| L8 | 2 |
| L9 | 4 |
| L10 | 3 |
| L11 | 4 |
| L12 | 3 |
| L13 | 3 |
| L14 | 4 |
| L15 | 4 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=64, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.969 at L0 step=136144 (expert-ours=62, alloc-kv=2)

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
| L0 | 138976 | 62/68 | 62 | 2 | 2 | 2 |
| L1 | 138977 | 62/68 | 62 | 2 | 2 | 2 |
| L2 | 138978 | 62/68 | 62 | 2 | 2 | 2 |
| L3 | 138979 | 62/68 | 62 | 2 | 2 | 2 |
| L4 | 138980 | 62/68 | 62 | 2 | 2 | 2 |
| L5 | 138981 | 62/68 | 62 | 2 | 2 | 2 |
| L6 | 138982 | 62/68 | 62 | 2 | 2 | 2 |
| L7 | 138983 | 62/68 | 62 | 2 | 2 | 2 |
| L8 | 138984 | 62/68 | 62 | 2 | 2 | 2 |
| L9 | 138985 | 62/68 | 62 | 2 | 2 | 2 |
| L10 | 138986 | 62/68 | 62 | 2 | 2 | 2 |
| L11 | 138987 | 62/68 | 62 | 2 | 2 | 2 |
| L12 | 138988 | 62/68 | 62 | 2 | 2 | 2 |
| L13 | 138989 | 62/68 | 62 | 2 | 2 | 2 |
| L14 | 138990 | 62/68 | 62 | 2 | 2 | 2 |
| L15 | 138991 | 62/68 | 62 | 2 | 2 | 2 |

Mean across layers: expert-ours=62.0, prefix=2.0, alloc-kv=2.0, free-pure=2.0.
