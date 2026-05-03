# Trace summary: `sweep_unified_from_bad_1A_seed1_g1_trace.log`

- File size: 14.0 MB
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

- `UNIFIED EVICT` total: **51**
- `UNIFIED KV_CLAIM` total: **10**
- `UNIFIED PREFIX_ADDED` total: 12
- `UNIFIED PREFIX_REMOVED` total: 8
- `UNIFIED CACHE` snapshots: 100128

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 32
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 2
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 8

## EVICT breakdown

By kind:
- `expert`: 51 (100.0%)

By cause:
- `kv-alloc`: 32 (62.7%)
- `expert-L2`: 4 (7.8%)
- `expert-L15`: 2 (3.9%)
- `expert-L7`: 2 (3.9%)
- `expert-L13`: 2 (3.9%)
- `expert-L5`: 2 (3.9%)
- `expert-L3`: 2 (3.9%)
- `expert-L4`: 1 (2.0%)
- `expert-L0`: 1 (2.0%)
- `expert-L11`: 1 (2.0%)
- `expert-L1`: 1 (2.0%)
- `expert-L6`: 1 (2.0%)

By tier:
- `kv-broadcast`: 32 (62.7%)
- `expert-local`: 19 (37.3%)

## KV_CLAIM breakdown by tier

- `truly-free`: 8 (80.0%)
- `kv-evicts-expert`: 2 (20.0%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 3 |
| L1 | 3 |
| L2 | 6 |
| L3 | 4 |
| L4 | 3 |
| L5 | 4 |
| L6 | 3 |
| L7 | 4 |
| L8 | 2 |
| L9 | 2 |
| L10 | 2 |
| L11 | 3 |
| L12 | 2 |
| L13 | 4 |
| L14 | 2 |
| L15 | 3 |

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
| L0 | 100112 | 62/68 | 62 | 2 | 2 | 2 |
| L1 | 100113 | 62/68 | 62 | 2 | 2 | 2 |
| L2 | 100114 | 62/68 | 62 | 2 | 2 | 2 |
| L3 | 100115 | 62/68 | 62 | 2 | 2 | 2 |
| L4 | 100116 | 62/68 | 62 | 2 | 2 | 2 |
| L5 | 100117 | 62/68 | 62 | 2 | 2 | 2 |
| L6 | 100118 | 62/68 | 62 | 2 | 2 | 2 |
| L7 | 100119 | 62/68 | 62 | 2 | 2 | 2 |
| L8 | 100120 | 62/68 | 62 | 2 | 2 | 2 |
| L9 | 100121 | 62/68 | 62 | 2 | 2 | 2 |
| L10 | 100122 | 62/68 | 62 | 2 | 2 | 2 |
| L11 | 100123 | 62/68 | 62 | 2 | 2 | 2 |
| L12 | 100124 | 62/68 | 62 | 2 | 2 | 2 |
| L13 | 100125 | 62/68 | 62 | 2 | 2 | 2 |
| L14 | 100126 | 62/68 | 62 | 2 | 2 | 2 |
| L15 | 100127 | 62/68 | 62 | 2 | 2 | 2 |

Mean across layers: expert-ours=62.0, prefix=2.0, alloc-kv=2.0, free-pure=2.0.
