# Trace summary: `manyprefix_unified_init40_seed1_g0_trace.log`

- File size: 139.7 MB
- Trace level: 1 (essential)
- Pool capacity (pages): 64

## Run config

- 'model_tag': 'allenai/OLMoE-1B-7B-0924-Instruct'
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

- `UNIFIED EVICT` total: **722**
- `UNIFIED KV_CLAIM` total: **60**
- `UNIFIED PREFIX_ADDED` total: 40
- `UNIFIED PREFIX_REMOVED` total: 26
- `UNIFIED CACHE` snapshots: 984640

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 640
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 40
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 20

## EVICT breakdown

By kind:
- `expert`: 696 (96.4%)
- `kv-prefix`: 26 (3.6%)

By cause:
- `kv-alloc`: 640 (88.6%)
- `expert-L1`: 60 (8.3%)
- `expert-L3`: 9 (1.2%)
- `expert-L0`: 8 (1.1%)
- `expert-L2`: 5 (0.7%)

By tier:
- `kv-broadcast`: 640 (88.6%)
- `expert-local`: 56 (7.8%)
- `prefix-global`: 26 (3.6%)

## KV_CLAIM breakdown by tier

- `kv-evicts-expert`: 40 (66.7%)
- `truly-free`: 20 (33.3%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 45 |
| L1 | 51 |
| L2 | 44 |
| L3 | 44 |
| L4 | 40 |
| L5 | 40 |
| L6 | 40 |
| L7 | 40 |
| L8 | 40 |
| L9 | 40 |
| L10 | 40 |
| L11 | 40 |
| L12 | 40 |
| L13 | 40 |
| L14 | 40 |
| L15 | 40 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=40, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.905 at L6 step=984566 (expert-ours=19, alloc-kv=2)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L1 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L2 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L3 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L4 | 0.931 (eo=27, kv=2) | 1.000 (eo=40, kv=0) | 0.069 |
| L5 | 0.913 (eo=21, kv=2) | 1.000 (eo=40, kv=0) | 0.087 |
| L6 | 0.905 (eo=19, kv=2) | 1.000 (eo=40, kv=0) | 0.095 |
| L7 | 0.905 (eo=19, kv=2) | 1.000 (eo=40, kv=0) | 0.095 |
| L8 | 0.913 (eo=21, kv=2) | 1.000 (eo=40, kv=0) | 0.087 |
| L9 | 0.939 (eo=31, kv=2) | 1.000 (eo=40, kv=0) | 0.061 |
| L10 | 0.931 (eo=27, kv=2) | 1.000 (eo=40, kv=0) | 0.069 |
| L11 | 0.931 (eo=27, kv=2) | 1.000 (eo=40, kv=0) | 0.069 |
| L12 | 0.931 (eo=27, kv=2) | 1.000 (eo=40, kv=0) | 0.069 |
| L13 | 0.926 (eo=25, kv=2) | 1.000 (eo=40, kv=0) | 0.074 |
| L14 | 0.933 (eo=28, kv=2) | 1.000 (eo=40, kv=0) | 0.067 |
| L15 | 0.938 (eo=30, kv=2) | 1.000 (eo=40, kv=0) | 0.062 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 984624 | 48/64 | 48 | 12 | 2 | 2 |
| L1 | 984625 | 48/64 | 48 | 12 | 2 | 2 |
| L2 | 984626 | 46/64 | 46 | 12 | 2 | 2 |
| L3 | 984627 | 43/64 | 43 | 12 | 2 | 2 |
| L4 | 984628 | 28/64 | 28 | 12 | 2 | 2 |
| L5 | 984629 | 22/64 | 22 | 12 | 2 | 2 |
| L6 | 984630 | 20/64 | 20 | 12 | 2 | 2 |
| L7 | 984631 | 19/64 | 19 | 12 | 2 | 2 |
| L8 | 984632 | 21/64 | 21 | 12 | 2 | 2 |
| L9 | 984633 | 31/64 | 31 | 12 | 2 | 2 |
| L10 | 984634 | 27/64 | 27 | 12 | 2 | 2 |
| L11 | 984635 | 28/64 | 28 | 12 | 2 | 2 |
| L12 | 984636 | 27/64 | 27 | 12 | 2 | 2 |
| L13 | 984637 | 27/64 | 27 | 12 | 2 | 2 |
| L14 | 984638 | 28/64 | 28 | 12 | 2 | 2 |
| L15 | 984639 | 33/64 | 33 | 12 | 2 | 2 |

Mean across layers: expert-ours=31.0, prefix=12.0, alloc-kv=2.0, free-pure=2.0.
