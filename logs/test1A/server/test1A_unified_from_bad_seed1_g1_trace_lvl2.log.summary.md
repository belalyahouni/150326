# Trace summary: `test1A_unified_from_bad_seed1_g1_trace_lvl2.log`

- File size: 162.1 MB
- Trace level: 2 (verbose)
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

- `UNIFIED EVICT` total: **70**
- `UNIFIED KV_CLAIM` total: **25**
- `UNIFIED PREFIX_ADDED` total: 42
- `UNIFIED PREFIX_REMOVED` total: 38
- `UNIFIED CACHE` snapshots: 104928
- `UNIFIED REQUEST` lines (lvl2): 104928

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 32
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 2
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 23

## EVICT breakdown

By kind:
- `expert`: 70 (100.0%)

By cause:
- `kv-alloc`: 32 (45.7%)
- `expert-L0`: 6 (8.6%)
- `expert-L3`: 6 (8.6%)
- `expert-L2`: 5 (7.1%)
- `expert-L1`: 4 (5.7%)
- `expert-L15`: 3 (4.3%)
- `expert-L7`: 3 (4.3%)
- `expert-L4`: 2 (2.9%)
- `expert-L13`: 2 (2.9%)
- `expert-L5`: 2 (2.9%)
- `expert-L11`: 2 (2.9%)
- `expert-L6`: 1 (1.4%)
- `expert-L10`: 1 (1.4%)
- `expert-L8`: 1 (1.4%)

By tier:
- `expert-local`: 38 (54.3%)
- `kv-broadcast`: 32 (45.7%)

## KV_CLAIM breakdown by tier

- `truly-free`: 23 (92.0%)
- `kv-evicts-expert`: 2 (8.0%)

## Expert diversity per layer

Distinct experts observed in `UNIFIED REQUEST` (full router demand). Each layer has 64 experts total.

| Layer | distinct requested |
|---|---|
| L0 | 63 |
| L1 | 62 |
| L2 | 60 |
| L3 | 60 |
| L4 | 40 |
| L5 | 37 |
| L6 | 28 |
| L7 | 27 |
| L8 | 28 |
| L9 | 38 |
| L10 | 39 |
| L11 | 43 |
| L12 | 40 |
| L13 | 42 |
| L14 | 41 |
| L15 | 43 |

Experts ever evicted per layer (lower bound on churn):

| Layer | distinct evicted |
|---|---|
| L0 | 7 |
| L1 | 6 |
| L2 | 7 |
| L3 | 8 |
| L4 | 4 |
| L5 | 4 |
| L6 | 3 |
| L7 | 5 |
| L8 | 3 |
| L9 | 2 |
| L10 | 3 |
| L11 | 4 |
| L12 | 2 |
| L13 | 4 |
| L14 | 2 |
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
| L0 | 104912 | 62/68 | 62 | 2 | 2 | 2 |
| L1 | 104913 | 62/68 | 62 | 2 | 2 | 2 |
| L2 | 104914 | 62/68 | 62 | 2 | 2 | 2 |
| L3 | 104915 | 62/68 | 62 | 2 | 2 | 2 |
| L4 | 104916 | 62/68 | 62 | 2 | 2 | 2 |
| L5 | 104917 | 62/68 | 62 | 2 | 2 | 2 |
| L6 | 104918 | 62/68 | 62 | 2 | 2 | 2 |
| L7 | 104919 | 62/68 | 62 | 2 | 2 | 2 |
| L8 | 104920 | 62/68 | 62 | 2 | 2 | 2 |
| L9 | 104921 | 62/68 | 62 | 2 | 2 | 2 |
| L10 | 104922 | 62/68 | 62 | 2 | 2 | 2 |
| L11 | 104923 | 62/68 | 62 | 2 | 2 | 2 |
| L12 | 104924 | 62/68 | 62 | 2 | 2 | 2 |
| L13 | 104925 | 62/68 | 62 | 2 | 2 | 2 |
| L14 | 104926 | 62/68 | 62 | 2 | 2 | 2 |
| L15 | 104927 | 62/68 | 62 | 2 | 2 | 2 |

Mean across layers: expert-ours=62.0, prefix=2.0, alloc-kv=2.0, free-pure=2.0.
