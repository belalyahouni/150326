# Trace summary: `manyprefix7_unified_init40_seed1_g1_trace.log`

- File size: 87.2 MB
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

- `UNIFIED EVICT` total: **443**
- `UNIFIED KV_CLAIM` total: **39**
- `UNIFIED PREFIX_ADDED` total: 28
- `UNIFIED PREFIX_REMOVED` total: 16
- `UNIFIED CACHE` snapshots: 615520

## Direction of pressure (the headline)

- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): 400
- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): 25
- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): 0
- **KV claim of truly-free page** (no eviction): 14

## EVICT breakdown

By kind:
- `expert`: 430 (97.1%)
- `kv-prefix`: 13 (2.9%)

By cause:
- `kv-alloc`: 400 (90.3%)
- `expert-L1`: 38 (8.6%)
- `expert-L3`: 4 (0.9%)
- `expert-L2`: 1 (0.2%)

By tier:
- `kv-broadcast`: 400 (90.3%)
- `expert-local`: 30 (6.8%)
- `prefix-global`: 13 (2.9%)

## KV_CLAIM breakdown by tier

- `kv-evicts-expert`: 25 (64.1%)
- `truly-free`: 14 (35.9%)

## Expert diversity per layer

Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. Reporting **distinct experts ever evicted** per layer (lower bound on the experts that were live at some point and got displaced).

| Layer | distinct evicted |
|---|---|
| L0 | 25 |
| L1 | 40 |
| L2 | 26 |
| L3 | 29 |
| L4 | 25 |
| L5 | 25 |
| L6 | 25 |
| L7 | 25 |
| L8 | 25 |
| L9 | 25 |
| L10 | 25 |
| L11 | 25 |
| L12 | 25 |
| L13 | 25 |
| L14 | 25 |
| L15 | 25 |

## Expert-vs-KV swing

Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, 0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot.

- **Most expert-heavy moment**: ratio=1.000 at L0 step=0 (expert-ours=40, alloc-kv=0)
- **Most KV-heavy moment**: ratio=0.933 at L7 step=615447 (expert-ours=28, alloc-kv=2)

Per-layer swing:

| Layer | min ratio (most KV) | max ratio (most expert) | swing |
|---|---|---|---|
| L0 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L1 | 0.951 (eo=39, kv=2) | 1.000 (eo=40, kv=0) | 0.049 |
| L2 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L3 | 0.955 (eo=42, kv=2) | 1.000 (eo=40, kv=0) | 0.045 |
| L4 | 0.943 (eo=33, kv=2) | 1.000 (eo=40, kv=0) | 0.057 |
| L5 | 0.944 (eo=34, kv=2) | 1.000 (eo=40, kv=0) | 0.056 |
| L6 | 0.939 (eo=31, kv=2) | 1.000 (eo=40, kv=0) | 0.061 |
| L7 | 0.933 (eo=28, kv=2) | 1.000 (eo=40, kv=0) | 0.067 |
| L8 | 0.933 (eo=28, kv=2) | 1.000 (eo=40, kv=0) | 0.067 |
| L9 | 0.950 (eo=38, kv=2) | 1.000 (eo=40, kv=0) | 0.050 |
| L10 | 0.946 (eo=35, kv=2) | 1.000 (eo=40, kv=0) | 0.054 |
| L11 | 0.946 (eo=35, kv=2) | 1.000 (eo=40, kv=0) | 0.054 |
| L12 | 0.943 (eo=33, kv=2) | 1.000 (eo=40, kv=0) | 0.057 |
| L13 | 0.946 (eo=35, kv=2) | 1.000 (eo=40, kv=0) | 0.054 |
| L14 | 0.947 (eo=36, kv=2) | 1.000 (eo=40, kv=0) | 0.053 |
| L15 | 0.944 (eo=34, kv=2) | 1.000 (eo=40, kv=0) | 0.056 |

## Final pool composition (last `UNIFIED CACHE` per layer)

| Layer | step | occ | expert-ours | prefix | alloc-kv | free-pure |
|---|---|---|---|---|---|---|
| L0 | 615504 | 48/64 | 48 | 10 | 2 | 2 |
| L1 | 615505 | 50/64 | 50 | 10 | 2 | 2 |
| L2 | 615506 | 48/64 | 48 | 10 | 2 | 2 |
| L3 | 615507 | 46/64 | 46 | 10 | 2 | 2 |
| L4 | 615508 | 33/64 | 33 | 10 | 2 | 2 |
| L5 | 615509 | 34/64 | 34 | 10 | 2 | 2 |
| L6 | 615510 | 31/64 | 31 | 10 | 2 | 2 |
| L7 | 615511 | 28/64 | 28 | 10 | 2 | 2 |
| L8 | 615512 | 28/64 | 28 | 10 | 2 | 2 |
| L9 | 615513 | 39/64 | 39 | 10 | 2 | 2 |
| L10 | 615514 | 36/64 | 36 | 10 | 2 | 2 |
| L11 | 615515 | 35/64 | 35 | 10 | 2 | 2 |
| L12 | 615516 | 34/64 | 34 | 10 | 2 | 2 |
| L13 | 615517 | 35/64 | 35 | 10 | 2 | 2 |
| L14 | 615518 | 36/64 | 36 | 10 | 2 | 2 |
| L15 | 615519 | 34/64 | 34 | 10 | 2 | 2 |

Mean across layers: expert-ours=37.2, prefix=10.0, alloc-kv=2.0, free-pure=2.0.
