# Attention Backend Layout Mismatch

## Symptom

With `--expert-unified-pool` on OLMoE/L40s, `/v1/chat/completions` returned token soup (`"M_P]E_C_C_T_E_T_R_E_T_E_T_E_D_M_M_M_T_C_H_M_0.0.0.0..."`). MoE staging was correct; the corruption was in attention reads.

## Root cause

The pool aliases each layer's KV byte tensor as a flat buffer and treats it as `num_gpu_blocks` contiguous pages of `page_size_bytes`, where pool page `b` = bytes `[b·page_size, (b+1)·page_size]`. That math assumes K and V for one logical block sit next to each other in memory.

vLLM's two backends differ:

| Backend | KV cache shape | K/V layout | Matches our aliasing? |
|---|---|---|---|
| `TRITON_ATTN` | `(num_blocks, 2, block_size, num_kv_heads, head_size)` | per block: K then V, contiguous | yes |
| `FLASH_ATTN` | `(2, num_blocks, block_size, num_kv_heads, head_size)` | all K's first, then all V's | no |

For OLMoE on L40s the platform default is `FLASH_ATTN`. Numerically: `K_per_block = 1536·16·128·2 = 6 MiB`, `page_size = 12 MiB`, so pool page `b` physically overlaps K bytes for scheduler blocks `2b` and `2b+1`. Expert DMAs into pool page `b` clobber live K cache for unrelated scheduler blocks; the pool's `_on_kv_allocation` invalidation only cleans its own page-`b` mapping, leaving the scheduler's view stale. Attention then reads garbage.

## Fix

Force the K/V-contiguous backend at launch:

```
--attention-backend TRITON_ATTN
```

(Not `VLLM_ATTENTION_BACKEND=...` — that env var is unrecognized in this build and silently does nothing.)

## Followup

Stage 1 should assert the resolved attention backend has the per-block K+V layout (i.e. `get_kv_cache_shape` starts with `num_blocks`, not `2`) and fail loudly otherwise. Same spirit as the existing TP=1 / prefix-caching / eager checks.
