# Unified Per-Layer Page Pool — Phase 2 Implementation

> Companion to `unified-pool-phase2-plan.md`. Describes what's actually built on the `unified-pool-phase-2` branch (forked from `unified-pool-mvp`). Phase 2 removes the staging tensor and has the unmodified Triton MoE kernel read expert weights directly from the unified pool buffer via a strided view.

---

## 1. Diff Summary vs Phase 1 (`unified-pool-mvp`)

```
vllm/vllm/model_executor/layers/fused_moe/unified_pool.py   ~30 LOC swapped: staging fields → strided views + block_id_at; defensive pinning assert
vllm/vllm/model_executor/layers/fused_moe/layer.py          ~30 LOC swapped: staging allocation deleted; forward rewrites for view + remap
vllm/vllm/v1/worker/gpu_model_runner.py                     ~20 LOC swapped: staging args removed; async_scheduling=False asserted; staging-overhead log replaced
```

No changes to: BlockPool, engine/core.py, gpu_worker.py, config/offload.py, arg_utils.py.

---

## 2. UnifiedPool Constructor (Phase 2 shape)

```python
UnifiedPool(
    layer_idx, num_experts,
    cpu_w13, cpu_w2,           # CPU-pinned source-of-truth (DMA src on miss)
    pool_buffer,               # int8 view of layer's KV byte tensor, narrowed
    page_size_bytes, w13_bytes, w2_bytes,
    device,                    # NEW: required for block_id_at allocation
)
```

Phase-1 fields removed: `staging_w13`, `staging_w2`. Construction-time additions:

- `self.pool_w13_view` — `torch.as_strided` over `pool_buffer.view(weight_dtype)` with shape `(num_gpu_blocks, *w13_per_expert_shape)` and stride `(page_size_bytes // elem_size, *natural_w13_strides)`. Storage offset = 0.
- `self.pool_w2_view` — same idea, shape `(num_gpu_blocks, *w2_per_expert_shape)`, storage_offset = `w13_bytes // elem_size` so each row points at a page's w2 region.
- `self.block_id_at` — `torch.full((num_experts,), -1, dtype=torch.int64, device=device)`. Mirror of `block_at_expert`, lives on GPU for `topk_ids` indexing in the forward path.
- `self.num_gpu_blocks` — derived from `pool_buffer.numel() // page_size_bytes`. Cached on the pool for the swap step.

Invariants asserted in `__init__`:
- `w13_bytes + w2_bytes == page_size_bytes`
- All three are multiples of `cpu_w13.element_size()`
- `pool_w13_view.stride(-1) == 1` and `pool_w2_view.stride(-1) == 1` (the only assertion the Triton kernel makes about `B`).

`UnifiedPool.assign(block_id, expert_id, step)` is unchanged at call sites; internally adds one line writing `self.block_id_at[expert_id] = block_id` (a host→device scalar write on the default stream, stream-ordered with the kernel that will read it).

`UnifiedPool.drop(block_id)` symmetrically writes the `_UNLOADED = -1` sentinel into `block_id_at[evicted_expert]`.

---

## 3. Forward Path

`_forward_with_unified_pool` (layer.py, ~line 1626):

```python
topk_weights, topk_ids = self.runner.router.select_experts(...)
needed_expert_ids = topk_ids.unique().tolist()
manager.ensure_loaded(pool, needed_expert_ids)

# Remap global expert ids → per-layer block_ids
remapped_ids = pool.block_id_at[topk_ids]

orig_w13 = self.w13_weight.data
orig_w2  = self.w2_weight.data
orig_num_experts = self.global_num_experts
orig_expert_map  = self._expert_map
try:
    self.w13_weight.data    = pool.pool_w13_view
    self.w2_weight.data     = pool.pool_w2_view
    self.global_num_experts = pool.num_gpu_blocks
    self._expert_map        = None
    result = self.quant_method.apply(
        layer=self, x=hidden_states,
        topk_weights=topk_weights, topk_ids=remapped_ids,
        shared_experts_input=None,
    )
finally:
    self.w13_weight.data    = orig_w13
    self.w2_weight.data     = orig_w2
    self.global_num_experts = orig_num_experts
    self._expert_map        = orig_expert_map

manager.release_pinned(pool)
manager.end_forward_step()
```

This mirrors `_forward_with_expert_cache` (on `main`) one-to-one, with two substitutions: cache slot → pool block_id, dense `[cache_size, ...]` tensor → strided `[num_gpu_blocks, ...]` view.

---

## 4. Stage 1 / Stage 2 Wiring

### Stage 1 (`gpu_model_runner._unified_pool_stage1`)

Same structure as Phase 1, with two adjustments:

- **New flag check**: rejects `async_scheduling=True`. The default in vLLM is auto-on; users must pass `--no-async-scheduling`. Rationale: the Phase 2 pinning contract requires scheduler KV-allocation and worker forward to be strictly serialized.
- **New backend check**: asserts the resolved attention backend's `get_kv_cache_shape` starts with `num_blocks`, not `2` — i.e., per-block K+V are contiguous in memory. Pass `--attention-backend TRITON_ATTN` at launch; the platform default `FLASH_ATTN` (`(2, num_blocks, ...)` layout) silently corrupts attention reads because pool pages no longer line up with scheduler blocks. Fail-loud if violated.
- **Staging-overhead log line removed**. Stage 1 message becomes a single line noting "No staging tensor — kernel reads pool buffer directly."

`FusedMoE.unified_pool_stage1` no longer allocates `staging_w13` / `staging_w2`. It still pins experts to CPU and stashes `_unified_pool_cpu_w13` / `_unified_pool_cpu_w2` for Stage 2.

### Stage 2 (`gpu_model_runner.setup_unified_pool`)

Same structure as Phase 1, with the `UnifiedPool(...)` call dropping `staging_w13` / `staging_w2` keyword args and adding `device=self.device`.

The strided views and `block_id_at` tensor are constructed inside `UnifiedPool.__init__`, so Stage 2 doesn't see them directly.

---

## 5. Cross-Layer Coordination

Same three paths as Phase 1: KV-allocation broadcast, cached-prefix global eviction, expert-driven invalidation in L only.

**Phase 2 hardening**: `_broadcast_drop_all_layers` now asserts `block_id not in layer.pinned_blocks` before dropping. With `async_scheduling=False` and the synchronous engine loop, scheduler `get_new_blocks` only fires between forwards (never during one), so no pinned block can ever be the target of a KV broadcast. The assertion is defense-in-depth — if it fires, something is violating the synchrony contract.

---

## 6. What Phase 2 Frees vs Phase 1

For OLMoE-1B-7B-0924-Instruct:

- Per-layer staging tensor: `64 experts × (2*1024*2048 + 2048*1024) × 2 bytes ≈ 0.75 GiB`.
- 16 MoE layers → ~12 GiB total staging removed.
- `block_id_at`: `64 × 8 bytes × 16 layers ≈ 8 KiB`.

Net memory reclaimed: ~12 GiB returned to the unified pool, increasing `num_gpu_blocks`.

---

## 7. Verification

Tracking against Phase 2 plan §7:

| Step | Status |
|---|---|
| 1. Imports compile (source + venv) | ✓ done |
| 2. Server boots with Phase 2 flags | _(in progress)_ |
| 3. Correctness smoke test | pending |
| 4. KV-hot benchmark | pending |
| 5. Expert-hot benchmark | pending |
| 6. Trace mode unchanged | pending |
| 7. Pinning regression test | pending |
| 8. Memory accounting | pending |

The `--no-async-scheduling` and `--attention-backend TRITON_ATTN` flags must both be passed at startup. Without `--no-async-scheduling`, Stage 1 fails fast. Without the Triton attention backend, the per-block K+V-contiguous layout assertion (Stage 1) fails fast — the platform's default FlashAttn layout silently corrupts attention reads via the unified pool's page aliasing.

---

## 8. Known Followups

- **`_unified_pool_enabled` field** on `FusedMoE` is still set-but-never-read (carried over from Phase 1). Safe to delete.
- **Trace-format**: unchanged from Phase 1. The `EVICT ... cause=kv-alloc tier=kv-broadcast` lines should be rare-or-zero under the synchronous engine loop, since scheduler KV-allocation only fires between forwards.
- **Quantized models**: still out of scope. Quantized weights have `B_scale` tensors indexed via `off_experts * stride_bse`; the strided-view trick generalizes but the pool layout currently has no place for scales.
- **Pinning unit test**: a paranoid-mode env flag could assert at the start of every kernel call that every block_id in `remapped_ids` is in `layer.pinned_blocks`. Useful for catching new code paths that bypass `ensure_loaded`.
