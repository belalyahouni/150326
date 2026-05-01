# Unified Per-Layer Page Pool — Phase 2 Plan

> Companion to `unified-pool-mvp-plan.md` and `unified-pool-mvp-implementation.md`. Phase 1 (the MVP on the `unified-pool-mvp` branch) keeps a full-width staging tensor on GPU so the unmodified MoE kernel can read all experts as if they were resident. **Phase 2 removes the staging tensor entirely.** The kernel reads expert weights directly from the unified pool buffer via a strided view, with `topk_ids` remapped per-layer from global expert ids to the block_ids that currently hold them.
>
> The MVP plan §3 originally framed Phase 2 as "modify the Triton `fused_moe_kernel` to take a page-table argument." **That turns out to be unnecessary.** The kernel already accepts `B.stride(0)` as a runtime argument — it does not assume expert rows pack tightly. A strided view over the pool buffer with `stride(0) = page_size_bytes` gives the kernel exactly the indirection a hand-modified kernel would have provided. See §9 for why this supersedes the original Phase 2 framing.

---

## 1. Goal

Eliminate the per-layer `staging_w13` / `staging_w2` tensors. The Triton `fused_moe_kernel` reads expert weights directly from the unified pool buffer.

End state:
- **Zero** GPU memory footprint beyond the pool itself (the ~12.9 GiB of staging overhead on Phase 1's OLMoE setup is freed).
- **Zero** per-forward GPU→GPU gather.
- **Zero** modification to the Triton `fused_moe_kernel`.
- The MVP plan §3 fairness invariant becomes stricter: baseline and unified-pool runs compete on identical *total* GPU memory, not just identical *dynamic* memory. The "static vs dynamic split" comparison no longer carries a staging-overhead asterisk.

---

## 2. Core Mechanic — Strided View + Per-Layer Remap

### Why the kernel doesn't need modification

The MoE kernel reads weight bytes via (`fused_moe.py:461-465`):

```python
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
b_ptrs = b_ptr + off_experts * stride_be \
       + offs_k * stride_bk + offs_bn * stride_bn
```

`stride_be`, `stride_bk`, `stride_bn` are **runtime arguments**, populated at launch from `B.stride(0)`, `B.stride(2)`, `B.stride(1)` (`fused_moe.py:811-813`). The kernel does *not* assume `stride_be == w13_bytes / element_size` — it just multiplies whatever stride it was given.

A strided view that says "consecutive expert rows are `page_size_bytes` apart, and within each row the layout is the natural `[2I, H]` BF16 matrix" is a valid `B`. The kernel walks `block_id * page_size_bytes` to find an expert's row, then reads tile-by-tile inside it. The fact that the next `w2_bytes` worth of memory after each w13 row contain that same expert's w2 (irrelevant to the w13 GEMM) is invisible to the kernel.

### Per-layer pool views (constructed once, in Stage 2)

```python
pool_bf16 = pool_buffer.view(torch.bfloat16)         # [num_pages × page_elems]
pool_w13_view = torch.as_strided(
    pool_bf16,
    size=(num_gpu_blocks, *w13_per_expert_shape),    # e.g. [143, 2*I, H]
    stride=(page_size_bytes // 2, *w13_natural_strides),
)
pool_w2_view = torch.as_strided(
    pool_bf16,
    size=(num_gpu_blocks, *w2_per_expert_shape),     # e.g. [143, H, I]
    stride=(page_size_bytes // 2, *w2_natural_strides),
    storage_offset=w13_bytes // 2,                   # skip past w13 in each page
)
```

Both views are **metadata-only** — no copy, no extra allocation. They share storage with the pool buffer (which itself shares storage with the layer's KV byte tensor).

### Per-layer `block_id_at` lookup table

A small GPU tensor, shape `[num_experts]`, dtype `int64`. `block_id_at[e]` is the block_id where expert `e` currently lives in this layer's pool, or a sentinel (e.g. `-1`) if not loaded.

Maintained by `UnifiedPool.assign` and `UnifiedPool.drop` — exactly the same call sites that already update `expert_at_block` / `block_at_expert`. One scalar write per assign, one scalar invalidation per drop. Memory cost across all layers: `num_experts * 8 * num_moe_layers` ≈ a few KiB.

After `manager.ensure_loaded(pool, needed_expert_ids)` returns, every `e ∈ needed_expert_ids` has a valid `block_id_at[e]`.

---

## 3. Forward Path

`_forward_with_unified_pool` becomes:

```python
topk_weights, topk_ids = self.runner.router.select_experts(...)
needed_expert_ids = topk_ids.unique().tolist()
manager.ensure_loaded(pool, needed_expert_ids)

# Remap global expert ids → per-layer block_ids
remapped_ids = pool.block_id_at[topk_ids]

orig_w13 = self.w13_weight.data
orig_w2  = self.w2_weight.data
orig_n_experts = self.global_num_experts
orig_em = self._expert_map
try:
    self.w13_weight.data = pool.pool_w13_view       # strided [num_gpu_blocks, ...]
    self.w2_weight.data  = pool.pool_w2_view
    self.global_num_experts = pool.num_gpu_blocks
    self._expert_map = None
    result = self.quant_method.apply(
        layer=self,
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=remapped_ids,
        shared_experts_input=None,
    )
finally:
    self.w13_weight.data = orig_w13
    self.w2_weight.data  = orig_w2
    self.global_num_experts = orig_n_experts
    self._expert_map = orig_em

manager.release_pinned(pool)
manager.end_forward_step()
```

This is structurally identical to `_forward_with_expert_cache` on `main`, with two substitutions:
- "Cache slot" → "pool block_id."
- "Contiguous `[cache_size, ...]` GPU tensor" → "strided `[num_gpu_blocks, ...]` view over the pool buffer."

---

## 4. What Changes vs Phase 1

| | Phase 1 (current MVP)                        | Phase 2 (this plan)                                 |
|---|----------------------------------------------|------------------------------------------------------|
| Per-layer staging tensors                  | Allocated, full-width `[num_experts, ...]` | **Removed**                                          |
| GEMM input                                 | Static staging                              | Strided view over pool                               |
| Per-forward GPU→GPU gather                 | None                                        | None                                                 |
| Per-forward `topk_ids` remap               | None                                        | Yes — `block_id_at[topk_ids]`                        |
| Triton kernel modification                 | None                                        | None                                                 |
| Pinning during forward                     | Precautionary                               | **Load-bearing** (kernel reads pool directly)        |
| GPU memory beyond pool                     | ~12.9 GiB staging (OLMoE)                   | ~few KiB `block_id_at` per layer                     |

---

## 5. Implementation Envelope

The agent owns concrete data structures, function signatures, and stream/sync code. This section names the touchpoints.

### `vllm/model_executor/layers/fused_moe/unified_pool.py`

`UnifiedPool` constructor adds three fields:
- `pool_w13_view` — strided `[num_gpu_blocks, *w13_shape]` view over `pool_buffer`. Built once at construction time (the pool buffer doesn't move).
- `pool_w2_view` — strided `[num_gpu_blocks, *w2_shape]` view, with `storage_offset = w13_bytes // elem_size`.
- `block_id_at` — `int64` GPU tensor `[num_experts]`, initialized to a sentinel value.

`UnifiedPool.assign(block_id, expert_id, step)` and `UnifiedPool.drop(block_id)` each get one extra line to keep `block_id_at` in sync. Sentinel-clear on drop, scalar set on assign. **`manager.warm_up`** writes `block_id_at` for the warmed experts.

The `staging_w13` / `staging_w2` fields and constructor arguments are **deleted**.

### `vllm/model_executor/layers/fused_moe/layer.py`

`unified_pool_stage1` stops allocating the staging tensors. The metadata it returns drops `staging_w13_nbytes` / `staging_w2_nbytes`. The `_unified_pool_staging_*` module attributes are removed.

`_forward_with_unified_pool` is rewritten per §3.

### `vllm/v1/worker/gpu_model_runner.py`

Stage 1's "staging overhead" log line is removed (there is no staging overhead). The fairness-invariant explanation in startup logs is updated.

Stage 2 (`setup_unified_pool`) unchanged in structure: still constructs `UnifiedPool` per layer, narrows the KV byte tensor as before. The two strided views and the `block_id_at` tensor are built inside the `UnifiedPool` constructor.

### Required flags (unchanged)

`--expert-offload --expert-unified-pool --enable-prefix-caching --enforce-eager --max-num-batched-tokens 1`, TP=1, PP=1.

The `--max-num-batched-tokens 1` justification (cap unique experts per forward at `top_k` to avoid pool exhaustion) is unchanged.

### Kernel-side considerations (unchanged but worth verifying)

- `naive_block_assignment` (`fused_moe.py:1801`) test is `num_tokens * top_k * 4 <= global_num_experts`. With `--max-num-batched-tokens 1`, `top_k=8`, and `global_num_experts = num_gpu_blocks` (e.g. 143 for OLMoE), this is `32 <= 143` → naive mode stays on. Simpler kernel path.
- `moe_align_block_size` allocates buffers sized by the `num_experts` argument. Phase 2 passes `num_gpu_blocks` here (e.g. 143 vs 64) — buffer growth is small, still tiny in absolute terms.

---

## 6. What Stays The Same

- Pool eviction tiers (Tier 1 free / Tier 2 cold-tail mixed LRU / Tier 3 fail) unchanged.
- Mixed LRU (per-layer expert recency + shared prefix recency) unchanged.
- BlockPool callbacks (`_on_kv_allocation`, `_on_prefix_added`, `_on_prefix_removed`) unchanged.
- KV-allocation cross-layer broadcast unchanged.
- Cached-prefix global eviction unchanged.
- Stage 1 / Stage 2 init sequencing unchanged except for the staging-tensor allocation.
- Trace mode format unchanged.
- The `VLLM_UNIFIED_POOL_TRACE=1` architectural acceptance grep checks (MVP plan §6.6) unchanged.

---

## 7. Verification

Work through these in order; do not declare done until every step passes.

1. **Imports compile.** Source-tree and venv-installed paths.
2. **Server boots.** Same command as Phase 1's verification step 2. Startup log no longer reports staging overhead. `# GPU blocks` should be **larger** than the Phase 1 number for the same memory budget — staging-equivalent bytes are returned to the unified pool.
3. **Correctness smoke test.** One short completion. Compare tokens to:
   - The same model with `--expert-offload` only (no unified pool). Match modulo BF16 nondeterminism.
   - The same model on the Phase 1 unified pool. Match modulo BF16 nondeterminism.
4. **KV-hot benchmark.** Same as MVP plan §6 step 4. Should pass with at least the same margin (the freed staging bytes give the dynamic pool more room to grow KV-heavy).
5. **Expert-hot benchmark.** Same as MVP plan §6 step 5.
6. **Trace mode.** Same architectural acceptance grep checks as MVP plan §6.6 — no changes to the trace format. Should pass identically.
7. **Pinning regression test (new for Phase 2).** Construct a stress workload that interleaves expert misses with KV writes targeting the same `block_id` within close temporal proximity. Verify the kernel never reads from a block whose bytes are mid-DMA or mid-overwrite. The pinning contract is now load-bearing — a bug here is *silent corruption* in Phase 2, vs. a *visible miss* in Phase 1.
8. **Memory accounting.** Print and audit GPU memory at startup: should show pool size exactly equal to baseline KV size (the unified-pool path adds zero extra allocation beyond `block_id_at`).

---

## 8. Caveats / Limitations

- **BF16-only.** Quantized models (FP8, INT8 with scales) need per-expert scale tensors that the kernel indexes via `off_experts * stride_bse` (`fused_moe.py:467-496`). The strided-view trick generalizes — same `as_strided` over a per-layer scale buffer would work — but the unified pool's page layout currently has no place for scales. Out of scope for Phase 2; consistent with MVP plan §4 out-of-scope list.
- **Pinning becomes load-bearing.** In Phase 1, even if pinning had a bug the GEMM would still read from staging (correct values). In Phase 2, a pinning bug means the GEMM reads stale or mid-DMA pool bytes, producing **silent** corruption. Verification step 7 is mandatory; the pinning contract should also be unit-tested.
- **`block_id_at` is per-layer.** Different layers can hold the same expert at different block_ids. This is intentional (it's the whole point of per-layer pool buffers) but means each layer maintains its own lookup tensor. Total bookkeeping memory: a few KiB.
- **`global_num_experts = num_gpu_blocks` during the kernel call.** Affects buffers `moe_align_block_size` allocates. For OLMoE (143 vs 64) this roughly doubles its bookkeeping, still tiny.
- **Phase 2 does not change the dissertation comparison.** KV-hot / expert-hot scenarios still demonstrate dynamic rebalance against a workload-tuned static baseline. What changes: the unified pool's footprint becomes exactly equal to baseline's (no staging asterisk), making the comparison cleaner to defend.

---

## 9. Why This Replaces the Original Phase 2

The MVP plan §3 framed Phase 2 as: "modify the Triton `fused_moe_kernel` to take a page-table argument." The implicit assumption was that the kernel needed an extra `expert_offsets[]` argument and corresponding `b_ptrs = b_ptr + offsets[off_experts]` arithmetic.

The kernel already does this implicitly. `b_ptrs = b_ptr + off_experts * stride_be` *is* page-table arithmetic — `stride_be` is the page stride, and we control it at launch time. Passing a strided view with `stride(0) = page_size_bytes` makes the kernel multiply `block_id * page_size_bytes` to find each expert's row, which is exactly the page-table computation a Phase-2-as-originally-conceived kernel would have done.

This plan delivers the full Phase 2 end state (no staging, no gather, no extra memory) with a strict subset of the changes the original framing envisioned. The Triton kernel is untouched.

---

## 10. Out of Scope

Same as MVP plan §4, with no additions:

- Quantization (FP8, INT8, AWQ, GPTQ).
- Hybrid models, MLA, Mamba, sliding-window attention.
- Multi-process, expert parallelism, runtime-adaptive cache sizing.
- Sub-page KV granularity.
- Modifying `fused_moe_kernel` (now permanently out of scope — not a Phase 3 followup).

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `as_strided` storage_offset interaction with `view`'d byte buffer breaks aliasing | Low | High (silent wrong reads) | Verify with a unit test that `pool_w13_view[block_id]` retrieves the same bytes as `pool_buffer.narrow(0, block_id*page_size_bytes, w13_bytes).view(bf16).reshape(w13_shape)`. |
| `naive_block_assignment` heuristic flips to non-naive mode if the inequality changes (e.g. `top_k` increases) and `sorted_token_ids`'s structure assumes `expert_ids` indexes into a "true" `[E, N, K]` tensor | Low | Medium | The non-naive path also uses `off_experts * stride_be` — same trick works. Verify in the verification step 6 trace under both paths. |
| Pinning bug surfaces only under load → silent corruption | Medium | High | Verification step 7. Add an assert in `_forward_with_unified_pool` that every block in `remapped_ids` is currently in `pool.pinned_blocks` (paranoid mode, env-gated). |
| `block_id_at` GPU updates create implicit synchronization cost | Low | Low | Single-element writes are cheap; if profiling shows overhead, batch updates inside `ensure_loaded`. |
