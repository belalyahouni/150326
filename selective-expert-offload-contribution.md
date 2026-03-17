# Selective Expert Loading for `--cpu-offload-params`

## Background

PR #34535 added `--cpu-offload-params` to vLLM, which lets users selectively offload
specific parameter name segments to CPU. The primary intended use case is MoE expert
weights, e.g.:

```bash
--cpu-offload-params w13_weight w2_weight
```

This frees GPU VRAM for KV cache without loading the full model to CPU.

### How it currently works

The implementation lives in `vllm/model_executor/offloader/uva.py` (`UVAOffloader`).
At model load time, for each matched parameter:

1. The weight is moved to CPU pinned memory.
2. **UVA path** (default): a CUDA device-side view of the CPU memory is created via
   `get_accelerator_view_from_cpu_tensor`. The GPU can then dereference this pointer
   directly, reading over PCIe on every access — no explicit copy, but every forward
   pass pays full PCIe bandwidth for every expert weight accessed.
3. **Non-UVA fallback**: `module.forward` is monkey-patched to call
   `module.state_dict().items()` → `.to(device, non_blocking=True)` on every forward
   call, copying **all offloaded params** to GPU before the kernel runs.

In both paths, **all offloaded expert weights are accessed/copied on every forward
pass**, regardless of which experts the router actually selects.

---

## The Problem

For MoE models, the router selects only `top_k` experts per token. In a layer with
256 experts where `top_k=2`, a batch might need at most ~20-30 distinct experts.
The other 230+ experts' weights are never touched by the kernel.

Under the current UVA approach:
- All 256 experts' weights cross PCIe on every forward pass (UVA reads), or
- All 256 are explicitly copied to GPU before the kernel (non-UVA fallback).

This is ~10x more PCIe traffic than necessary.

---

## The Opportunity

The MoE forward pass has a natural split point. In `FusedMoELayer.forward_native`
(`layer.py`), the router runs first:

```python
topk_weights, topk_ids = self.runner.router.select_experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

`topk_ids` tells us exactly which experts are needed *before* any expert weight
is accessed. We can use this to copy only the needed experts CPU → GPU, run the
kernel, then free the temporary GPU memory.

---

## Proposed Change

### Core idea

When `--cpu-offload-params` targets expert weight names (e.g. `w13_weight`,
`w2_weight`), the `FusedMoELayer` should detect that its expert weights are
CPU-resident and activate a **selective load path** instead of deferring to the
generic UVA module wrapper.

The selective load path, already structured in `_forward_with_expert_cache`, would:

1. Run the router → get `topk_ids`.
2. Compute `needed = topk_ids.unique()` (only the distinct experts used this batch).
3. Copy only `needed` experts CPU → GPU into a temporary buffer (async, on a
   dedicated CUDA stream).
4. Remap global expert IDs → local indices `[0 .. len(needed)-1]`.
5. Temporarily swap `w13_weight.data` and `w2_weight.data` to the small temporary
   buffers.
6. Call the kernel.
7. Restore original (CPU-resident) weight data pointers.
8. Free the temporary GPU buffers.

No LRU cache, no persistent GPU slots — just allocate-copy-use-free per forward pass.

### How to detect "expert weights are CPU-offloaded"

The cleanest detection point is in `FusedMoELayer.load_weights` or
`_init_expert_cache`. After the `UVAOffloader` has run, check whether
`self.w13_weight.data.device == cpu` (or carries the `_vllm_is_uva_offloaded` flag).
If so, activate the selective path.

Alternatively, pass a flag from `OffloadConfig` — if `uva.cpu_offload_params`
intersects `{"w13_weight", "w2_weight"}`, set `use_selective_expert_load = True`
on the layer.

### Sketch of the new forward

```python
def _forward_with_cpu_offload_experts(self, hidden_states, router_logits):
    # Step 1: route
    topk_weights, topk_ids = self.runner.router.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

    # Step 2: which experts do we actually need?
    needed = topk_ids.unique().tolist()

    # Step 3: copy only needed experts CPU → GPU (temp buffers)
    device = hidden_states.device
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        temp_w13 = self.w13_weight.data[needed].to(device, non_blocking=True)
        temp_w2  = self.w2_weight.data[needed].to(device, non_blocking=True)
    torch.cuda.current_stream(device).wait_stream(stream)

    # Step 4: remap IDs
    id_map = {gid: lid for lid, gid in enumerate(needed)}
    remapped_ids = topk_ids.clone()
    for gid, lid in id_map.items():
        remapped_ids[topk_ids == gid] = lid

    # Step 5-6-7: swap, call kernel, restore
    orig_w13 = self.w13_weight.data
    orig_w2  = self.w2_weight.data
    orig_n   = self.global_num_experts
    try:
        self.w13_weight.data = temp_w13
        self.w2_weight.data  = temp_w2
        self.global_num_experts = len(needed)
        result = self.quant_method.apply(
            layer=self, x=hidden_states,
            topk_weights=topk_weights, topk_ids=remapped_ids,
            shared_experts_input=None,
        )
    finally:
        self.w13_weight.data = orig_w13
        self.w2_weight.data  = orig_w2
        self.global_num_experts = orig_n

    # Step 8: temp buffers go out of scope, CUDA frees them
    return result
```

Note: `self.w13_weight.data[needed]` works because the UVA offloader places the
weights in CPU pinned memory and CUDA can index-gather from it via the UVA pointer,
or we do it explicitly as `.to(device)` on the gathered slice.

---

## What needs to change

| File | Change |
|---|---|
| `vllm/model_executor/layers/fused_moe/layer.py` | Add `_forward_with_cpu_offload_experts`; detect CPU-resident expert weights and route to it from `forward_native` |
| `vllm/model_executor/offloader/uva.py` | When wrapping a `FusedMoELayer` and `cpu_offload_params` targets expert weights, skip the generic forward-wrapper (the selective path handles it) |
| `vllm/config/offload.py` | Optionally add a convenience flag or document that `cpu_offload_params={"w13_weight","w2_weight"}` activates selective loading |
| Tests | Add test that verifies only `len(needed)` experts are transferred per forward, not all `num_experts` |

---

## Tradeoffs vs existing approaches

| | Current UVA | Proposed selective load | ExpertCache (LRU) |
|---|---|---|---|
| GPU memory for expert weights | 0 (zero-copy PCIe read) | `len(needed) * expert_size` per batch | `cache_size * expert_size` persistent |
| PCIe traffic per batch | All experts (read via PCIe) | Only needed experts | Only misses |
| Persistent GPU allocation | No | No | Yes |
| LRU / warm cache benefit | No | No | Yes |
| Implementation complexity | Low | Medium | High |
| Best for | Memory-constrained, slow inference OK | Single-batch, no repeated experts | High-throughput, expert locality |

The selective load path is a strict improvement over the current UVA path when
expert weights are the target: same GPU memory footprint, but PCIe traffic scales
with `top_k` instead of `num_experts`.

---

## Open questions before implementing

1. **UVA index gather**: Does `cpu_pinned_tensor[indices].to(device)` correctly
   issue a gather-then-copy, or does it first gather on CPU (fine) or read all rows
   via PCIe (bad)? Need to verify with a microbenchmark. If gather-on-CPU is
   expensive, pre-building a CPU-side `[expert_id → flat offset]` index is an
   alternative.

2. **Quantized weights**: `w13_weight` and `w2_weight` may carry scales/zeros
   alongside them. The remap and copy logic needs to handle all associated
   quantization parameters, not just the weight tensor. Check what
   `quant_method.apply` actually reads from `self`.

3. **Shared experts**: Some MoE architectures (e.g. DeepSeek) have shared experts
   that are always active. These should stay on GPU and not go through the selective
   load path.

4. **Stream reuse vs allocation**: Allocating a fresh `torch.cuda.Stream` per
   forward is wasteful; the stream should be created once at layer init and reused.

5. **Duplicate-work check**: Before opening a PR, verify with
   `gh pr list --repo vllm-project/vllm --state open --search "selective expert"`
   that no one else is working on this.
