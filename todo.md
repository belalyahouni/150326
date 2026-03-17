# Expert Cache TODO

- **Bypass GPU allocation in `create_weights` for expert offload mode.** Currently `create_weights` calls `torch.empty([num_experts, ...])` which allocates on GPU (due to the `with target_device:` context in model init), then our code immediately moves it to CPU pinned RAM. This means one layer's full expert tensor must fit on GPU momentarily, which fails for large unquantized models (e.g. 256 experts in bf16 = ~21 GB per layer). **Plan:**
  1. Move `_expert_cache_size` calculation to before `create_weights` (safe — only depends on `vllm_config.offload_config`, not on anything `create_weights` produces).
  2. Wrap `create_weights` conditionally: if `_expert_cache_size > 0`, run inside `with torch.device("cpu")` so tensors allocate directly on CPU; otherwise run normally.
  3. Simplify the post-`create_weights` block: tensors are already on CPU, so only `.pin_memory()` is needed (drop the `.cpu()` call). Keep the `hasattr` guards.
  - `_maybe_init_expert_cache`, `_forward_with_expert_cache`, and everything else in `__init__` are untouched.
