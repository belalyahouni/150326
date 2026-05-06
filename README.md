# 150326

This is the code repository for my BSc dissertation at the University of Leeds (COMP3931, 2025/26). The project is about letting expert weights and the KV cache share the same per-layer GPU memory in vLLM, instead of splitting the memory between them at startup and never changing the split.

The code is built on top of vLLM v0.17.1 and was tested with OLMoE-1B-7B-0924-Instruct on a single NVIDIA L40.

## Branches

The project was built in phases, and each phase lives on its own branch so they can be checked out and run independently.

- `base-vllm`: the unmodified vLLM v0.17.1 snapshot that the project was forked from. Used as a clean baseline.
- `phase-1-expert-offload`: a static expert cache. Each MoE layer keeps a fixed number of experts on the GPU and pages the rest in from CPU pinned memory. This is the offloading foundation that everything else builds on.
- `phase-2-unified-pool-mvp`: the first end-to-end unified pool. Expert pages and KV blocks share a single per-layer buffer with a dual LRU. The kernel still reads from a per-layer staging tensor, so the pool runs end to end but the kernel does not depend on it.
- `phase-3-unified-pool-no-staging`: the staging tensors are removed and the kernel reads expert weights directly from the pool buffer through a strided view. This is the version evaluated in the report.

A fourth phase was in the original plan, which would have shrunk the page size and grouped pages into per-expert super-blocks. It was fully designed but not implemented within the project window.

## What is where

Phase 3 has most of the material because the evaluation was done on that branch.

- `vllm/`: the modified vLLM source. The project's code changes are mostly under `vllm/vllm/model_executor/layers/fused_moe/` and `vllm/vllm/v1/`.
- `scripts/`: prompt generators, the trace summariser, and the shell scripts that drive each test.
- `prompts/`: the JSONL prompt files the generators produce.
- `logs/`: bench, server and trace logs from each test (`test1A`, `test1B`, `test2A`, `test2B`, `static_sweep`, `budget_sweep`, `many_prefixes`).
- `results/`: parsed JSON results from those runs.

The earlier branches contain only the code for that phase plus a small amount of supporting material.

## Running it

The exact commands and flags for each test are in the matching `scripts/run_*.sh` file. The relevant flags are:

- `--expert-offload`: turn on expert offloading.
- `--expert-cache-size N`: number of expert slots per layer.
- `--expert-unified-pool`: use the unified pool (only on the phase-2 and phase-3 branches).
- `VLLM_UNIFIED_POOL_TRACE=1`: emit the per-step pool trace.

## Notes

This is a research prototype written for an academic project. It is not meant as a general purpose inference framework.
