# Unified pool-size sweep — superseded

This standalone unified pool-size sweep was **merged into the memory-budget
sweep** (`memory_budget_sweep.md`).

## Why merged

A standalone unified-only sweep (varying pool size N alone) was decided to
be too narrow on its own — it shows unified at different pool sizes but
doesn't compare against static at the same memory budget. The memory-budget
sweep is the stronger comparison: at each per-layer memory budget M, it runs
both unified (pool=M) AND static at multiple cache/KV splits with C+K=M, all
on the same Test 2A two-phase workload.

The unified data points the original plan would have collected (pool ∈ {9,
16, 24, 32, 48, 64}) are subsumed by the memory-budget sweep's unified cells
at M ∈ {16, 24, 32, 48, 64}.

## Outcome

See `memory_budget_sweep.md` for the active plan and `results/budget*.json`
+ `logs/budget*.log` for the data.
