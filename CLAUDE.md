## Dissertation Research Idea: Unified GPU VRAM Pool

**Core idea:** Instead of statically partitioning GPU VRAM between KV cache and expert weights, treat them as a single unified pool. Evict whichever is less valuable at any given moment.

**The two swap directions:**
- KV cache is hot (high prefix reuse) → evict idle experts, grow KV budget
- Experts are hot (frequent expert switching, high swap cost) → evict cold KV entries, grow expert cache

**Test scenarios (2x2 matrix):**

| | High KV reuse (repeated long prefills) | Low KV reuse (no prefix repetition) |
|---|---|---|
| Small KV budget / large expert budget | Unified pool should win (evict experts for KV) | Baseline is fine |
| Large KV budget / small expert budget | Baseline is fine | Unified pool should win (evict KV for experts) |

**Instructions:**
Always plan before implementing. Only implement when asked to. If anything goes not according to the plan during implmentation, do not push through, come back and plan again.