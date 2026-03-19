# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident LRU cache for MoE expert weights.

All experts permanently live on CPU pinned RAM. The GPU has a small cache
(``cache_size`` slots) where replicas of hot experts live. On a cache miss
we DMA-copy from CPU pinned → GPU, evicting the LRU expert's replica.
"""

from collections import OrderedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ExpertCache:
    """LRU cache that keeps hot expert weight replicas on GPU.

    Args:
        cache_size: Number of expert slots to keep on GPU.
        cpu_w13: CPU-pinned tensor ``[num_experts, ...]`` (gate+up weights).
        cpu_w2: CPU-pinned tensor ``[num_experts, ...]`` (down weights).
        device: Target CUDA device for the GPU cache buffers.
    """

    def __init__(
        self,
        cache_size: int,
        cpu_w13: torch.Tensor,
        cpu_w2: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.cache_size = cache_size
        self.num_experts = cpu_w13.shape[0]
        self.cpu_w13 = cpu_w13
        self.cpu_w2 = cpu_w2
        self.device = device

        assert cache_size <= self.num_experts, (
            f"cache_size ({cache_size}) must be <= num_experts "
            f"({self.num_experts})"
        )

        # Allocate GPU cache buffers
        self.cache_w13 = torch.empty(
            (cache_size, *cpu_w13.shape[1:]),
            dtype=cpu_w13.dtype,
            device=device,
        )
        self.cache_w2 = torch.empty(
            (cache_size, *cpu_w2.shape[1:]),
            dtype=cpu_w2.dtype,
            device=device,
        )

        # LRU tracking
        self.expert_to_slot: dict[int, int] = {}
        # OrderedDict: oldest-first iteration order
        self.lru_order: OrderedDict[int, None] = OrderedDict()

        # Dedicated DMA stream for async CPU→GPU copies
        self.transfer_stream = torch.cuda.Stream(device=device)

        # Stats
        self.hits = 0
        self.misses = 0

        # Pre-populate cache with experts 0..cache_size-1
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Copy first ``cache_size`` experts from CPU→GPU."""
        for slot in range(self.cache_size):
            expert_id = slot
            self.cache_w13[slot].copy_(self.cpu_w13[expert_id])
            self.cache_w2[slot].copy_(self.cpu_w2[expert_id])
            self.expert_to_slot[expert_id] = slot
            self.lru_order[expert_id] = None

        logger.info(
            "ExpertCache: warmed %d/%d experts on %s "
            "(w13: %s, w2: %s)",
            self.cache_size,
            self.num_experts,
            self.device,
            list(self.cache_w13.shape),
            list(self.cache_w2.shape),
        )

    def ensure_experts_loaded(self, needed_expert_ids: list[int]) -> None:
        """Ensure all ``needed_expert_ids`` are present in the GPU cache.

        Uses a two-pass approach to avoid evicting an expert that is itself
        needed in the same batch:
          Pass 1 — classify each expert as a hit or miss.
          Pass 2 — for each miss, evict the coldest cached expert that is
                   not in the needed set (walking lru_order LRU→MRU so the
                   hottest non-needed survivors are preserved), then assign
                   that slot to the incoming expert.
          Pass 3 — batch all CPU→GPU DMA copies on the transfer stream and
                   make the compute stream wait.

        After this call returns, the compute stream is safe to read from
        the cache slots for all ``needed_expert_ids``.

        # TODO: consider coalescing per-expert cudaMemcpyAsync calls into
        # fewer larger transfers if expert tensors are small and miss counts
        # are high.
        """
        needed_set = set(needed_expert_ids)

        # --- Pass 1: classify hits and misses ---
        missing_expert_ids: list[int] = []
        hit_ids: list[int] = []
        for expert_id in needed_expert_ids:
            if expert_id in self.expert_to_slot:
                self.hits += 1
                hit_ids.append(expert_id)
            else:
                self.misses += 1
                missing_expert_ids.append(expert_id)

        print(f"ExpertCache: needed={needed_expert_ids} hits={hit_ids} misses={missing_expert_ids}", flush=True)

        if not missing_expert_ids:
            return

        # --- Pass 2: evict coldest non-needed experts, assign slots ---
        # lru_order is oldest-first, so iterating from the left gives us
        # the coldest candidates first — hottest non-needed experts survive.
        eviction_candidates = iter([
            expert_id
            for expert_id in self.lru_order
            if expert_id not in needed_set
        ])
        experts_and_slots_to_copy: list[tuple[int, int]] = []  # (expert_id, slot)
        for expert_id in missing_expert_ids:
            evicted_expert_id = next(eviction_candidates)
            slot = self.expert_to_slot.pop(evicted_expert_id)
            del self.lru_order[evicted_expert_id]
            print(f"ExpertCache: evict expert {evicted_expert_id} from slot {slot}, load expert {expert_id}", flush=True)

            self.expert_to_slot[expert_id] = slot
            self.lru_order[expert_id] = None  # Add as MRU
            experts_and_slots_to_copy.append((expert_id, slot))

        # --- Pass 3: batch DMA CPU→GPU ---
        with torch.cuda.stream(self.transfer_stream):
            for expert_id, slot in experts_and_slots_to_copy:
                self.cache_w13[slot].copy_(self.cpu_w13[expert_id], non_blocking=True)
                self.cache_w2[slot].copy_(self.cpu_w2[expert_id], non_blocking=True)

        # Make the default (compute) stream wait for transfers
        torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)

    def mark_recently_used(self, expert_ids: list[int]) -> None:
        """Move ``expert_ids`` to the MRU end of the LRU list."""
        for expert_id in expert_ids:
            # move_to_end(last=True) makes it the most-recently-used
            self.lru_order.move_to_end(expert_id, last=True)

    def log_stats(self) -> None:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0.0
        logger.info(
            "ExpertCache stats: hits=%d misses=%d total=%d hit_rate=%.1f%%",
            self.hits,
            self.misses,
            total,
            hit_rate,
        )
