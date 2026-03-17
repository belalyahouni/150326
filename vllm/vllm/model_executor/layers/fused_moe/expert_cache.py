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
        self.free_slots: list[int] = []

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
        torch.cuda.synchronize(self.device)
        logger.info(
            "ExpertCache: warmed %d/%d experts on %s "
            "(w13: %s, w2: %s)",
            self.cache_size,
            self.num_experts,
            self.device,
            list(self.cache_w13.shape),
            list(self.cache_w2.shape),
        )

    def ensure(self, needed_ids: list[int]) -> None:
        """Ensure all ``needed_ids`` are present in the GPU cache.

        Cache hits are free. Misses evict the LRU expert and DMA the
        needed expert from CPU pinned RAM into the freed slot.

        After this call returns, the compute stream is safe to read from
        the cache slots for all ``needed_ids``.
        """
        misses: list[tuple[int, int]] = []  # (expert_id, slot)

        for eid in needed_ids:
            if eid in self.expert_to_slot:
                self.hits += 1
                continue

            # Cache miss — need a slot
            self.misses += 1

            if self.free_slots:
                slot = self.free_slots.pop()
            else:
                # Evict LRU expert
                lru_expert, _ = self.lru_order.popitem(last=False)
                slot = self.expert_to_slot.pop(lru_expert)

            self.expert_to_slot[eid] = slot
            self.lru_order[eid] = None  # Add as MRU
            misses.append((eid, slot))

        if misses:
            # DMA copies on the transfer stream
            with torch.cuda.stream(self.transfer_stream):
                for eid, slot in misses:
                    self.cache_w13[slot].copy_(self.cpu_w13[eid], non_blocking=True)
                    self.cache_w2[slot].copy_(self.cpu_w2[eid], non_blocking=True)

            # Make the default (compute) stream wait for transfers
            torch.cuda.current_stream(self.device).wait_stream(
                self.transfer_stream
            )

    def get_cached_weights(
        self, needed_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather cached expert weights into contiguous tensors.

        Args:
            needed_ids: Expert IDs that are guaranteed to be in the cache
                (call ``ensure`` first).

        Returns:
            ``(temp_w13, temp_w2)`` each of shape ``[len(needed_ids), ...]``
            indexed as local experts ``0..len(needed_ids)-1``.
        """
        slots = [self.expert_to_slot[eid] for eid in needed_ids]
        slot_indices = torch.tensor(slots, dtype=torch.long, device=self.device)
        temp_w13 = self.cache_w13[slot_indices]
        temp_w2 = self.cache_w2[slot_indices]
        return temp_w13, temp_w2

    def record_use(self, expert_ids: list[int]) -> None:
        """Move ``expert_ids`` to the MRU end of the LRU list."""
        for eid in expert_ids:
            # move_to_end(last=True) makes it the most-recently-used
            self.lru_order.move_to_end(eid, last=True)

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
