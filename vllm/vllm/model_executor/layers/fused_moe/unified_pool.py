# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified KV + expert page pool ("Design D" / Phase 1).

The scheduler's `BlockPool` already owns a true LRU (`FreeKVCacheBlockQueue`:
HEAD=LRU, TAIL=MRU). This module makes *expert pages* look like KV blocks
in that queue. When an MoE layer uses an expert, the worker calls
`BlockPool.touch([block])` (pins ref_cnt, removes from free queue); after
the forward finishes it calls `BlockPool.free_blocks([...])` (ref_cnt → 0,
appended at TAIL). Cold expert pages drift to HEAD and get popped when the
scheduler requests a new KV block. An on-alloc callback clears the worker's
expert mapping before the page is reused for KV data.

Design D keeps a static per-layer `staging_w13` / `staging_w2` tensor
(shape ``[num_experts, ...]``) filled once at startup. The unmodified Triton
`fused_moe_kernel` reads from staging every step — no per-forward gather,
no `topk_ids` remap. The pool still does real CPU→GPU DMA on expert misses
so miss latency is measured honestly, even though the kernel never reads
from pool pages. The transfer-stream `wait_stream` in `ensure_experts_loaded`
is the simulation-fidelity barrier that makes the kernel's start time equal
`max(DMA completion)`, matching Phase 2 behavior.
"""

from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PerLayerPool:
    """Per-layer view of the unified pool.

    ``pool_buffer`` aliases the attention layer's raw int8 KV tensor; we
    do not own it and must not free it. Expert pages and KV blocks share
    this buffer at byte offsets `page * page_bytes`.
    """

    layer_idx: int
    pool_buffer: torch.Tensor  # int8, shape [n_pages * page_bytes]
    n_pages: int
    page_bytes: int
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    expert_dtype: torch.dtype
    w13_bytes: int
    w2_bytes: int
    cpu_w13: torch.Tensor  # [num_experts, *w13_shape], pinned
    cpu_w2: torch.Tensor
    staging_w13: torch.Tensor  # [num_experts, *w13_shape], static, filled once
    staging_w2: torch.Tensor
    expert_to_page: dict[int, int] = field(default_factory=dict)
    page_to_expert: dict[int, int] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def slot_w13_view(self, page: int) -> torch.Tensor:
        off = page * self.page_bytes
        return (
            self.pool_buffer[off : off + self.w13_bytes]
            .view(self.expert_dtype)
            .view(self.w13_shape)
        )

    def slot_w2_view(self, page: int) -> torch.Tensor:
        off = page * self.page_bytes + self.w13_bytes
        return (
            self.pool_buffer[off : off + self.w2_bytes]
            .view(self.expert_dtype)
            .view(self.w2_shape)
        )

    def invalidate_page(self, page: int) -> None:
        eid = self.page_to_expert.pop(page, None)
        if eid is not None:
            self.expert_to_page.pop(eid, None)


class UnifiedPagePoolManager:
    """Coordinates the per-layer pools with the scheduler's `BlockPool`."""

    def __init__(
        self,
        pools: dict[int, PerLayerPool],
        block_pool,  # vllm.v1.core.block_pool.BlockPool
        device: torch.device,
    ) -> None:
        self.pools = pools
        self.block_pool = block_pool
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device=device)
        # block_id -> set of layer_idx that hold an expert at this page.
        self.expert_layers_at_page: dict[int, set[int]] = {}
        # layer_idx -> pool pages pinned this forward, to free in release.
        self._pending_release: dict[int, list[int]] = {}
        self._step_counter = 0

    # -- Callbacks -----------------------------------------------------------

    def on_block_allocated(self, block_id: int) -> None:
        """Fired by `BlockPool.get_new_blocks` for each block returned.

        Any layer that currently holds an expert at this page must forget
        that mapping: the caller (scheduler for KV, worker for an incoming
        expert DMA) is about to overwrite the page.
        """
        layers = self.expert_layers_at_page.pop(block_id, None)
        if not layers:
            return
        for li in layers:
            pool = self.pools.get(li)
            if pool is not None:
                pool.invalidate_page(block_id)

    # -- One-shot init -------------------------------------------------------

    def warm_pool_all_layers(self, n_per_layer: int) -> None:
        """Populate the pool with the first ``n_per_layer`` experts of each
        layer. This provides realistic starting residency for the LRU; the
        kernel does not read from pool pages (it reads static staging).
        """
        for li, layer in self.pools.items():
            n = min(n_per_layer, layer.cpu_w13.shape[0])
            for eid in range(n):
                (block,) = self.block_pool.get_new_blocks(1)
                p = block.block_id
                layer.slot_w13_view(p).copy_(layer.cpu_w13[eid])
                layer.slot_w2_view(p).copy_(layer.cpu_w2[eid])
                layer.expert_to_page[eid] = p
                layer.page_to_expert[p] = eid
                self.expert_layers_at_page.setdefault(p, set()).add(li)
                self.block_pool.free_blocks([block])

    # -- Per-forward hot path ------------------------------------------------

    def ensure_experts_loaded(
        self, layer_idx: int, needed_ids: list[int]
    ) -> None:
        """Make sure every expert in ``needed_ids`` is resident in this
        layer's pool and pin the corresponding pages for the duration of
        the forward. No return value — the kernel reads from static staging.
        """
        layer = self.pools[layer_idx]
        hit_blocks = []
        miss_ids: list[int] = []
        for eid in needed_ids:
            page = layer.expert_to_page.get(eid)
            if page is None:
                miss_ids.append(eid)
                layer.misses += 1
            else:
                hit_blocks.append(self.block_pool.blocks[page])
                layer.hits += 1

        # Pin hits first so they can't be evicted by our own miss allocations.
        if hit_blocks:
            self.block_pool.touch(hit_blocks)

        pinned_pages: list[int] = [b.block_id for b in hit_blocks]

        if miss_ids:
            new_blocks = self.block_pool.get_new_blocks(len(miss_ids))
            # DMA on the transfer stream; wait_stream before any compute-stream
            # consumer can run.
            with torch.cuda.stream(self.transfer_stream):
                for eid, block in zip(miss_ids, new_blocks):
                    p = block.block_id
                    layer.slot_w13_view(p).copy_(
                        layer.cpu_w13[eid], non_blocking=True
                    )
                    layer.slot_w2_view(p).copy_(
                        layer.cpu_w2[eid], non_blocking=True
                    )
                    layer.expert_to_page[eid] = p
                    layer.page_to_expert[p] = eid
                    self.expert_layers_at_page.setdefault(p, set()).add(
                        layer_idx
                    )
                    pinned_pages.append(p)
            # Simulation-fidelity barrier: Phase 2's kernel would read from
            # pool pages and thus genuinely need these DMAs to complete.
            # Design D's unmodified kernel reads staging, so without this
            # wait the GEMM would launch artificially early. wait_stream
            # forces start_time = max(DMA completion), matching Phase 2.
            torch.cuda.current_stream(self.device).wait_stream(
                self.transfer_stream
            )

        self._pending_release[layer_idx] = pinned_pages

    def release_after_forward(self, layer_idx: int) -> None:
        """Release the pages pinned in `ensure_experts_loaded`.

        Safe to call immediately after the GEMM — the kernel reads from
        static staging, not pool pages, so there is no kernel-vs-free race
        and no `cuda.synchronize()` is needed.
        """
        pages = self._pending_release.pop(layer_idx, None)
        if not pages:
            return
        blocks = [self.block_pool.blocks[p] for p in pages]
        self.block_pool.free_blocks(blocks)

    # -- Instrumentation -----------------------------------------------------

    def log_stats(self) -> None:
        total_hits = sum(p.hits for p in self.pools.values())
        total_misses = sum(p.misses for p in self.pools.values())
        denom = total_hits + total_misses
        hit_rate = (total_hits / denom * 100.0) if denom else 0.0

        # HEAD-of-queue composition: classify the first few free blocks as
        # KV (has a block_hash) or expert (holds a mapping in some layer).
        q = self.block_pool.free_block_queue
        sample = 32
        head = []
        node = q.fake_free_list_head.next_free_block
        while node is not None and node is not q.fake_free_list_tail and len(head) < sample:
            head.append(node)
            node = node.next_free_block
        kv_count = 0
        expert_count = 0
        empty_count = 0
        for blk in head:
            if blk.block_hash is not None:
                kv_count += 1
            elif blk.block_id in self.expert_layers_at_page:
                expert_count += 1
            else:
                empty_count += 1

        logger.info(
            "UnifiedPool stats: hits=%d misses=%d hit_rate=%.1f%% | "
            "HEAD(%d): kv=%d expert=%d empty=%d",
            total_hits,
            total_misses,
            hit_rate,
            len(head),
            kv_count,
            expert_count,
            empty_count,
        )

    def maybe_log_stats(self, every: int = 100) -> None:
        self._step_counter += 1
        if self._step_counter % every == 0:
            self.log_stats()
