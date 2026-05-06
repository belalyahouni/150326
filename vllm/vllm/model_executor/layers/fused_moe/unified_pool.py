# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified per-layer page pool for expert weights and KV blocks.

Each layer's KV byte tensor is aliased as a pool buffer shared between
cached-prefix KV blocks and expert weight pages. Eviction looks at a
per-layer expert recency list and a shared prefix recency list and takes
whichever is colder. The unmodified Triton fused MoE kernel reads expert
weights straight out of the pool buffer through a strided view: row b
holds whatever block_id b currently has. topk_ids is rewritten per layer
from global expert ids to block_ids, and global_num_experts is set to
num_gpu_blocks for the kernel call. No staging tensor, no extra GPU
memory beyond the pool itself.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


# Trace gate is resolved once at module load. Per-call os.environ.get
# was measurable here because the gate fires tens of thousands of
# times per request. Levels: 0/unset = off, 1 = essential lines only
# (composition, evict, kv_claim, prefix add/remove), 2 = also dump
# step headers and the LRU snapshots (debug only, slow).
_TRACE_LEVEL = os.environ.get("VLLM_UNIFIED_POOL_TRACE", "0")
_TRACE_ENABLED = _TRACE_LEVEL in ("1", "2")
_TRACE_VERBOSE = _TRACE_LEVEL == "2"


def _trace_enabled() -> bool:
    return _TRACE_ENABLED


def _trace_verbose() -> bool:
    return _TRACE_VERBOSE


def move_experts_to_cpu(
    w13_weight: torch.nn.Parameter,
    w2_weight: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move expert weight tensors to CPU pinned memory and return them."""
    cpu_w13 = w13_weight.data
    cpu_w2 = w2_weight.data
    if cpu_w13.is_cuda:
        cpu_w13 = cpu_w13.cpu()
    if cpu_w2.is_cuda:
        cpu_w2 = cpu_w2.cpu()
    if not cpu_w13.is_pinned():
        cpu_w13 = cpu_w13.pin_memory()
    if not cpu_w2.is_pinned():
        cpu_w2 = cpu_w2.pin_memory()
    return cpu_w13, cpu_w2


class UnifiedPool:
    """Per-layer pool state.

    Tracks which expert sits at which block, the per-layer LRU of
    experts, and the set of blocks pinned for the current forward.
    The kernel reads weights directly out of pool_buffer through the
    strided views (pool_w13_view, pool_w2_view), so a stale eviction
    here is silent corruption rather than a recoverable miss.
    """

    _UNLOADED = -1  # sentinel for block_id_at

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        cpu_w13: torch.Tensor,
        cpu_w2: torch.Tensor,
        pool_buffer: torch.Tensor,
        page_size_bytes: int,
        w13_bytes: int,
        w2_bytes: int,
        device: torch.device,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.cpu_w13 = cpu_w13
        self.cpu_w2 = cpu_w2
        self.pool_buffer = pool_buffer
        self.page_size_bytes = page_size_bytes
        self.w13_bytes = w13_bytes
        self.w2_bytes = w2_bytes
        self.device = device
        assert w13_bytes + w2_bytes == page_size_bytes, (
            f"page_size_bytes ({page_size_bytes}) must equal "
            f"w13_bytes + w2_bytes ({w13_bytes + w2_bytes})"
        )
        # The element size has to divide page_size, w13_bytes, and
        # w2_bytes cleanly so the strided views land on whole elements.
        elem_size = cpu_w13.element_size()
        assert page_size_bytes % elem_size == 0, (
            f"page_size_bytes ({page_size_bytes}) must be a multiple of "
            f"element size ({elem_size})"
        )
        assert w13_bytes % elem_size == 0
        assert w2_bytes % elem_size == 0

        self._cpu_w13_bytes = cpu_w13.view(torch.int8).reshape(num_experts, -1)
        self._cpu_w2_bytes = cpu_w2.view(torch.int8).reshape(num_experts, -1)

        # Strided views over the pool buffer; the kernel reads from these.
        # Reinterpret the int8 buffer as the layer's weight dtype.
        pool_typed = pool_buffer.view(cpu_w13.dtype)
        page_size_elems = page_size_bytes // elem_size
        w13_offset_elems = 0
        w2_offset_elems = w13_bytes // elem_size

        # cpu_w13 has shape [num_experts, *w13_per_expert_shape]. The
        # view exposes the same per-expert layout but one row per pool
        # block, separated by page_size_elems instead of the natural
        # per-expert size. Inner strides stay row-major so DMAs deposit
        # into the same layout the view reads from.
        w13_per_expert_shape = cpu_w13.shape[1:]
        w13_per_expert_strides = cpu_w13[0].contiguous().stride()
        num_gpu_blocks = pool_typed.numel() // page_size_elems
        self.num_gpu_blocks = num_gpu_blocks

        self.pool_w13_view = torch.as_strided(
            pool_typed,
            size=(num_gpu_blocks, *w13_per_expert_shape),
            stride=(page_size_elems, *w13_per_expert_strides),
            storage_offset=w13_offset_elems,
        )
        w2_per_expert_shape = cpu_w2.shape[1:]
        w2_per_expert_strides = cpu_w2[0].contiguous().stride()
        self.pool_w2_view = torch.as_strided(
            pool_typed,
            size=(num_gpu_blocks, *w2_per_expert_shape),
            stride=(page_size_elems, *w2_per_expert_strides),
            storage_offset=w2_offset_elems,
        )
        # The fused MoE kernel asserts stride(-1) == 1 on its weight
        # tensors, so check up front.
        assert self.pool_w13_view.stride(-1) == 1
        assert self.pool_w2_view.stride(-1) == 1

        # GPU-side expert -> block_id table for the topk_ids remap.
        # int64 matches the kernel's int64 cast on expert ids and
        # avoids overflow in stride * offset.
        self.block_id_at = torch.full(
            (num_experts,),
            self._UNLOADED,
            dtype=torch.int64,
            device=device,
        )

        self.expert_at_block: dict[int, int] = {}
        self.block_at_expert: dict[int, int] = {}
        self.expert_lru: OrderedDict[int, int] = OrderedDict()
        self.pinned_blocks: set[int] = set()

        self.hits = 0
        self.misses = 0
        self.forward_count = 0

    def has_expert(self, expert_id: int) -> bool:
        return expert_id in self.block_at_expert

    def block_of_expert(self, expert_id: int) -> int:
        return self.block_at_expert[expert_id]

    def expert_of_block(self, block_id: int) -> int | None:
        return self.expert_at_block.get(block_id)

    def assign(self, block_id: int, expert_id: int, step: int) -> None:
        assert block_id not in self.expert_at_block, (
            f"L{self.layer_idx}: block {block_id} already mapped to expert "
            f"{self.expert_at_block[block_id]}"
        )
        assert expert_id not in self.block_at_expert, (
            f"L{self.layer_idx}: expert {expert_id} already mapped to block "
            f"{self.block_at_expert[expert_id]}"
        )
        self.expert_at_block[block_id] = expert_id
        self.block_at_expert[expert_id] = block_id
        self.expert_lru[expert_id] = step  # most-recently-used on insert
        # Mirror onto the GPU lookup for the forward-path remap.
        self.block_id_at[expert_id] = block_id

    def drop(self, block_id: int) -> int | None:
        expert_id = self.expert_at_block.pop(block_id, None)
        if expert_id is None:
            return None
        del self.block_at_expert[expert_id]
        self.expert_lru.pop(expert_id, None)
        # Invalidate the GPU lookup so block_id_at[topk_ids] can't
        # return a stale value. ensure_loaded is responsible for
        # making sure no expert id in the next forward resolves to
        # _UNLOADED.
        self.block_id_at[expert_id] = self._UNLOADED
        return expert_id

    def bump_expert(self, expert_id: int, step: int) -> None:
        """Mark expert as MRU and stamp it with the current step.

        Called for every expert touched this forward, hit or miss, so
        the eviction step can compare expert recency against prefix
        recency on equal terms.
        """
        if expert_id in self.expert_lru:
            self.expert_lru[expert_id] = step
            self.expert_lru.move_to_end(expert_id, last=True)


class UnifiedPoolManager:
    """Owns the per-layer pools and the cross-layer bookkeeping.

    block_holder maps block_id -> the set of layers that currently
    hold an expert there. It's used both for membership lookups and
    to broadcast invalidations when KV reclaims the block. Expert
    misses only touch the calling layer; KV allocations broadcast
    to every holder.

    prefix_lru is shared because attention touches the same block id
    at every layer in lockstep, so a single global prefix recency
    list is enough.
    """

    def __init__(self, block_pool, device: torch.device) -> None:
        from vllm.v1.core.block_pool import BlockPool

        assert isinstance(block_pool, BlockPool)
        self.block_pool: BlockPool = block_pool
        self.device = device
        self.layers: dict[int, UnifiedPool] = {}
        self.block_holder: dict[int, set[int]] = {}
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.step = 0  # incremented per forward; used as the MRU timestamp

        # block_id -> last-used step, oldest first. Updated by the
        # BlockPool prefix callbacks below.
        self.prefix_lru: OrderedDict[int, int] = OrderedDict()

        self.block_pool.register_on_allocation_callback(self._on_kv_allocation)
        self.block_pool.register_on_prefix_added_callback(self._on_prefix_added)
        self.block_pool.register_on_prefix_removed_callback(
            self._on_prefix_removed
        )
        # Override BlockPool's default popleft_n so KV allocations also
        # consult the unified LRUs, the same way expert misses do.
        self.block_pool.register_kv_victim_selector(
            self._select_kv_victim_blocks
        )

    def register_layer(self, layer: UnifiedPool) -> None:
        assert layer.layer_idx not in self.layers, (
            f"Layer {layer.layer_idx} already registered with the unified pool"
        )
        self.layers[layer.layer_idx] = layer

    # Mapping helpers.

    def _add_holder(self, layer_idx: int, block_id: int) -> None:
        self.block_holder.setdefault(block_id, set()).add(layer_idx)

    def _remove_holder(self, layer_idx: int, block_id: int) -> None:
        holders = self.block_holder.get(block_id)
        if holders is None:
            return
        holders.discard(layer_idx)
        if not holders:
            del self.block_holder[block_id]

    def _drop_layer_mapping(
        self, layer: UnifiedPool, block_id: int, cause: str,
        tier: str | None = None,
    ) -> None:
        """Drop only this layer's mapping for the block.

        Used on an expert miss where the layer is about to overwrite
        its own bytes. Other layers keep their mappings; nothing
        physical changes for them.
        """
        evicted = layer.drop(block_id)
        if evicted is None:
            return
        self._remove_holder(layer.layer_idx, block_id)
        if _trace_enabled():
            tier_str = f" tier={tier}" if tier else ""
            print(
                f"UNIFIED EVICT page={block_id} L{layer.layer_idx} "
                f"kind=expert E{evicted} cause={cause}{tier_str}",
                flush=True,
            )

    def _broadcast_drop_all_layers(self, block_id: int, cause: str) -> None:
        """Drop every layer's mapping at the block (KV is overwriting it).

        The kernel reads pool_buffer directly, so dropping a mapping
        on a pinned block would silently corrupt a live read. Assert
        on it. async_scheduling=False makes this trivial in practice
        but it's worth catching if that ever changes.
        """
        holders = self.block_holder.pop(block_id, None)
        if not holders:
            return
        for layer_idx in list(holders):
            layer = self.layers.get(layer_idx)
            if layer is None:
                continue
            assert block_id not in layer.pinned_blocks, (
                f"KV-allocation broadcast tried to drop a pinned block: "
                f"page={block_id} L{layer_idx} cause={cause}. The MoE "
                f"kernel may be reading those bytes — refusing to "
                f"corrupt. Check async_scheduling is disabled."
            )
            evicted = layer.drop(block_id)
            if evicted is not None and _trace_enabled():
                print(
                    f"UNIFIED EVICT page={block_id} L{layer_idx} "
                    f"kind=expert E{evicted} cause={cause} tier=kv-broadcast",
                    flush=True,
                )

    def _evict_prefix_globally(
        self, block_id: int, cause: str, tier: str | None = None
    ) -> None:
        """Clear the block's prefix hash everywhere.

        Once any layer overwrites the block's bytes the prefix is
        broken in every layer. Clearing the hash fires
        on_prefix_removed which drops it from prefix_lru.
        """
        block = self.block_pool.blocks[block_id]
        if block.block_hash is None:
            return
        self.block_pool.evict_prefix_hash(block_id)
        if _trace_enabled():
            tier_str = f" tier={tier}" if tier else ""
            print(
                f"UNIFIED EVICT page={block_id} L=all "
                f"kind=kv-prefix cause={cause}{tier_str}",
                flush=True,
            )

    # BlockPool callbacks.

    def _on_kv_allocation(self, block_ids: list[int]) -> None:
        """KV is about to overwrite these blocks; drop every layer's
        expert mapping for each one. Prefix hashes were already
        cleared upstream and fired _on_prefix_removed.
        """
        for block_id in block_ids:
            self._broadcast_drop_all_layers(block_id, cause="kv-alloc")

    def _on_prefix_added(self, block_id: int) -> None:
        """A cached-prefix block has been freed: bump it to MRU.

        Returning to the free queue with a hash counts as a use, so
        prefix recency stamps it with the current step.
        """
        self.prefix_lru[block_id] = self.step
        self.prefix_lru.move_to_end(block_id, last=True)
        if _trace_enabled():
            print(
                f"UNIFIED PREFIX_ADDED p{block_id} step={self.step} "
                f"size={len(self.prefix_lru)}",
                flush=True,
            )

    def _on_prefix_removed(self, block_id: int) -> None:
        """Block is no longer an evictable cached prefix; drop from prefix_lru."""
        removed = self.prefix_lru.pop(block_id, None)
        if _trace_enabled():
            was_present = "yes" if removed is not None else "no"
            print(
                f"UNIFIED PREFIX_REMOVED p{block_id} "
                f"was_present={was_present} size={len(self.prefix_lru)}",
                flush=True,
            )

    # Stage 2 warm-up.

    def warm_up(self, warm_count: int) -> None:
        """Pre-load warm_count experts per layer at startup.

        Each warmed expert reuses the same block_id across every
        layer (different physical bytes per layer because the pool
        buffer is per-layer), so total block_ids consumed equals
        warm_count regardless of layer count. Minimum pool size is
        warm_count + 1 (block 0 is the null block).

        The shared-id approach means a KV broadcast on a warmed
        block invalidates that expert in every layer at once.
        """
        if warm_count <= 0:
            return
        for layer in self.layers.values():
            assert warm_count <= layer.num_experts, (
                f"warm_count ({warm_count}) > num_experts "
                f"({layer.num_experts}) for L{layer.layer_idx}"
            )
        layers_list = list(self.layers.values())
        if not layers_list:
            return
        for expert_id in range(warm_count):
            # Pick one block_id, share it across all layers for this
            # expert. ``layers_list[0]`` is just the perspective used
            # to walk the free queue; result is whichever block sits
            # at the queue head.
            block, _tier = self._select_victim_block(
                layers_list[0], needed_set=set()
            )
            block_id = block.block_id
            self._evict_prefix_globally(
                block_id, cause=f"warmup-E{expert_id}"
            )
            for layer in layers_list:
                layer.assign(block_id, expert_id, step=self.step)
                self._add_holder(layer.layer_idx, block_id)
                self._dma_expert_into_block_sync(
                    layer, expert_id, block_id
                )
            self.block_pool.free_block_queue.append(block)
        # Warm-up DMAs run on transfer_stream. Wait for them and then
        # device-sync. If the first forward is all hits, ensure_loaded
        # won't call wait_stream itself, so any unflushed warm-up
        # would surface as stale reads.
        torch.cuda.current_stream(self.device).wait_stream(
            self.transfer_stream
        )
        torch.cuda.synchronize(self.device)
        for layer in layers_list:
            logger.info(
                "UnifiedPool L%d: warmed %d/%d experts",
                layer.layer_idx,
                warm_count,
                layer.num_experts,
            )

        # Post-warm-up sanity check. Each warmed (expert, block) pair
        # should round-trip: pool_w13_view[block_id] equals
        # cpu_w13[expert_id], same for w2. A failure here means the
        # strided view is misaligned or the DMA didn't land.
        for layer in layers_list:
            for expert_id, block_id in layer.block_at_expert.items():
                w13_view_row = layer.pool_w13_view[block_id]
                w2_view_row = layer.pool_w2_view[block_id]
                w13_truth = layer.cpu_w13[expert_id].to(layer.device)
                w2_truth = layer.cpu_w2[expert_id].to(layer.device)
                if not torch.equal(w13_view_row, w13_truth):
                    raise RuntimeError(
                        f"UnifiedPool L{layer.layer_idx}: pool_w13_view"
                        f"[{block_id}] != cpu_w13[{expert_id}] after warm-up. "
                        f"Strided view layout is wrong, or DMA didn't land."
                    )
                if not torch.equal(w2_view_row, w2_truth):
                    raise RuntimeError(
                        f"UnifiedPool L{layer.layer_idx}: pool_w2_view"
                        f"[{block_id}] != cpu_w2[{expert_id}] after warm-up."
                    )
        logger.info(
            "UnifiedPool warm-up sanity check passed: %d (expert, block) "
            "pairs verified across %d layers.",
            warm_count * len(layers_list),
            len(layers_list),
        )

    # Forward-path API.

    def ensure_loaded(
        self, layer: UnifiedPool, needed_expert_ids: list[int]
    ) -> None:
        """Make sure every needed expert is loaded at layer.

        Hits and miss-claimed blocks are pinned for the rest of this
        forward (released by release_pinned). DMAs end with a
        wait_stream barrier on the compute stream. Every needed
        expert is bumped to MRU regardless of hit/miss, so the LRU
        tracks use recency rather than claim recency. Trace
        snapshots are captured before any mutation.
        """
        hit_results: list[tuple[int, int]] = []  # (eid, block_id)
        miss_eids: list[int] = []
        needed_set = set(needed_expert_ids)
        for eid in needed_expert_ids:
            if layer.has_expert(eid):
                hit_results.append((eid, layer.block_of_expert(eid)))
            else:
                miss_eids.append(eid)

        # Trace before any state changes.
        if _trace_enabled():
            self._trace_pre_mutation(layer, needed_expert_ids)

        # Counters, pinning hit blocks, and MRU bumps for hits.
        layer.hits += len(hit_results)
        layer.misses += len(miss_eids)
        for eid, block_id in hit_results:
            layer.pinned_blocks.add(block_id)
            layer.bump_expert(eid, self.step)

        # Claim a block per miss.
        miss_assignments: list[tuple[int, int, str]] = []  # (eid, bid, tier)
        for eid in miss_eids:
            block, tier = self._select_victim_block(layer, needed_set)
            block_id = block.block_id
            cause = f"expert-L{layer.layer_idx}"
            if _TRACE_VERBOSE and tier.startswith("free"):
                # Free-tier claims don't generate an EVICT line, so emit
                # a CLAIM so the verbose trace still shows where each
                # miss landed.
                print(
                    f"UNIFIED CLAIM page={block_id} L{layer.layer_idx} "
                    f"E{eid} cause={cause} tier={tier}",
                    flush=True,
                )
            self._evict_prefix_globally(block_id, cause=cause, tier=tier)
            self._drop_layer_mapping(layer, block_id, cause=cause, tier=tier)
            # assign() stamps the expert with the current step as MRU.
            layer.assign(block_id, eid, step=self.step)
            self._add_holder(layer.layer_idx, block_id)
            layer.pinned_blocks.add(block_id)
            miss_assignments.append((eid, block_id, tier))
            self.block_pool.free_block_queue.append(block)

        if _TRACE_VERBOSE:
            hit_parts = [f"E{eid}@p{bid}" for eid, bid in hit_results]
            miss_parts = [
                f"E{eid}->p{bid}({tier})"
                for eid, bid, tier in miss_assignments
            ]
            print(
                f"UNIFIED RESULT L{layer.layer_idx} "
                f"hits=[{','.join(hit_parts)}] "
                f"misses=[{','.join(miss_parts)}]",
                flush=True,
            )
            print(f"--- end L{layer.layer_idx} ---", flush=True)

        # Issue all miss DMAs on the transfer stream, then sync.
        if miss_assignments:
            with torch.cuda.stream(self.transfer_stream):
                for eid, block_id, _tier in miss_assignments:
                    self._dma_expert_into_block_async(layer, eid, block_id)
            torch.cuda.current_stream(self.device).wait_stream(
                self.transfer_stream
            )

    def release_pinned(self, layer: UnifiedPool) -> None:
        layer.pinned_blocks.clear()
        layer.forward_count += 1

    def end_forward_step(self) -> None:
        self.step += 1

    # Per-layer victim selection.

    def _select_victim_block(
        self, layer: UnifiedPool, needed_set: set[int]
    ):
        """Pick the coldest evictable block for this layer.

        Returns (block, tier). Tier 1 walks the free queue and grabs
        the first block with no hash and no expert mapping in this
        layer. Tier 2 compares the oldest non-pinned entry in this
        layer's expert LRU against the head of the shared prefix
        LRU and takes the colder. Raises if neither tier has a
        candidate.
        """
        queue = self.block_pool.free_block_queue

        # Tier 1: free from this layer's perspective.
        cursor = queue.fake_free_list_head.next_free_block
        while cursor is not None and cursor is not queue.fake_free_list_tail:
            block_id = cursor.block_id
            nxt = cursor.next_free_block
            if block_id in layer.pinned_blocks:
                cursor = nxt
                continue
            our_eid = layer.expert_of_block(block_id)
            if our_eid is not None:
                cursor = nxt
                continue
            if cursor.block_hash is not None:
                cursor = nxt
                continue
            queue.remove(cursor)
            holders = self.block_holder.get(block_id)
            if holders:
                tier = "free-cross-layer-expert"
            else:
                tier = "free-pure"
            return cursor, tier

        # Tier 2: pick whichever LRU has the colder head.
        oldest_expert_eid: int | None = None
        oldest_expert_step: int | None = None
        for eid, step in layer.expert_lru.items():
            if eid in needed_set:
                continue
            block_id = layer.block_at_expert.get(eid)
            if block_id is None or block_id in layer.pinned_blocks:
                continue
            oldest_expert_eid = eid
            oldest_expert_step = step
            break

        oldest_prefix_id: int | None = None
        oldest_prefix_step: int | None = None
        for block_id, step in self.prefix_lru.items():
            if block_id in layer.pinned_blocks:
                continue
            oldest_prefix_id = block_id
            oldest_prefix_step = step
            break

        chosen_block_id: int | None = None
        chosen_tier: str | None = None
        if oldest_expert_eid is not None and oldest_prefix_id is not None:
            assert oldest_expert_step is not None
            assert oldest_prefix_step is not None
            if oldest_prefix_step <= oldest_expert_step:
                chosen_block_id = oldest_prefix_id
                chosen_tier = "prefix-global"
            else:
                chosen_block_id = layer.block_at_expert[oldest_expert_eid]
                chosen_tier = "expert-local"
        elif oldest_expert_eid is not None:
            chosen_block_id = layer.block_at_expert[oldest_expert_eid]
            chosen_tier = "expert-local"
        elif oldest_prefix_id is not None:
            chosen_block_id = oldest_prefix_id
            chosen_tier = "prefix-global"

        if chosen_block_id is not None:
            assert chosen_tier is not None
            block = self.block_pool.blocks[chosen_block_id]
            queue.remove(block)
            return block, chosen_tier

        raise RuntimeError(
            f"UnifiedPool L{layer.layer_idx}: pool exhausted while "
            "resolving expert miss. No pure-free block, no evictable "
            "expert, and no evictable prefix entry. Reduce "
            "--max-num-batched-tokens or increase --num-gpu-blocks-override."
        )

    # KV-side victim selection (no layer of origin).

    def _any_layer_pins(self, block_id: int) -> bool:
        """True if any registered layer currently pins ``block_id``."""
        for layer in self.layers.values():
            if block_id in layer.pinned_blocks:
                return True
        return False

    def _oldest_global_expert(
        self,
    ) -> tuple[int | None, int | None]:
        """Oldest non-pinned expert across all layers.

        Returns (block_id, step) or (None, None) if every expert is
        pinned by some layer.
        """
        best_step: int | None = None
        best_block: int | None = None
        for layer in self.layers.values():
            for eid, step in layer.expert_lru.items():
                block_id = layer.block_at_expert.get(eid)
                if block_id is None:
                    continue
                if self._any_layer_pins(block_id):
                    continue
                if best_step is None or step < best_step:
                    best_step = step
                    best_block = block_id
                # The LRU is oldest-first, so the first non-pinned
                # entry is the best this layer can offer.
                break
        return best_block, best_step

    def _select_kv_victim_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Pick num_blocks victims for KV allocation.

        Mirrors _select_victim_block but for the KV side. Tier 1
        takes blocks that are truly free (no hash, no expert
        mapping, not pinned). Tier 2 compares the oldest expert
        across every layer with the oldest prefix and takes the
        colder. Returned blocks are already off the free queue;
        BlockPool.get_new_blocks does the rest of the bookkeeping.
        """
        if num_blocks == 0:
            return []
        queue = self.block_pool.free_block_queue
        ret: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self._pick_one_kv_victim()
            ret.append(block)
        assert queue.num_free_blocks >= 0
        return ret

    def _pick_one_kv_victim(self) -> KVCacheBlock:
        """Pick one KV victim block and remove it from the free queue."""
        queue = self.block_pool.free_block_queue

        # Tier 1: truly free — no hash, no expert mapping, not pinned.
        cursor = queue.fake_free_list_head.next_free_block
        while cursor is not None and cursor is not queue.fake_free_list_tail:
            nxt = cursor.next_free_block
            block_id = cursor.block_id
            if cursor.block_hash is not None:
                cursor = nxt
                continue
            if self.block_holder.get(block_id):
                cursor = nxt
                continue
            if self._any_layer_pins(block_id):
                cursor = nxt
                continue
            queue.remove(cursor)
            if _trace_enabled():
                print(
                    f"UNIFIED KV_CLAIM page={block_id} tier=truly-free",
                    flush=True,
                )
            return cursor

        # Tier 2: pick whichever LRU has the colder head.
        oldest_expert_bid, oldest_expert_step = self._oldest_global_expert()

        oldest_prefix_bid: int | None = None
        oldest_prefix_step: int | None = None
        for block_id, step in self.prefix_lru.items():
            if self._any_layer_pins(block_id):
                continue
            oldest_prefix_bid = block_id
            oldest_prefix_step = step
            break

        chosen_block_id: int | None = None
        chosen_tier: str | None = None
        if oldest_expert_bid is not None and oldest_prefix_bid is not None:
            assert oldest_expert_step is not None
            assert oldest_prefix_step is not None
            if oldest_expert_step <= oldest_prefix_step:
                chosen_block_id = oldest_expert_bid
                chosen_tier = "kv-evicts-expert"
            else:
                chosen_block_id = oldest_prefix_bid
                chosen_tier = "kv-evicts-prefix"
        elif oldest_expert_bid is not None:
            chosen_block_id = oldest_expert_bid
            chosen_tier = "kv-evicts-expert"
        elif oldest_prefix_bid is not None:
            chosen_block_id = oldest_prefix_bid
            chosen_tier = "kv-evicts-prefix"

        if chosen_block_id is not None:
            assert chosen_tier is not None
            block = self.block_pool.blocks[chosen_block_id]
            queue.remove(block)
            if _trace_enabled():
                print(
                    f"UNIFIED KV_CLAIM page={chosen_block_id} "
                    f"tier={chosen_tier}",
                    flush=True,
                )
            return block

        raise RuntimeError(
            "UnifiedPool: pool exhausted resolving KV allocation. "
            "No truly-free block, no evictable expert, no evictable "
            "prefix entry. Reduce --max-num-batched-tokens or "
            "increase --num-gpu-blocks-override."
        )

    # DMA helpers.

    def _dma_expert_into_block_async(
        self, layer: UnifiedPool, expert_id: int, block_id: int
    ) -> None:
        """Async CPU->GPU copy of expert weights into the layer's pool page."""
        page_offset = block_id * layer.page_size_bytes
        w13_dst = layer.pool_buffer.narrow(
            0, page_offset, layer.w13_bytes
        )
        w2_dst = layer.pool_buffer.narrow(
            0, page_offset + layer.w13_bytes, layer.w2_bytes
        )
        w13_dst.copy_(layer._cpu_w13_bytes[expert_id], non_blocking=True)
        w2_dst.copy_(layer._cpu_w2_bytes[expert_id], non_blocking=True)

    def _dma_expert_into_block_sync(
        self, layer: UnifiedPool, expert_id: int, block_id: int
    ) -> None:
        with torch.cuda.stream(self.transfer_stream):
            self._dma_expert_into_block_async(layer, expert_id, block_id)

    # Trace helpers (active when VLLM_UNIFIED_POOL_TRACE>=1).

    def _trace_pre_mutation(
        self, layer: UnifiedPool, needed_expert_ids: list[int]
    ) -> None:
        """Dump pool composition (level >=1) and, at verbose level,
        the step header, both LRUs and the router's request. Runs
        before any mutation in ensure_loaded.

        UNIFIED CACHE includes step=self.step so the line is
        self-contained for downstream parsers.
        """
        capacity = len(self.block_pool.blocks)
        n_expert_ours = len(layer.expert_at_block)
        n_prefix = len(self.prefix_lru)
        n_alloc_kv = sum(
            1
            for b in self.block_pool.blocks
            if b.block_hash is not None and b.block_id not in self.prefix_lru
        )
        n_pinned = len(layer.pinned_blocks)
        n_held_other = sum(
            1
            for bid, holders in self.block_holder.items()
            if layer.layer_idx not in holders
        )
        n_free_pure = (
            capacity
            - n_expert_ours
            - n_prefix
            - n_alloc_kv
            - n_held_other
        )

        # Required at level 1 for the dissertation overlay figure.
        print(
            f"UNIFIED CACHE L{layer.layer_idx} step={self.step} "
            f"occ {n_expert_ours}/{capacity} ours "
            f"(expert-ours={n_expert_ours}, expert-other={n_held_other}, "
            f"prefix={n_prefix}, alloc-kv={n_alloc_kv}, "
            f"pinned={n_pinned}, free-pure={n_free_pure})",
            flush=True,
        )

        if not _TRACE_VERBOSE:
            return

        # Verbose-only per-step diagnostics.
        print(
            f"=== STEP {self.step} L{layer.layer_idx} "
            f"need={needed_expert_ids} ===",
            flush=True,
        )

        # OrderedDict has MRU at the end (move_to_end last=True); flip
        # so the printed order reads MRU first.
        expert_lru_str = ", ".join(
            f"E{eid}@p{layer.block_at_expert.get(eid, '?')}#step{step}"
            for eid, step in reversed(layer.expert_lru.items())
        )
        print(
            f"UNIFIED EXPERT_LRU L{layer.layer_idx} "
            f"MRU→LRU [{len(layer.expert_lru)}]: {expert_lru_str}",
            flush=True,
        )

        prefix_items = list(reversed(self.prefix_lru.items()))[:8]
        prefix_lru_str = ", ".join(
            f"p{bid}#step{step}" for bid, step in prefix_items
        )
        print(
            f"UNIFIED PREFIX_LRU MRU→LRU "
            f"[top 8 of {len(self.prefix_lru)}]: {prefix_lru_str}",
            flush=True,
        )

        print(
            f"UNIFIED REQUEST L{layer.layer_idx}: "
            + ",".join(f"E{e}" for e in needed_expert_ids),
            flush=True,
        )

    # Stats / introspection.

    def log_stats(self) -> None:
        num_kv_prefix = len(self.prefix_lru)
        for layer in self.layers.values():
            total = layer.hits + layer.misses
            hit_rate = layer.hits / total * 100 if total > 0 else 0.0
            num_expert_pages = len(layer.expert_at_block)
            logger.info(
                "UnifiedPool L%d: hits=%d misses=%d hit_rate=%.1f%% "
                "expert_pages=%d kv_prefix_pages=%d",
                layer.layer_idx,
                layer.hits,
                layer.misses,
                hit_rate,
                num_expert_pages,
                num_kv_prefix,
            )

    def shutdown_log(self) -> None:
        logger.info("UnifiedPool shutdown stats:")
        self.log_stats()
