# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified per-layer page pool for MoE expert weights and KV blocks.

Phase 2: each MoE layer's KV byte tensor is aliased as a "pool buffer"
shared between cached-prefix KV blocks and expert weight pages. A per-layer
expert-recency list and one shared prefix-recency list together form the
"per-layer mixed LRU" — both bumped on every hit, both consulted at
eviction time. The unmodified Triton fused_moe kernel reads expert weights
**directly** from the pool buffer via a strided view: row ``b`` of the
view is whatever block_id ``b`` currently holds. ``topk_ids`` is remapped
per-layer from global expert ids to block_ids (via ``block_id_at``), and
``global_num_experts`` is set to ``num_gpu_blocks`` for the kernel call.
No staging tensor, no GPU->GPU gather, no extra GPU memory beyond the pool.
"""

import os
from collections import OrderedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _trace_enabled() -> bool:
    return os.environ.get("VLLM_UNIFIED_POOL_TRACE", "0") == "1"


def move_experts_to_cpu(
    w13_weight: torch.nn.Parameter,
    w2_weight: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move expert weight tensors to CPU pinned memory.

    Returns the new pinned (cpu_w13, cpu_w2) tensors. Used by both the
    plain expert-cache path and the unified pool setup.
    """
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
    """Per-layer pool state (Phase 2).

    Holds:
      - ``cpu_w13``, ``cpu_w2``: CPU-pinned source-of-truth tensors used
        as DMA source on a miss.
      - ``pool_buffer``: flat int8 view of the layer's KV byte tensor.
        Real CPU->GPU DMAs land here on miss; **the MoE kernel reads
        directly from this buffer via the strided views below**.
      - ``pool_w13_view``: strided ``[num_gpu_blocks, 2*I, H]`` view onto
        ``pool_buffer`` with ``stride(0) = page_size_bytes / element_size``.
        Row ``b`` of this view is whatever block_id ``b`` currently holds
        (an expert's w13 if loaded; arbitrary bytes if KV/free/empty).
        Handed to the unmodified Triton kernel each forward.
      - ``pool_w2_view``: same idea, with ``storage_offset = w13_bytes /
        element_size`` so each row points at the w2 bytes within a page.
      - ``expert_at_block: dict[block_id -> expert_id]``: which expert
        ``L`` currently has loaded at this block.
      - ``block_at_expert: dict[expert_id -> block_id]``: inverse.
      - ``block_id_at: torch.Tensor[num_experts] int64`` (GPU): mirror of
        ``block_at_expert`` for GPU-side ``topk_ids`` remap. Sentinel ``-1``
        means "not loaded." Updated on every ``assign``/``drop``. Indexed
        by ``topk_ids`` in the forward path: ``remapped = block_id_at[topk_ids]``.
      - ``expert_lru: OrderedDict[expert_id, step]``: per-layer expert
        recency, oldest first. ``move_to_end`` on every hit. **This is
        the structure consulted at eviction time** — it is not a side
        record.
      - ``pinned_blocks: set[block_id]``: blocks pinned during the
        current forward (kept off the eviction-candidate list until
        release). In Phase 2 the kernel reads ``pool_buffer`` directly,
        so this set is **load-bearing** — a stale eviction here is
        silent corruption, not a recoverable miss.
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
        # Element size has to divide page_size / w13 / w2 cleanly so the
        # strided views land on whole elements.
        elem_size = cpu_w13.element_size()
        assert page_size_bytes % elem_size == 0, (
            f"page_size_bytes ({page_size_bytes}) must be a multiple of "
            f"element size ({elem_size})"
        )
        assert w13_bytes % elem_size == 0
        assert w2_bytes % elem_size == 0

        self._cpu_w13_bytes = cpu_w13.view(torch.int8).reshape(num_experts, -1)
        self._cpu_w2_bytes = cpu_w2.view(torch.int8).reshape(num_experts, -1)

        # ---- Strided views over the pool buffer (Phase 2 kernel input) ----
        # Reinterpret the int8 pool_buffer as the layer's weight dtype.
        pool_typed = pool_buffer.view(cpu_w13.dtype)
        page_size_elems = page_size_bytes // elem_size
        w13_offset_elems = 0
        w2_offset_elems = w13_bytes // elem_size

        # cpu_w13 has shape [num_experts, *w13_per_expert_shape]. The
        # strided view exposes the same per-expert layout but with one
        # row per pool block, separated by `page_size_elems` rather than
        # by the natural per-expert size. Within a row, the strides
        # match the natural row-major layout that DMAs deposit.
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
        # Sanity: kernel asserts stride(-1) == 1 on the weight tensors
        # (vllm/model_executor/layers/fused_moe/fused_moe.py).
        assert self.pool_w13_view.stride(-1) == 1
        assert self.pool_w2_view.stride(-1) == 1

        # ---- Per-layer expert -> block_id lookup (GPU) ----
        # int64 to match the kernel's int64 cast on expert ids and to
        # avoid overflow in stride * offset when `topk_ids` is indexed
        # through it. Sentinel `_UNLOADED = -1` means "not in pool."
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
        self.expert_lru[expert_id] = step  # MRU on insert
        # Mirror to GPU lookup for the forward-path topk_ids remap.
        self.block_id_at[expert_id] = block_id

    def drop(self, block_id: int) -> int | None:
        expert_id = self.expert_at_block.pop(block_id, None)
        if expert_id is None:
            return None
        del self.block_at_expert[expert_id]
        self.expert_lru.pop(expert_id, None)
        # Invalidate the GPU lookup entry so a stale block_id can't be
        # returned by ``block_id_at[topk_ids]``. ``ensure_loaded`` is
        # responsible for ensuring no expert id in the next forward
        # ever resolves to ``_UNLOADED``.
        self.block_id_at[expert_id] = self._UNLOADED
        return expert_id

    def bump_expert(self, expert_id: int, step: int) -> None:
        """Bump an expert to MRU on this layer's expert-recency list.

        Called for every expert touched in a forward — both hits (the
        expert was already present) and misses (the expert was just
        DMA'd in). The bump records ``step`` as the last-used time so
        eviction can compare against the prefix list's recency.
        """
        if expert_id in self.expert_lru:
            self.expert_lru[expert_id] = step
            self.expert_lru.move_to_end(expert_id, last=True)


class UnifiedPoolManager:
    """Manages per-layer pools, the cross-layer page index, and the
    BlockPool callbacks for KV-driven cross-layer invalidation and
    prefix-LRU upkeep.

    The cross-layer index ``block_holder`` tracks two concerns
    simultaneously (per plan §7.5): membership ("which layers hold an
    expert at page P") and eviction driver ("which layers must be
    invalidated when P is reused for KV"). Care is taken that
    expert-driven invalidation only touches the calling layer; only
    KV-driven allocation broadcasts.

    The shared ``prefix_lru`` is the prefix-side half of the per-layer
    mixed LRU (§2). Every layer reads from it at eviction time; we
    keep one shared copy rather than N redundant per-layer copies
    because attention reads block_id B at every layer in lockstep, so
    prefix hit-recency is naturally global.
    """

    def __init__(self, block_pool, device: torch.device) -> None:
        from vllm.v1.core.block_pool import BlockPool

        assert isinstance(block_pool, BlockPool)
        self.block_pool: BlockPool = block_pool
        self.device = device
        self.layers: dict[int, UnifiedPool] = {}
        self.block_holder: dict[int, set[int]] = {}
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.step = 0  # increments per forward; used as MRU timestamp

        # Shared prefix-recency list. Block_id -> last-used step.
        # OrderedDict iteration gives oldest first; move_to_end on
        # every prefix hit. Maintained via BlockPool callbacks.
        self.prefix_lru: OrderedDict[int, int] = OrderedDict()

        self.block_pool.register_on_allocation_callback(self._on_kv_allocation)
        self.block_pool.register_on_prefix_added_callback(self._on_prefix_added)
        self.block_pool.register_on_prefix_removed_callback(
            self._on_prefix_removed
        )

    def register_layer(self, layer: UnifiedPool) -> None:
        assert layer.layer_idx not in self.layers, (
            f"Layer {layer.layer_idx} already registered with the unified pool"
        )
        self.layers[layer.layer_idx] = layer

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

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
        """Drop ``layer``'s mapping at ``block_id`` if any.

        Used when ``layer`` itself is overwriting bytes at this block
        (expert miss in this layer). Other layers' mappings are left
        untouched — their physical bytes are not affected.
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
        """Drop every layer's mapping at ``block_id`` (KV allocation).

        Used when KV writes physically overwrite all layers' bytes at
        this block. In Phase 2 the kernel reads ``pool_buffer`` directly,
        so dropping a mapping while the kernel is reading those bytes
        would be silent corruption. Guard against it by asserting no
        layer has the block pinned. The synchronous engine loop
        (``async_scheduling=False`` is required) guarantees scheduler
        KV allocations and worker forwards never overlap, so this
        assertion should hold trivially — it's defense-in-depth.
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
        """Clear the prefix hash for ``block_id`` across the BlockPool.

        Cached prefix is a global concept (same hash at every layer),
        so once any layer overwrites its bytes the prefix is broken
        everywhere. Hash-clear fires the on_prefix_removed callback
        which drops the entry from ``prefix_lru``.
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

    # ------------------------------------------------------------------
    # BlockPool callbacks
    # ------------------------------------------------------------------

    def _on_kv_allocation(self, block_ids: list[int]) -> None:
        """Scheduler claimed these blocks for active KV — KV writes are
        about to overwrite all layers' bytes there. Drop every layer's
        expert mapping at each block. (Prefix hashes were already
        cleared by ``_maybe_evict_cached_block`` before this callback
        fired, which also fired ``_on_prefix_removed``.)
        """
        for block_id in block_ids:
            self._broadcast_drop_all_layers(block_id, cause="kv-alloc")

    def _on_prefix_added(self, block_id: int) -> None:
        """A block re-entered the free queue with a hash → it is now
        an evictable cached-prefix entry. Bump it to MRU on the
        prefix-recency list. The bump-on-free is the bump-on-hit
        signal for prefix recency: the request that just released it
        finished using these bytes at this step.
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
        """The block stopped being an evictable cached-prefix entry —
        either ``touch`` pinned it for active use, or its hash was
        cleared. Either way, drop from the prefix-recency list.
        """
        removed = self.prefix_lru.pop(block_id, None)
        if _trace_enabled():
            was_present = "yes" if removed is not None else "no"
            print(
                f"UNIFIED PREFIX_REMOVED p{block_id} "
                f"was_present={was_present} size={len(self.prefix_lru)}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Warm-up at Stage 2
    # ------------------------------------------------------------------

    def warm_up(self, warm_count: int) -> None:
        """Pre-load ``warm_count`` experts per layer at startup.

        One block_id is shared across all layers per expert: every
        layer's E_k lives at the same block_id (different bytes in
        each layer's pool buffer, since pool buffers are per-layer).
        Total block_ids consumed = ``warm_count``, regardless of
        ``num_layers``. So the minimum pool size is
        ``warm_count + 1`` (the +1 is the null block at id 0).

        Trade-off vs disjoint warming: because every warmed block_id
        has all layers as holders, any KV-driven kv-broadcast on one
        of these ids will drop expert mappings at every layer at
        once. That's the cost of the shared-id contract.
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
        # Warm-up DMAs are queued on ``transfer_stream``. Make the
        # default (compute) stream wait for them, then full-device
        # synchronize. Phase-2 critical: if the first forward has all
        # hits and no miss-DMA, ``ensure_loaded`` never calls
        # ``wait_stream`` and the kernel would otherwise read stale
        # pool bytes. Make sure warm-up is fully flushed before any
        # forward can start.
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

        # ---- Post-warm-up sanity check (Phase 2) ----
        # Verify pool_w13_view[block_id] and pool_w2_view[block_id] hold
        # exactly cpu_w13[expert_id] / cpu_w2[expert_id] for every warmed
        # (expert, block) pair. If this fails, either the strided view
        # is misaligned with the DMA byte layout, or the warm-up DMAs
        # have not actually landed.
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

    # ------------------------------------------------------------------
    # Forward-path API
    # ------------------------------------------------------------------

    def ensure_loaded(
        self, layer: UnifiedPool, needed_expert_ids: list[int]
    ) -> None:
        """Ensure all ``needed_expert_ids`` are present at ``layer``'s
        pool. Pins all hits and miss-claimed blocks for the duration
        of the forward (released by ``release_pinned``). Mandatory
        ``wait_stream`` barrier after DMAs.

        Bumps every needed expert to MRU on ``layer.expert_lru`` —
        hits and misses both. Without this bump the LRU would track
        claim-recency rather than use-recency (the §7.6 failure mode).

        Trace snapshots are captured BEFORE any state mutation
        (per plan §7.2).
        """
        # --- Pass 1: classify hits/misses (no mutation yet) ---
        hit_results: list[tuple[int, int]] = []  # (eid, block_id)
        miss_eids: list[int] = []
        needed_set = set(needed_expert_ids)
        for eid in needed_expert_ids:
            if layer.has_expert(eid):
                hit_results.append((eid, layer.block_of_expert(eid)))
            else:
                miss_eids.append(eid)

        # Trace BEFORE mutation. Per-step header + per-layer composition
        # snapshot + LRU orderings + the router's request.
        if _trace_enabled():
            self._trace_pre_mutation(layer, needed_expert_ids)

        # Now apply mutation: bump hit/miss counters, pin hit blocks,
        # and bump every hit expert to MRU on the layer's LRU.
        layer.hits += len(hit_results)
        layer.misses += len(miss_eids)
        for eid, block_id in hit_results:
            layer.pinned_blocks.add(block_id)
            layer.bump_expert(eid, self.step)

        # --- Pass 2: claim a block per miss. ---
        miss_assignments: list[tuple[int, int, str]] = []  # (eid, bid, tier)
        for eid in miss_eids:
            block, tier = self._select_victim_block(layer, needed_set)
            block_id = block.block_id
            cause = f"expert-L{layer.layer_idx}"
            if _trace_enabled() and tier.startswith("free"):
                # Tier 1 free-claim has no eviction line; emit a CLAIM
                # so the trace still shows where each miss landed.
                print(
                    f"UNIFIED CLAIM page={block_id} L{layer.layer_idx} "
                    f"E{eid} cause={cause} tier={tier}",
                    flush=True,
                )
            self._evict_prefix_globally(block_id, cause=cause, tier=tier)
            self._drop_layer_mapping(layer, block_id, cause=cause, tier=tier)
            # ``assign`` records ``step`` as MRU for this expert.
            layer.assign(block_id, eid, step=self.step)
            self._add_holder(layer.layer_idx, block_id)
            layer.pinned_blocks.add(block_id)
            miss_assignments.append((eid, block_id, tier))
            self.block_pool.free_block_queue.append(block)

        if _trace_enabled():
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

        # --- Pass 3: batch DMAs on transfer_stream + barrier. ---
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
        if self.step % 100 == 0:
            self.log_stats()

    # ------------------------------------------------------------------
    # Victim selection — the plan's per-layer mixed LRU at query time.
    # ------------------------------------------------------------------

    def _select_victim_block(
        self, layer: UnifiedPool, needed_set: set[int]
    ):
        """Pick the coldest evictable block for ``layer``.

        Returns a ``(block, tier)`` tuple where ``tier`` is one of
        ``"free-pure"`` (truly empty slot), ``"free-cross-layer-expert"``
        (slot holds another layer's expert; safe to claim because pool
        buffers are per-layer), ``"expert-local"`` (evicted from L's
        own expert LRU), or ``"prefix-global"`` (evicted from the
        shared prefix LRU).

        Three-tier search:

        1. **Free space from L's view.** Walk ``free_block_queue`` and
           return the first block that has no hash and no expert
           mapping in ``layer``.
        2. **Coldest entry in L's mixed LRU.** Compare front of
           ``layer.expert_lru`` against front of the shared
           ``prefix_lru``; whichever has the smaller ``step`` (last
           used longer ago) wins. Skip pinned and currently-needed.
        3. **Fail loudly.** If neither side has a non-pinned
           candidate, the pool is exhausted — raise.
        """
        queue = self.block_pool.free_block_queue

        # Tier 1: free space from L's perspective.
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

        # Tier 2: coldest of (L's expert LRU, shared prefix LRU).
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

    # ------------------------------------------------------------------
    # DMA helpers
    # ------------------------------------------------------------------

    def _dma_expert_into_block_async(
        self, layer: UnifiedPool, expert_id: int, block_id: int
    ) -> None:
        """Async CPU->GPU copy of expert weights into ``layer``'s pool
        page at ``block_id``. The Phase 1 kernel never reads these
        bytes, but the DMA still measures PCIe latency honestly.
        """
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

    # ------------------------------------------------------------------
    # Trace helpers (only called when VLLM_UNIFIED_POOL_TRACE=1)
    # ------------------------------------------------------------------

    def _trace_pre_mutation(
        self, layer: UnifiedPool, needed_expert_ids: list[int]
    ) -> None:
        """Emit the per-step header, per-layer pool composition, both
        LRU orderings (MRU→LRU), and the router's request — all
        captured BEFORE any state mutation in ``ensure_loaded``.
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

        print(
            f"=== STEP {self.step} L{layer.layer_idx} "
            f"need={needed_expert_ids} ===",
            flush=True,
        )
        print(
            f"UNIFIED CACHE L{layer.layer_idx} "
            f"occ {n_expert_ours}/{capacity} ours "
            f"(expert-ours={n_expert_ours}, expert-other={n_held_other}, "
            f"prefix={n_prefix}, alloc-kv={n_alloc_kv}, "
            f"pinned={n_pinned}, free-pure={n_free_pure})",
            flush=True,
        )

        # MRU→LRU iteration: OrderedDict puts MRU at the END
        # (move_to_end last=True), so reverse for display.
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

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

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
