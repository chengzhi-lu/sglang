"""CUDA physical pages shared between stable KV and expert virtual addresses.

No extension build is needed: use the optional CUDA Python driver bindings.
Transfers synchronize the device deliberately; this is a correctness-first path.
Only whole driver-granularity pages can change owner. No CUDA allocation occurs
in transfer/return. See https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html.
"""

from __future__ import annotations

import math
import time
import weakref

import torch


def _check(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"LayerKV CUDA VMM: {error}")
    return values[0] if len(values) == 1 else tuple(values)


class _ArrayView:
    def __init__(self, allocation, nbytes):
        # torch's CUDA array interface importer retains this object, which
        # keeps both the VA and arena alive as long as any tensor aliases it.
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "strides": None,
            "typestr": "|u1",
            "data": (allocation.ptr, False),
            "version": 3,
        }


class VMMAllocation:
    def __init__(self, arena, shape, dtype, mapped_bytes, kind):
        self.arena = arena
        self.shape, self.dtype, self.kind = tuple(shape), dtype, kind
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.size = arena.round_up(self.nbytes)
        self.ptr = int(_check(arena.driver.cuMemAddressReserve(self.size, 0, 0, 0)))
        self.pages = {}
        self.closed = False
        try:
            for page in range(arena.round_up(mapped_bytes) // arena.page_bytes):
                handle = arena.new_handle()
                try:
                    self.map(page, handle)
                except Exception:
                    arena.release_handle(handle)
                    raise
        except Exception:
            self._close()
            raise

    def map(self, page, handle):
        if page in self.pages or not 0 <= page < self.size // self.arena.page_bytes:
            raise ValueError("VMM page already mapped or outside reservation")
        d = self.arena.driver
        ptr = self.ptr + page * self.arena.page_bytes
        _check(d.cuMemMap(ptr, self.arena.page_bytes, 0, handle, 0))
        try:
            _check(d.cuMemSetAccess(ptr, self.arena.page_bytes, [self.arena.access], 1))
        except Exception:
            _check(d.cuMemUnmap(ptr, self.arena.page_bytes))
            raise
        self.pages[page] = handle

    def unmap(self, page):
        handle = self.pages[page]
        _check(
            self.arena.driver.cuMemUnmap(
                self.ptr + page * self.arena.page_bytes, self.arena.page_bytes
            )
        )
        del self.pages[page]
        return handle

    def tensor(self, shape=None, *, allow_unmapped=False):
        shape = self.shape if shape is None else tuple(shape)
        nbytes = math.prod(shape) * self.dtype.itemsize
        if not 0 < nbytes <= self.nbytes:
            raise ValueError("tensor exceeds VMM reservation")
        if not allow_unmapped and any(
            page not in self.pages
            for page in range(self.arena.round_up(nbytes) // self.arena.page_bytes)
        ):
            raise ValueError("tensor includes unmapped VMM pages")
        raw = torch.as_tensor(_ArrayView(self, nbytes), device=self.arena.device)
        return raw.view(self.dtype).view(shape)

    def _close(self):
        if self.closed:
            return
        # Destruction, like transfer, must not race kernels on another stream.
        torch.cuda.synchronize(self.arena.device)
        with torch.cuda.device(self.arena.device):
            for page in list(self.pages):
                self.arena.release_handle(self.unmap(page))
            _check(self.arena.driver.cuMemAddressFree(self.ptr, self.size))
        self.closed = True

    def __del__(self):
        try:
            self._close()
        except Exception:
            # CUDA may already be deinitialized during interpreter shutdown.
            pass


class SharedVMM:
    """One physical-page budget; KV donors fund expert growth and get it back.

    allocate() creates initial backing. lend() and recall() only move existing
    handles. Callers must first back up KV and exclude donor slots from all KV
    allocators. They must back up/invalidate expert tail slots before recall.
    The tensors are process-local and must not be used with CUDA graph capture.
    """

    def __init__(self, device, *, initial_kv_tokens=None):
        from cuda.bindings import driver as d

        self.driver = d
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("SharedVMM requires CUDA")
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        with torch.cuda.device(self.device):
            torch.cuda.init()
            self.prop = d.CUmemAllocationProp()
            self.prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            self.prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            self.prop.location.id = self.device.index
            self.page_bytes = int(
                _check(
                    d.cuMemGetAllocationGranularity(
                        self.prop,
                        d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                    )
                )
            )
        self.access = d.CUmemAccessDesc()
        self.access.location = self.prop.location
        self.access.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.handles = set()
        self.allocations = weakref.WeakValueDictionary()
        # KV buffers created through the pool factory must stay alive for the
        # lifetime of the pool; direct allocate() callers retain ownership
        # themselves and are intentionally not added here.
        self._kv_allocations = []
        self.initial_kv_tokens = (
            None if initial_kv_tokens is None else max(0, int(initial_kv_tokens))
        )
        self.loans = []  # (donor, donor_page, recipient, recipient_page)
        # Keep ownership separate from the public transfer tuple.  Multiple
        # shared expert layers can use one physical page budget, while a
        # layer-local recall must not return another layer's loans.
        self._loan_owners = []
        self.kv_overflow_segments = []
        self.create_count = 0
        self.transfer_count = 0
        self.return_count = 0
        self.peak_loan_bytes = 0
        self.lend_wait_ms = 0.0
        self.lend_remap_ms = 0.0
        self.recall_wait_ms = 0.0
        self.recall_remap_ms = 0.0

    def round_up(self, nbytes):
        return (int(nbytes) + self.page_bytes - 1) // self.page_bytes * self.page_bytes

    def new_handle(self):
        handle = _check(self.driver.cuMemCreate(self.page_bytes, self.prop, 0))
        self.handles.add(int(handle))
        self.create_count += 1
        return handle

    def release_handle(self, handle):
        _check(self.driver.cuMemRelease(handle))
        self.handles.remove(int(handle))

    def allocate(self, shape, dtype, *, kind, mapped_bytes=None):
        if kind not in ("kv", "expert") or any(int(x) <= 0 for x in shape):
            raise ValueError("invalid VMM allocation kind/shape")
        nbytes = math.prod(shape) * dtype.itemsize
        mapped_bytes = nbytes if mapped_bytes is None else int(mapped_bytes)
        if not 0 < mapped_bytes <= nbytes:
            raise ValueError("invalid initial VMM mapped size")
        with torch.cuda.device(self.device):
            allocation = VMMAllocation(self, shape, dtype, mapped_bytes, kind)
        self.allocations[allocation.ptr] = allocation
        return allocation

    def kv_zeros(self, shape, *, dtype, device):
        if torch.device(device).type != "cuda":
            raise ValueError("KV VMM factory requires CUDA tensors")
        initial_rows = shape[0]
        if self.initial_kv_tokens is not None:
            initial_rows = min(initial_rows, self.initial_kv_tokens + 1)
        row_bytes = math.prod(shape[1:]) * dtype.itemsize
        allocation = self.allocate(
            shape,
            dtype,
            kind="kv",
            mapped_bytes=max(1, initial_rows * row_bytes),
        )
        self._kv_allocations.append(allocation)
        # The full tensor is a valid VA view, but only the mapped prefix may be
        # touched before an overflow segment is activated.
        tensor = allocation.tensor(allow_unmapped=True)
        allocation.tensor((initial_rows,) + tuple(shape[1:])).zero_()
        return tensor

    def _validate_page_transfer(self, transfers, *, source_kind=None, target_kind=None):
        donors, targets = set(), set()
        for src, sp, dst, dp in transfers:
            if src.arena is not self or dst.arena is not self:
                raise ValueError("VMM transfer allocations must belong to this arena")
            if src.arena is not dst.arena:
                raise ValueError("VMM transfer allocations must share an arena")
            if source_kind is not None and src.kind != source_kind:
                raise ValueError("VMM transfer has an invalid source kind")
            if target_kind is not None and dst.kind != target_kind:
                raise ValueError("VMM transfer has an invalid target kind")
            if (
                sp not in src.pages
                or dp in dst.pages
                or not 0 <= dp < dst.size // src.arena.page_bytes
            ):
                raise ValueError("VMM donor absent or recipient not vacant")
            if (src.ptr, sp) in donors or (dst.ptr, dp) in targets:
                raise ValueError("duplicate page in VMM transfer")
            donors.add((src.ptr, sp))
            targets.add((dst.ptr, dp))

    def _move_pages(self, transfers):
        transfers = tuple(transfers)
        self._validate_page_transfer(transfers)
        if not transfers:
            return (), 0.0, 0.0
        wait_start = time.perf_counter()
        torch.cuda.synchronize(self.device)
        remap_start = time.perf_counter()
        completed = []
        with torch.cuda.device(self.device):
            try:
                for src, sp, dst, dp in transfers:
                    handle = src.unmap(sp)
                    try:
                        dst.map(dp, handle)
                    except Exception:
                        src.map(sp, handle)
                        raise
                    completed.append((src, sp, dst, dp))
            except Exception:
                for src, sp, dst, dp in reversed(completed):
                    src.map(sp, dst.unmap(dp))
                raise
        return (
            tuple(completed),
            (remap_start - wait_start) * 1000,
            (time.perf_counter() - remap_start) * 1000,
        )

    def move_pages(self, transfers):
        """Move exclusive physical pages between arbitrary arena allocations."""
        return self._move_pages(transfers)[0]

    def release_mapped_pages(self, pages):
        """Unmap and release pages that have no remaining owner."""
        pages = tuple(pages)
        seen = set()
        for allocation, page in pages:
            key = (allocation.ptr, int(page))
            if key in seen or page not in allocation.pages:
                raise ValueError("release requires unique mapped VMM pages")
            seen.add(key)
        if not pages:
            return
        torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            for allocation, page in pages:
                self.release_handle(allocation.unmap(page))

    def _kv_overflow_destinations(self, base_tokens, requested_tokens):
        allocations = tuple(self._kv_allocations)
        if not allocations or requested_tokens <= 0:
            return (), 0
        row_layout = []
        for allocation in allocations:
            row_bytes = math.prod(allocation.shape[1:]) * allocation.dtype.itemsize
            if row_bytes <= 0 or self.page_bytes % row_bytes:
                raise ValueError("sparse KV overflow requires page-aligned rows")
            row_layout.append(row_bytes)
        if len(set(row_layout)) != 1:
            raise ValueError("sparse KV overflow requires uniform K/V row sizes")
        row_bytes = row_layout[0]
        base_rows = int(base_tokens) + 1  # include padded slot zero
        end_rows = base_rows + int(requested_tokens)
        start_page = self.round_up(base_rows * row_bytes) // self.page_bytes
        end_page = self.round_up(end_rows * row_bytes) // self.page_bytes
        destinations = [
            (allocation, page)
            for allocation in allocations
            for page in range(start_page, end_page)
            if page not in allocation.pages
        ]
        return tuple(destinations), row_bytes

    def activate_kv_overflow(self, source_pages, *, base_tokens, requested_tokens):
        """Map expert-owned pages into the sparse KV tail.

        The returned segment is a list of exact page transfers.  The caller is
        responsible for making the corresponding token range visible to its
        allocator only after this method succeeds.
        """
        requested_tokens = max(0, int(requested_tokens))
        source_pages = tuple(source_pages)
        if requested_tokens <= 0:
            return {"base_tokens": int(base_tokens), "tokens": 0, "transfers": ()}, ()

        def destinations(tokens):
            return self._kv_overflow_destinations(base_tokens, tokens)[0]

        dests = destinations(requested_tokens)
        if len(dests) > len(source_pages):
            lo, hi = 0, requested_tokens
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(destinations(mid)) <= len(source_pages):
                    lo = mid
                else:
                    hi = mid - 1
            requested_tokens = lo
            dests = destinations(requested_tokens)
        if len(dests) > len(source_pages):
            raise ValueError("expert pages cannot fund requested KV overflow")
        used = source_pages[: len(dests)]
        self._validate_page_transfer(
            tuple((src, page, dst, dp) for (src, page), (dst, dp) in zip(used, dests)),
        )
        completed = self.move_pages(
            tuple((src, page, dst, dp) for (src, page), (dst, dp) in zip(used, dests))
        )
        segment = {
            "base_tokens": int(base_tokens),
            "tokens": int(requested_tokens),
            "transfers": completed,
        }
        self.kv_overflow_segments.append(segment)
        unused = source_pages[len(dests) :]
        return segment, unused

    def deactivate_kv_overflow(self, segment):
        """Return one previously activated KV tail segment to its expert pages."""
        if not self.kv_overflow_segments or self.kv_overflow_segments[-1] is not segment:
            raise ValueError("KV overflow segments must be deactivated in order")
        transfers = tuple(segment["transfers"])
        self.move_pages((dst, dp, src, sp) for src, sp, dst, dp in reversed(transfers))
        self.kv_overflow_segments.pop()
        return transfers

    def lend(self, transfers, *, owner=None):
        """Atomically transfer a batch of exclusive pages, or roll it back."""
        transfers = tuple(transfers)
        self._validate_page_transfer(
            transfers, source_kind="kv", target_kind="expert"
        )
        completed, wait_ms, remap_ms = self._move_pages(transfers)
        self.lend_wait_ms += wait_ms
        self.lend_remap_ms += remap_ms
        self.loans.extend(completed)
        self._loan_owners.extend([owner] * len(completed))
        self.transfer_count += len(completed)
        self.peak_loan_bytes = max(
            self.peak_loan_bytes, len(self.loans) * self.page_bytes
        )

    def recall(self, *, owner=None):
        """Return borrowed pages after the caller invalidates expert tail rows.

        ``owner=None`` preserves the original whole-arena behavior.  A
        controller owner returns only its own transfers, which is required
        when all expert layers share this arena.
        """
        wait_start = time.perf_counter()
        torch.cuda.synchronize(self.device)
        remap_start = time.perf_counter()
        self.recall_wait_ms += (remap_start - wait_start) * 1000
        with torch.cuda.device(self.device):
            indices = (
                range(len(self.loans) - 1, -1, -1)
                if owner is None
                else (
                    index
                    for index in range(len(self.loans) - 1, -1, -1)
                    if self._loan_owners[index] is owner
                )
            )
            for index in tuple(indices):
                src, sp, dst, dp = self.loans[index]
                handle = dst.unmap(dp)
                try:
                    src.map(sp, handle)
                except Exception:
                    dst.map(dp, handle)
                    raise
                self.loans.pop(index)
                self._loan_owners.pop(index)
                self.return_count += 1
        self.recall_remap_ms += (time.perf_counter() - remap_start) * 1000

    def loan_count(self, owner=None):
        if owner is None:
            return len(self.loans)
        return sum(item is owner for item in self._loan_owners)

    def summary(self):
        owners = [
            int(handle)
            for a in self.allocations.values()
            for handle in a.pages.values()
        ]
        return {
            "page_bytes": self.page_bytes,
            "physical_bytes": len(self.handles) * self.page_bytes,
            "physical_create_count": self.create_count,
            "kv_to_expert_pages": self.transfer_count,
            "expert_to_kv_pages": self.return_count,
            "loan_bytes": len(self.loans) * self.page_bytes,
            "peak_loan_bytes": self.peak_loan_bytes,
            "kv_overflow_segments": len(self.kv_overflow_segments),
            "kv_overflow_pages": sum(
                len(segment["transfers"]) for segment in self.kv_overflow_segments
            ),
            "lend_wait_ms": self.lend_wait_ms,
            "lend_remap_ms": self.lend_remap_ms,
            "recall_wait_ms": self.recall_wait_ms,
            "recall_remap_ms": self.recall_remap_ms,
            "ownership_guard_pass": len(owners) == len(set(owners))
            and set(owners) == self.handles,
        }
