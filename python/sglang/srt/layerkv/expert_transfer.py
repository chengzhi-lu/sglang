"""CUDA API batch copies with explicit tensor and host-pool lifetimes."""

from collections import Counter, deque

import torch


class ExpertBatchTransfer:
    def __init__(self, stats):
        from cuda.bindings import runtime

        if not hasattr(runtime, "cudaMemcpyBatchAsync"):
            raise RuntimeError(
                "cuda-batch requires CUDA bindings with cudaMemcpyBatchAsync"
            )
        self.cuda = runtime
        self.stats = stats
        self.pending = deque()
        self.host_inflight = Counter()
        self.deferred = {}
        self.streams = {}
        self.failed_refs = []
        self.failed_host_keys = set()
        # Only CUDA API attributes are invariant across submissions. Never cache
        # tensor views, pointers or pinned attestations: backing and VMM slots
        # can be replaced/reused while this copier remains alive.
        self._attributes = {}

    def _copy_attributes(self, direction, device):
        key = (direction, device)
        attr = self._attributes.get(key)
        if attr is None:
            cuda = self.cuda
            attr = cuda.cudaMemcpyAttributes()
            attr.srcAccessOrder = (
                cuda.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
            )
            host = cuda.cudaMemLocationType.cudaMemLocationTypeHost
            gpu = cuda.cudaMemLocationType.cudaMemLocationTypeDevice
            h2d = direction == "h2d"
            attr.srcLocHint.type = host if h2d else gpu
            attr.dstLocHint.type = gpu if h2d else host
            attr.srcLocHint.id = 0 if h2d else device.index
            attr.dstLocHint.id = device.index if h2d else 0
            attr.flags = 0
            self._attributes[key] = attr
        return attr

    def collect(self, *, block=False):
        for _ in range(len(self.pending)):
            # Keep owners queued if CUDA event inspection itself raises.
            event, refs, host_keys = self.pending[0]
            if block:
                event.synchronize()
            elif not event.query():
                self.pending.rotate(-1)
                continue
            self.pending.popleft()
            for key in host_keys:
                self.host_inflight[key] -= 1
                if not self.host_inflight[key]:
                    del self.host_inflight[key]
                    release = self.deferred.pop(key, None)
                    if release is not None:
                        tensor, callback = release
                        callback(tensor)
        self.stats.expert_cuda_batch_pending = len(self.pending)

    def defer_release(self, tensor, callback):
        key = tensor.untyped_storage().data_ptr()
        if key not in self.host_inflight and key not in self.failed_host_keys:
            return False
        self.deferred[key] = (tensor, callback)
        self.stats.expert_cuda_batch_deferred_release_count += 1
        return True

    def copy(self, pairs, stream):
        """Submit independent (destination, source) tensor copies in stream order.

        CPU buffers must be pinned; CUDA views are retained until completion.
        Errors are fatal, never retried as a partially submitted batch.
        """
        if not pairs:
            return
        if self.failed_refs:
            raise RuntimeError(
                "CUDA batch copier cannot be reused after a failed submission"
            )
        self.collect()
        dsts, srcs, sizes = [], [], []
        host_keys = set()
        destinations = []
        direction = None
        stream_device = stream.device
        for dst, src in pairs:
            dst_is_cuda = dst.is_cuda
            if (
                not dst.is_contiguous()
                or not src.is_contiguous()
                or dst.shape != src.shape
                or dst.dtype != src.dtype
                or dst_is_cuda == src.is_cuda
            ):
                raise ValueError(
                    "batch copies require matching contiguous CPU/CUDA tensors"
                )
            host, device = (src, dst) if dst_is_cuda else (dst, src)
            if not host.is_pinned() or device.device != stream_device:
                raise ValueError(
                    "batch copies require pinned CPU memory and one CUDA device"
                )
            copy_direction = "h2d" if dst_is_cuda else "d2h"
            if direction is not None and direction != copy_direction:
                raise ValueError("a batch must have one transfer direction")
            direction = copy_direction
            nbytes = src.nbytes
            if not nbytes:
                continue
            dst_ptr = dst.data_ptr()
            dsts.append(dst_ptr)
            srcs.append(src.data_ptr())
            sizes.append(nbytes)
            # Equal shape/dtype and contiguous layouts were checked above.
            destinations.append((dst_ptr, dst_ptr + nbytes))
            host_keys.add(host.untyped_storage().data_ptr())
        if not sizes:
            return
        destinations.sort()
        if any(a[1] > b[0] for a, b in zip(destinations, destinations[1:])):
            raise ValueError("CUDA batch destinations must not overlap")

        # The batch API does not accept the legacy NULL stream. Bridge it with
        # device-side event dependencies, not a host synchronization.
        active = stream
        if stream.cuda_stream == 0:
            active = self.streams.get(stream_device)
            if active is None:
                active = torch.cuda.Stream(device=stream_device)
                self.streams[stream_device] = active
            active.wait_stream(stream)
        cuda = self.cuda
        attr = self._copy_attributes(direction, stream_device)
        event = torch.cuda.Event()
        self.failed_refs = pairs  # Retain raw-pointer owners even if submission fails.
        self.failed_host_keys = host_keys
        with torch.cuda.device(stream_device):
            result = cuda.cudaMemcpyBatchAsync(
                dsts, srcs, sizes, len(sizes), [attr], [0], 1, active.cuda_stream
            )
        if result[0] != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyBatchAsync failed: {result}")
        event.record(active)
        self.pending.append((event, pairs, host_keys))
        self.host_inflight.update(host_keys)
        self.failed_refs = []
        self.failed_host_keys = set()
        if active is not stream:
            stream.wait_event(event)
        field = f"expert_cuda_batch_{direction}_count"
        setattr(self.stats, field, getattr(self.stats, field) + 1)
        self.stats.expert_cuda_batch_copy_count += len(sizes)
        self.stats.expert_cuda_batch_bytes += sum(sizes)
        self.stats.expert_cuda_batch_pending = len(self.pending)
