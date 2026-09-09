"""Exact-shape replay for an immutable, standard, unquantized MoE core.

Not a whole-layer graph: routing/hotness, dispatch/combine and all KV work remain
outside. Callers must exclude offloaded experts and invoke this only in decode.
This module is not enabled or installed by default.
"""

import time

import torch


def validate_native_moe_graph_options(args):
    limit = args.layerkv_native_moe_graph_max_batch_size
    if limit < 0:
        raise ValueError("native MoE graph batch limit must be nonnegative")
    if limit and (
        not args.enable_layerkv
        or not (
            getattr(args, "layerkv_shared_expert_all_layers", False)
            or args.layerkv_shared_expert_layer >= 0
        )
        or not args.disable_overlap_schedule
        or not args.disable_cuda_graph
        or not args.disable_piecewise_cuda_graph
        or args.tp_size != 1
        or args.ep_size != 1
    ):
        raise ValueError(
            "native MoE graph requires LayerKV shared expert mode, TP/EP=1, "
            "disabled overlap and disabled outer CUDA graphs"
        )


def resident_core_eligible(module, layer_id, selected_layer):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    quant_method = getattr(module, "quant_method", None)
    runner = getattr(quant_method, "runner", None)
    return (
        selected_layer >= 0
        and layer_id != selected_layer
        and not getattr(module, "_layerkv_expert_wrapped", False)
        and getattr(runner, "runner_backend", None) == MoeRunnerBackend.TRITON
        and getattr(module, "moe_tp_size", None) == 1
        and getattr(module, "moe_ep_size", None) == 1
        and type(getattr(module, "quant_method", None)).__name__
        == "UnquantizedFusedMoEMethod"
        and all(
            p is not None and p.is_cuda and p.dtype in (torch.float16, torch.bfloat16)
            for p in (
                getattr(module, "w13_weight", None),
                getattr(module, "w2_weight", None),
            )
        )
    )


class NativeMoEGraph:
    def __init__(self, core, parameters, *, max_batch_size=32, warmup_calls=2):
        if max_batch_size <= 0 or warmup_calls < 1:
            raise ValueError("require positive batch limit and warmup calls")
        self.core, self.parameters = core, parameters
        self.max_batch_size, self.warmup_calls = max_batch_size, warmup_calls
        self.entry = None
        self.warm_key, self.warm_count = None, 0
        self.captures = self.replays = self.fallbacks = 0
        self.workspace_bytes = 0
        self.recapture_count = 0
        self.capture_wall_ms = 0.0
        self.capture_sync_ms = 0.0
        self.capture_context_ms = 0.0

    @staticmethod
    def _signature(tensor):
        if tensor is None:
            return None
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    @staticmethod
    def _inputs(dispatch):
        topk = dispatch.topk_output
        return (
            dispatch.hidden_states,
            topk.topk_weights,
            topk.topk_ids,
            topk.router_logits,
        )

    def _eligible(self, dispatch):
        if (
            torch.is_grad_enabled()
            or getattr(dispatch, "hidden_states_scale", None) is not None
        ):
            return False
        topk = getattr(dispatch, "topk_output", None)
        if (
            not hasattr(dispatch, "_replace")
            or not hasattr(topk, "_replace")
            or not all(
                hasattr(topk, name)
                for name in ("topk_weights", "topk_ids", "router_logits")
            )
        ):
            return False
        tensors = self._inputs(dispatch)
        hidden = tensors[0]
        return (
            hidden.is_cuda
            and hidden.dtype in (torch.float16, torch.bfloat16)
            and hidden.ndim == 2
            and 0 < hidden.shape[0] <= self.max_batch_size
            and all(
                t is None or (t.is_contiguous() and t.device == hidden.device)
                for t in tensors
            )
            and not torch.cuda.is_current_stream_capturing()
        )

    def __call__(self, dispatch):
        if not self._eligible(dispatch):
            self.fallbacks += 1
            return self.core(dispatch)
        inputs = self._inputs(dispatch)
        parameters = tuple(self.parameters())
        key = (
            tuple(self._signature(t) for t in inputs),
            tuple((p.data_ptr(), self._signature(p)) for p in parameters),
        )
        if self.entry is None or self.entry[0] != key:
            if self.warm_key != key:
                self.warm_key, self.warm_count = key, 0
            self.warm_count += 1
            if self.warm_count <= self.warmup_calls:
                self.fallbacks += 1
                return self.core(dispatch)
            self._capture(dispatch, parameters, key)
        _, graph, static, output, _keep_weights = self.entry
        for target, source in zip(self._inputs(static), inputs):
            if target is not None:
                target.copy_(source)
        graph.replay()
        # Native inplace MoE mutates dispatch.hidden_states. Preserve that contract
        # and never expose graph-owned output that a later replay can overwrite.
        dispatch.hidden_states.copy_(static.hidden_states)
        hidden = (
            dispatch.hidden_states
            if output.hidden_states.data_ptr() == static.hidden_states.data_ptr()
            else output.hidden_states.clone()
        )
        self.replays += 1
        return output._replace(hidden_states=hidden)

    def _capture(self, dispatch, parameters, key):
        started = time.perf_counter()
        replacing = self.entry is not None
        device = dispatch.hidden_states.device
        # Shape/weight changes are cold paths. One entry bounds graph storage.
        # Keep old weights alive until all uses finish before dropping old replay.
        torch.cuda.synchronize(device)
        self.capture_sync_ms += (time.perf_counter() - started) * 1000
        self.entry = None
        before = torch.cuda.memory_allocated(device)
        clones = tuple(
            t.clone() if t is not None else None for t in self._inputs(dispatch)
        )
        static = dispatch._replace(
            hidden_states=clones[0],
            topk_output=dispatch.topk_output._replace(
                topk_weights=clones[1], topk_ids=clones[2], router_logits=clones[3]
            ),
        )
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.core(static)
            for target, source in zip(self._inputs(static), self._inputs(dispatch)):
                if target is not None:
                    target.copy_(source)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Capture errors are not swallowed: an unsupported core must not silently
        # produce misleading replay results or mask a CUDA stream failure.
        context_started = time.perf_counter()
        with torch.cuda.graph(graph, stream=stream):
            output = self.core(static)
        self.capture_context_ms += (time.perf_counter() - context_started) * 1000
        if getattr(output, "_fields", ()) != ("hidden_states",):
            raise ValueError("native MoE replay requires a standard combine result")
        current.wait_stream(stream)
        self.entry = (key, graph, static, output, tuple(p.detach() for p in parameters))
        self.workspace_bytes = max(0, torch.cuda.memory_allocated(device) - before)
        self.captures += 1
        self.recapture_count += int(replacing)
        self.capture_wall_ms += (time.perf_counter() - started) * 1000
