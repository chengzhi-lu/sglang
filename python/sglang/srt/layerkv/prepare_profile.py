"""Opt-in CPU call attribution for chunked expert preparation (no CUDA sync)."""

from __future__ import annotations

import cProfile
import sys
import time


class PrepareProfiler:
    def __init__(self):
        self.functions = {name: {} for name in ("first_shape", "repeat_shape")}
        self.call_edges = {name: {} for name in self.functions}
        self.calls = dict.fromkeys(self.functions, 0)
        self.wall_ms = dict.fromkeys(self.functions, 0.0)
        self.shapes = {}

    def run(self, prepare, state, dispatch, **kwargs):
        # Do not silently replace another tool's thread profiler.
        if sys.getprofile() is not None:
            raise RuntimeError(
                "prepare profiling cannot nest inside another Python profiler"
            )
        ids = dispatch.topk_output.topk_ids
        key = (
            tuple(ids.shape),
            str(ids.dtype),
            str(ids.device),
            int(state.slot_capacity),
        )
        first = key not in self.shapes
        shape = self.shapes.setdefault(
            key,
            dict(
                shape=list(ids.shape),
                dtype=str(ids.dtype),
                device=str(ids.device),
                slots=int(state.slot_capacity),
                calls=0,
                first_wall_ms=0.0,
                repeat_wall_ms=0.0,
            ),
        )
        bucket = "first_shape" if first else "repeat_shape"
        # Keep each profile's stack local to one invocation. Aggregate counters,
        # not live profiler instances that repeatedly enable/disable in Python.
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        try:
            return prepare(state, dispatch, **kwargs)
        finally:
            profiler.disable()
            elapsed = (time.perf_counter() - started) * 1000
            self.calls[bucket] += 1
            self.wall_ms[bucket] += elapsed
            shape["calls"] += 1
            shape["first_wall_ms" if first else "repeat_wall_ms"] += elapsed
            for entry in profiler.getstats():
                file, line, name = self._identity(entry.code)
                row = self.functions[bucket].setdefault(
                    (file, line, name),
                    dict(
                        file=file,
                        line=line,
                        name=name,
                        calls=0,
                        recursive_calls=0,
                        self_ms=0.0,
                        total_ms=0.0,
                    ),
                )
                self._add_counters(row, entry)
                # Direct caller attribution distinguishes e.g. remap scalar
                # waits from unrelated Tensor.item calls. Merge only after the
                # profiler is disabled; do not add any device synchronization.
                for child in entry.calls or ():
                    child_file, child_line, child_name = self._identity(child.code)
                    edge = self.call_edges[bucket].setdefault(
                        ((file, line, name), (child_file, child_line, child_name)),
                        dict(
                            caller_file=file,
                            caller_line=line,
                            caller_name=name,
                            file=child_file,
                            line=child_line,
                            name=child_name,
                            calls=0,
                            recursive_calls=0,
                            self_ms=0.0,
                            total_ms=0.0,
                        ),
                    )
                    self._add_counters(edge, child)

    @staticmethod
    def _identity(code):
        if isinstance(code, str):
            return "~", 0, code
        return code.co_filename, code.co_firstlineno, code.co_name

    @staticmethod
    def _add_counters(row, entry):
        row["calls"] += entry.callcount
        row["recursive_calls"] += entry.reccallcount
        row["self_ms"] += entry.inlinetime * 1000
        row["total_ms"] += entry.totaltime * 1000

    def summary(self):
        result = {
            "schema_version": 2,
            "shapes": [dict(shape) for shape in self.shapes.values()],
        }
        for name, counters in self.functions.items():
            result[name] = dict(
                calls=self.calls[name],
                wall_ms=self.wall_ms[name],
                functions=sorted(
                    (dict(row) for row in counters.values()),
                    key=lambda row: -row["total_ms"],
                ),
                call_edges=sorted(
                    (dict(row) for row in self.call_edges[name].values()),
                    key=lambda row: -row["total_ms"],
                ),
            )
        return result
