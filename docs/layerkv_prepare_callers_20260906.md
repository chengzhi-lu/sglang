# Remaining prepare costs: direct caller attribution

## Outcome and scope

The retained post-load scalar readback is the largest individual self-time in
late CPU-known preparation, but it is a **host wait boundary**, not a measurement
of the remap kernel itself. Ready-event query/submission calls are small.
Backing collection spends much more time in CPU release callbacks than querying
events. The next low-risk candidate is avoiding duplicate validation/metadata
work between deferred release and actual pool admission, not deleting waits.

This turn changes only the opt-in `PrepareProfiler` and its tests. No allocator,
expert loading, GPU synchronization, backing ownership or default behavior was
changed. The prior unprofiled [CPU-known results](layerkv_cpu_known_prepare_20260906.md)
remain the end-to-end evidence; this instrumented run is diagnostic only.

The baseline was the previous two matched unprofiled repetitions, rechecked
from their summaries. The ranked probes were: backing ownership/control cost,
post-load scalar waiting, and ready-event processing. There is no new functional
failure or deterministic latency threshold to claim fixed. The new observation
seam was tested red first: scalar-caller and exception-edge tests both failed
with `KeyError: 'call_edges'` before implementation and pass afterwards.

## Instrumentation and interpretation

`prepare_profile.schema_version=2` retains existing function/shape tables and
adds `first_shape.call_edges` / `repeat_shape.call_edges`. Each edge records
caller file/first line/name, callee file/first line/name, calls, recursive calls,
self time and inclusive total time. These are **direct caller edges**, not a
complete stack trace or individual call-site line attribution. Multiple calls
to the same callee within one caller are combined. Builtins such as Tensor
`item` can now be separated by their Python caller.

The native cProfile subcall records are aggregated only after disabling each
per-invocation profiler. No GPU synchronization or device events were added.
The feature is enabled by the existing `--profile-chunks --profile-prepare`;
without those options the profiler is not instantiated. Shape buckets continue
to mean first/repeated routing signatures, not JIT or allocator cache keys.

For each request, merge both shape buckets by function identity or full
caller/callee identity, then subtract the preceding request's cumulative table.
The scalar edge call counts were checked against scalar function call counts.
The existing engine guard also checks primitive root calls against explicit
profile and chunk counts. Counts, exception handling, immutable snapshots and
distinct scalar callers are covered by tests.

`inner prepare wall` excludes profile-result aggregation but includes profiling
overhead inside preparation. `outer prepare wall` includes setup/aggregation
and hence is larger. In request 6, fixed16 outer is 293.346 → 262.126 ms and
grow17 is 294.970 → 263.700 ms, generic → CPU-known. Do not interpret that as an
uninstrumented improvement. Instrumentation between chunks can also affect
submission timing. Inclusive helper totals overlap; never sum nested totals.
CPU copy-submission intervals are not DMA durations. A scalar read can wait for
preceding gather/remap/transfer work; these profiles cannot split that waiting
into pure kernel, PCIe and scheduler components.

## Bounded reproduction

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1.
Six sequential 4096-input/64-output greedy requests per fixed16/grow17 arm,
same request-index inputs as the previous unprofiled runs. Keep BF16 for the
previously validated native GDN configuration; this is not an FP16 model result.
Only one matched diagnostic repetition was run, serialized without concurrent
GPU tests. No new packages or compilation were needed.

Artifacts under `/mnt/vdb/chengzhi/`:

- `layerkv_prepare_callers_20260906_generic_01/`
- `layerkv_prepare_callers_20260906_cpu_known_01/`

Both directories contain `plan.json`, `fixed16/grow17.engine_args.json`, logs,
per-request `fixed16/grow17.json`, and `summary.json`. Full caller tables live
in each request's `final_stats.shared_vmm.prepare_profile`.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_prepare_callers_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known \
  --profile-chunks --profile-prepare --execute
```

Use `--prepare-path generic` in a fresh directory for the control, or omit
`--execute` for a dry plan. `PATH` exposes external framework tools from the
existing environment; ordinary configuration is in CLI arguments. Other
recorded settings remain context 8192, KV capacity 16384, scratch 4095, block
2048, pressure 40 MiB, shared expert layer 0, initial slots 16 and extra slots
0/1. Profiling flags are identical between the two configurations.

## Per-request measurements

Milliseconds; G = generic, C = CPU-known. Allocation, release and ready columns
are inclusive CPU helper times. Scalar column is CPU-known caller → `item`
self time. Request 1 has explicitly zero profiled chunks, not zero total work.

| Arm | Request | G inner prepare | C inner prepare | C scalar | C backing allocation | C backing release | C ready helper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 2 | 417.318 | 384.000 | 50.631 | 154.904 | 23.823 | 6.345 |
| fixed16 | 3 | 288.662 | 262.651 | 51.009 | 37.853 | 24.562 | 6.055 |
| fixed16 | 4 | 251.004 | 227.282 | 45.194 | 26.127 | 21.851 | 5.445 |
| fixed16 | 5 | 270.719 | 245.372 | 47.487 | 33.709 | 23.304 | 5.730 |
| fixed16 | 6 | 233.962 | 210.110 | 43.530 | 13.450 | 21.572 | 5.463 |
| grow17 | 2 | 416.442 | 388.573 | 49.618 | 158.719 | 24.313 | 6.346 |
| grow17 | 3 | 290.263 | 264.423 | 49.847 | 38.380 | 25.240 | 6.216 |
| grow17 | 4 | 252.585 | 227.888 | 44.417 | 25.823 | 22.317 | 5.469 |
| grow17 | 5 | 271.602 | 247.679 | 47.252 | 34.072 | 23.820 | 5.816 |
| grow17 | 6 | 234.309 | 210.352 | 43.267 | 13.517 | 21.921 | 5.355 |

Generic request-6 post-load scalar time is 45.401/45.253 ms for fixed/grow,
with another 1.158/1.219 ms in `_try_gpu_expert_topk_remap -> item` demand
discovery. CPU-known removes the latter calls and unique discovery; it still
executes 55 post-load scalar reads in request 6. In fixed16, the generic GPU
remap helper and unique helper total 13.264 and 11.379 ms respectively (the
1.158 ms demand scalar is nested in the remap helper). This explains why
eliminating discovery alone leaves most prepare work, rather than proving a
large end-to-end gain.

Allocation remains significant early even with pooled backing. First-prefill
allocation-helper time is about 155–159 ms; by request 6 it is about 13.5 ms.
These are measured host call intervals, not proof of a particular driver
allocation mechanism or steady-state allocator behavior.

## Distinguishing event queries from CPU cleanup

CPU-known request 6, direct caller edges, inclusive milliseconds:

| Caller → callee | Calls, each arm | fixed16 | grow17 |
| --- | ---: | ---: | ---: |
| ready helper → Event.query | 55 | 0.132 | 0.137 |
| ready helper → Stream.wait_event | 55 | 0.143 | 0.146 |
| collect → Event.query | 163 | 0.636 | 0.617 |
| collect → backing release callback | 1066 | 14.280 | 14.499 |
| release params → initial backing release | 1074 | 7.292 | 7.422 |
| backing release → owns-storage check | 2140 | 5.232 | 5.344 |
| backing release → defer-release check | 2140 | 2.715 | 2.783 |

`collect` totals 20.364/20.570 ms, including those callbacks and queries;
backing release totals 21.572/21.921 ms across its two callers. These totals
overlap and must not be added. The ready helper's roughly 5.4 ms includes
bookkeeping; its event query/submission calls are much smaller. GPU waiting
scheduled by `wait_event` can instead be exposed at the later scalar read.

The source matches the duplicate-call evidence:
`_release_cpu_backing_tensor` checks full storage ownership and pinned status
**before** asking `defer_release`. When its copy finishes, `collect` invokes
that same release function again. It therefore repeats the checks before
actual pool admission. The 1074 initial calls correspond to 537 experts × 2
parameter tensors. Not all final callbacks run inside the profiled region,
hence 1066 callbacks rather than 1074. This is not evidence of a leak.

Other remaining CPU costs include materialization control and batch submission:
fixed16 request-6 `_materialize_experts` self time is 17.599 ms, transfer `copy`
self time 15.672 ms, and copy-descriptor recording self time 7.945 ms. H2D helper
inclusive time is 40.925 ms; do not add it to nested submission costs.

## Next implementation boundary

Prioritize the concrete, low-risk CPU duplication: run the expensive full
pool-eligibility validation at actual admission, not both before deferral and
again in the completion callback. Preserve early rejection where necessary for
non-owning views, keep the callback/strong references until completion, and
retain pool size limits and failed-transfer fail-closed behavior. The test
target is an owning pinned tensor with an incomplete real transfer: no early
reuse; one final full validation; exact copied data; no admission of views,
mutated/ineligible storage or failed-transfer owners. Recheck both individual
and batched-view backing modes so this does not improve one by penalizing the
other. Cache immutable metadata only with an explicit lifetime guarantee.

This optimization is **not implemented in this diagnostic turn**, and the
observed 14 ms callback total is not all removable. A subsequent matched
unprofiled test must establish any actual benefit. The larger scalar interval
requires a CUDA execution timeline before attempting overlap changes; simply
removing it does not remove required H2D work and can invalidate remap guards.
Event polling itself is not supported as the next primary target by this trace.

The subsequent [admission-time backing validation](layerkv_backing_admission_20260906.md)
implements that narrow validation-order change with an explicit eager control.

## Validation

Both summaries are valid. All 24 requests (1536 generated tokens) match the
corresponding prior unprofiled generic control **exactly**, including logprobs.
KVC/expert/ownership guards and comparability pass, and per-request chunk loads,
materializations, H2D/D2H batch counts and KV loan/return counts match. CPU-known
executes 295 eligible chunks per arm. There are no extra growth-time physical
pages, outstanding loans, blocked KV slots or pending CUDA batch copies at
request completion. No runtime default was changed.

After the model runs, the LayerKV suite passed **92 tests**, including the two
new caller-edge regressions:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Focused Black/isort/Ruff checks and `git diff --check` pass. No repository-wide
pre-commit rewrite was performed. Existing dirty LayerKV work was preserved;
no commit was made, and both GPUs were idle after validation.
