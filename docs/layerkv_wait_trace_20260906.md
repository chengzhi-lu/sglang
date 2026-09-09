# Shared expert wait attribution: H2D tail, not remap kernel cost

## Scope and observation validation

This change adds opt-in diagnostic labels and a read-only trace analyzer. It
does not implement chunk lookahead, alter residency/cache policy, remove the
GPU remap guard, or remove synchronization. Default execution has no profiler
wrappers installed. `scan` remains the default cache-accounting mode.

Following the diagnosis feedback-loop workflow, first check that the profiler
actually observes the installed CUDA batch-copy backend. A real pinned 512 KiB
H2D copy with exact output verification succeeds with both profiling tools,
but installed Nsight Systems 2024.6.2 omits its H2D activity. The SQLite assertion
requiring `CUPTI_ACTIVITY_KIND_MEMCPY.copyKind=1` fails. The D2H copy used to
verify the output is visible; absence of H2D is missing measurement, not zero
transfer time. Retain that failed measurement under:

- `/mnt/vdb/chengzhi/layerkv_wait_probe_20260906.nsys-rep`
- `/mnt/vdb/chengzhi/layerkv_wait_probe_20260906.sqlite`

The existing PyTorch Kineto profiler captures the same real batch H2D (11.841 us
for this probe), without installing packages or rebuilding CUDA code. Its CPU
batch API is named `INVALID` (cbid 509), but the GPU memcpy record has a matching
correlation ID. The probe trace is
`/mnt/vdb/chengzhi/layerkv_wait_probe_kineto_20260906.json`.
This tiny probe validates visibility, not model throughput.

## Instrumentation and analysis boundaries

Server flag `--layerkv-shared-expert-trace-waits` labels prepare, materialize,
H2D submission, cache trim, ready wait, remap, guard reduction and scalar
readback. It requires an installed shared expert layer with `cpu-known`
preparation, and is incompatible with the existing chunk/cProfile diagnostics.
No extra CUDA events, waits or synchronizations are introduced. The actual
`any().item()` guard remains in place, including invalid-remap failure behavior.

Driver `--trace-waits-request N` starts CPU/GPU profiling for just request N
(1-based, zero disables). The preceding requests preserve the same workload and
residency history. CPU stack/shape recording is disabled explicitly. This is
diagnostic timing: profiler overhead changes CPU submission and overlap. Do not
use its E2E summaries, or the accounting comparison's timing percentages, as an
unprofiled speedup measurement. The paired unprofiled accounting result remains
the one documented in `layerkv_cache_accounting_20260906.md`.

`scripts/layerkv_wait_trace_analyze.py` accepts one single-worker, single-device
trace per arm. It uses CPU `user_annotation` ranges only, not Kineto's GPU range
replicas. Device work is assigned by the correlation ID of its CPU launch;
GPU execution need not fall within the launching CPU phase. Multiple device
records per batch correlation are supported. Missing/ambiguous correlations,
missing guard device activities, group/count/byte mismatches fail closed.

For every prefill readback interval, the analyzer partitions elapsed time into
the union of own expert H2D, own remap/guard/scalar-copy work not already covered,
other observed GPU work, and time with no observed GPU activity. These four
categories do not double-count overlap. Nested CPU phase totals must not be
added together. GPU activity overlap is descriptive, not proof of a causal
dependency or time recoverable by prefetch. In particular, a GPU-idle gap is
not automatically CPU computation; runtime submission/wakeup and profiler
overhead remain possible contributors.

## Matched result

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, **BF16**, TP=1, eager, batch=1.
This retains the validated model/GDN setup, not a full-model FP16 result.
FP16 and BF16 both have real-copy regression coverage.

One scan/incremental diagnostic pair, each with fixed16 then KV-funded grow17,
six sequential different 4096-input/64-output greedy requests. Capture only
request 6. No GPU tests or other model runs overlap either capture.
Artifacts:

- `/mnt/vdb/chengzhi/layerkv_wait_trace_20260906_scan/`
- `/mnt/vdb/chengzhi/layerkv_wait_trace_20260906_incremental/`

Each directory includes plan, effective engine args, logs, raw requests,
`fixed16.trace/*.trace.json.gz`, `grow17.trace/*.trace.json.gz`, and
`analysis_v2.json` (final analyzer output; the earlier `analysis.json` is retained).
Matched correctness/transfer/budget checks are in
`/mnt/vdb/chengzhi/layerkv_wait_trace_20260906_comparison.json`.

All four traces contain 55 prefill groups and 537 prefill expert materializations
(3222 MiB). Each complete captured request has 119 expert H2D batches and 989
materializations (5934 MiB): 55 prefill batches and 64 other batches, including
the existing runtime prefetch. No expert eviction D2H occurs in request 6.
Whole-request expert H2D activity is about 108.27 ms; the following table is
**prefill only**, summed over its 55 groups, in milliseconds:

| Measurement | scan fixed16 | incr fixed16 | scan grow17 | incr grow17 |
| --- | ---: | ---: | ---: | ---: |
| CPU chunk prepare, inclusive | 120.217 | 121.905 | 119.548 | 122.014 |
| CPU cache trim, inclusive | 15.329 | 0.990 | 14.847 | 0.933 |
| CPU scalar readback, inclusive | 22.666 | 37.470 | 23.181 | 36.917 |
| Own H2D overlapping readback | 19.673 | 34.340 | 20.258 | 33.893 |
| Own remap/guard/scalar-copy work in readback | 0.745 | 0.779 | 0.727 | 0.780 |
| Other GPU activity in readback | 0 | 0 | 0 | 0 |
| No observed GPU activity in readback | 2.248 | 2.350 | 2.195 | 2.244 |
| Expert H2D device activity, all prefill | 58.758 | 58.757 | 58.756 | 58.756 |
| Remap + reduction + scalar-copy device activity | 0.780 | 0.779 | 0.780 | 0.781 |
| CPU before entry into H2D submission helper | 27.801 | 29.901 | 27.384 | 30.245 |

Incremental trim saves about 14 ms inside these profiled calls, while the
readback's exposed H2D grows by approximately the same amount. About 92% of
incremental readback time overlaps its own H2D. This directly supports the
previous overlap explanation; it is not evidence that the remap kernel got
slower. The actual remap kernels take only about 0.275 ms over all 55 groups.

H2D is already starting around batch API return: the sum of positive
API-return-to-first-device-start gaps is at most 0.003 ms in each arm, and the
largest individual gap is below 0.001 ms. No previously submitted GPU activity
overlaps the measured prepare or readback windows. Thus these captures do not
support earlier GPU compute delaying the copies as the dominant remaining
wait. This does not establish behavior outside this workload or under a lighter
profiler. Trace timestamps near API boundaries should not be read as precise
sub-microsecond launch-overhead measurements.

Before bulk expert H2D, each arm also has **1074 small H2D copies totaling only
8592 bytes and 1074 `cudaStreamSynchronize` calls**. The sync API durations total
6.31–6.38 ms. In `_materialize_experts`, the two scalar GPU remap assignments per
replacement (evict old ID, install new ID) explain this pattern: 2 x 537.
The 27–30 ms pre-submission CPU span includes more than these API calls and is
not an estimated recoverable saving.

## Next optimization decision

Prioritize a bounded experiment batching the scalar remap-table updates before
bulk H2D, retaining the final GPU remap validation and the same materialization,
LRU and slot choices. These updates delay H2D submission itself; unlike cache
trim, they do not run after that copy has started. First test exact mappings,
guard failures, repeated replacements and async source/slot lifetimes, then
compare unprofiled end-to-end runs. No such runtime change is included here.

Chunk lookahead remains a subsequent candidate, but H2D dominance alone does
not make it safe: a future copy must not overwrite slots still read by the
current chunk. It needs an explicit reusable-slot/lifetime boundary and bounded
budget, not simply earlier calls to `_materialize_experts` or removal of `.item()`.
Existing scheduler prefetch is unchanged by this diagnostic.

## Reproduction and validation

From `/home/chengzhi/github/sglang-perf-layerkv`:

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_wait_trace_fresh_scan \
  --gpu 0 --dtype bfloat16 --rounds 6 --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual --expert-demand-d2h-wait stream \
  --prepare-path cpu-known --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 \
  --expert-backing-cache-accounting scan --trace-waits-request 6 --execute

/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_wait_trace_analyze.py \
  --run-dir /mnt/vdb/chengzhi/layerkv_wait_trace_20260906_scan \
  --expert-bytes 6291456 --output /mnt/vdb/chengzhi/layerkv_wait_trace_fresh_analysis.json
```

Use `incremental` and a fresh directory for the second arm family; omit
`--execute` to inspect the plan first. `PATH` only exposes external framework
executables from the reused virtualenv; ordinary experiment configuration uses
explicit CLI arguments. Context/KV/scratch/block limits remain 8192/16384/4095/2048,
reclaim target 40 MiB, shared expert layer 0, initial slots 16, extra slots 0/1.
Resident CPU cache and idle pool caps remain 128 MiB each.

All 24 requests pass runtime guards. Output token IDs and logprobs match exactly
across accounting modes, and all 24 also match their corresponding previous
unprofiled runs. Materializations, H2D/D2H, pool operations, host storage and KV
loan work match across modes. Final tracked host storage is 1440 MiB mandatory
+ 96 MiB cached, zero idle/batch-owner storage. KVC stale/alignment counts are
zero; KVC/expert guard and comparability flags pass. Incremental ledgers match
the independent scan. These are scoped correctness checks, not general model
quality evaluation.

Run the full LayerKV suite from the worktree's `python/` directory:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

The final full LayerKV suite passes **173 tests** (19 warnings). The trace tests
exercise real FP16/BF16 batch transfers, exact traced/untraced
weights and maps, and invalid-remap guards. CPU-only analyzer tests cover late
device execution, multi-record batch correlations, overlap unions, missing and
ambiguous measurements, and preservation of measured zero categories. Targeted
Black/isort/F821 and `git diff --check` pass. Full-repository pre-commit is not
run over the large pre-existing dirty worktree; no commit is made.
