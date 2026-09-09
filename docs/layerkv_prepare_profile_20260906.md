# Chunk prepare CPU attribution

## Scope and instrumentation

This follows [the group-order experiment](layerkv_chunk_order_20260906.md).
It instruments preparation only; it does not change backing allocation,
materialization, remapping, donor admission, kernels or the default `input`
order. Enable `--profile-chunks --profile-prepare` in
`scripts/layerkv_shared_perf.py`, or the corresponding server flags
`--layerkv-shared-expert-profile-chunks` and
`--layerkv-shared-expert-profile-prepare`. Both are disabled by default.
Prepare profiling requires chunk profiling, which requires a shared expert layer.

Each prepare invocation uses a fresh CPU cProfile instance and merges function
counters after disabling it. No CUDA synchronization is added. The existing
outer chunk prepare wall interval also includes profile setup/aggregation;
`prepare_profile.*.wall_ms` excludes counter aggregation. These diagnostic runs
are not uninstrumented latency benchmarks. Function `total_ms` is inclusive;
`self_ms` excludes callees. Nested totals must not be added together, and host
H2D submission time is not device transfer duration.

`prepare_profile` in `final_stats.shared_vmm` contains cumulative function
tables and shape records. `first_shape` is the first invocation of a signature
(top-k shape, dtype, device, slot capacity); `repeat_shape` contains later
invocations of that signature. This is not a JIT-cache key or an allocator
size-class key. In particular, first shape does not mean a compilation occurred,
and repeated shape does not imply all its allocation sizes are warm.

The driver checks explicit invocation counts against chunk counts and root
primitive calls (`calls - recursive_calls`). The initial diagnostic directory
`/mnt/vdb/chengzhi/layerkv_prepare_input_20260906_01` used persistent profilers
and exhibited inconsistent root accounting; retain it for diagnosis, but do not
use it for the final attribution. Fresh per-invocation profiles avoid the
cross-invocation accounting error; raw recursive-call counters are retained.
The input rerun's original total-call-only verdict is preserved as
`summary.total-call-check.json`; `summary.json` was regenerated using primitive
root calls, without rerunning inference or changing any model/ownership guards.

## Reproduction

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot below, BF16, TP=1, batch=1.
BF16 is deliberate: the previously observed native GDN FP16 issue is outside
this experiment. Fixed16 then KV-funded grow17, six sequential requests per arm,
4096 input IDs and 64 greedy output tokens. All other settings match the linked
group-order experiment. Request 1 precedes compact prefill and has zero measured
chunk-prepare calls. Different requests have different prompts, not repeated
identical workloads.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_prepare_input_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --profile-chunks --profile-prepare --execute
```

Use `--chunk-order reuse` and a fresh directory for the second order. Omit
`--execute` for a dry plan. `PATH` supplies external framework tools such as
Ninja from the existing environment; ordinary options use explicit CLI arguments.

## Mechanism found in the code

`ExpertBackingMixin._copy_slots_to_cpu_batched()` allocates a pinned CPU tensor
for each parameter batch and saves `dst[offset]` views as individual expert
backing. `_release_cpu_backing_tensor()` rejects non-owning views, correctly
preventing unsafe reuse while another expert can still reference the storage.
The zero optional backing-cache budget also drops CPU copies for newly resident
experts, so later eviction needs a fresh D2H backup.

Thus the LayerKV pool cannot reuse those batched backing tensors. This does not
mean every `torch.empty(pin_memory=True)` reaches the driver: the underlying
framework may still cache host allocations. Late-request allocation latency
must be measured separately from initial allocator growth. No allocator-policy
change is implemented in this turn.

## Input-order measurements

Artifacts: `/mnt/vdb/chengzhi/layerkv_prepare_input_20260906_02/`.
Values below are per-request cumulative-counter differences, in milliseconds.
Prepare is the outer wall interval; backing allocation, remap and H2D helpers
are inclusive CPU attribution. Stream wait includes only the Python CUDA
stream `synchronize` helper, not all implicit device waits.

| Arm | Request | Prepare wall | CPU backing allocation | Remap | H2D helper | Stream wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 2 | 2873.339 | 2502.051 | 16.900 | 47.292 | 30.285 |
| fixed16 | 3 | 831.073 | 542.268 | 14.619 | 41.104 | 23.341 |
| fixed16 | 4 | 481.692 | 228.014 | 13.967 | 35.904 | 18.297 |
| fixed16 | 5 | 263.278 | 1.875 | 13.682 | 36.561 | 19.645 |
| fixed16 | 6 | 243.666 | 1.733 | 13.254 | 33.938 | 18.250 |
| grow17 | 2 | 2729.031 | 2402.545 | 17.177 | 46.232 | 26.295 |
| grow17 | 3 | 819.704 | 533.158 | 14.811 | 40.275 | 22.416 |
| grow17 | 4 | 432.396 | 181.741 | 13.785 | 34.916 | 18.434 |
| grow17 | 5 | 260.993 | 1.883 | 13.936 | 35.901 | 19.618 |
| grow17 | 6 | 242.800 | 1.780 | 13.153 | 33.624 | 18.293 |

Over all five measured prefills, backing allocation consumes 3275.940 of
4549.899 ms of inner prepare wall in fixed16 (72.0%), and 3121.107 of
4345.349 ms in grow17 (71.8%). Almost all allocation-helper time is inside
`torch.empty`; this does not establish the precise driver/host allocator cause.
No remap compilation or JIT-loader invocation appears inside these profiles.
It may already have warmed during unprofiled initialization/decode.

Shape buckets across requests 2--6:

| Arm | Bucket | Calls | Inner prepare wall (ms) | Backing allocation (ms) |
| --- | --- | ---: | ---: | ---: |
| fixed16 | first shape | 134 | 2806.305 | 2211.716 |
| fixed16 | repeated shape | 161 | 1743.594 | 1064.224 |
| grow17 | first shape | 134 | 2684.856 | 2132.751 |
| grow17 | repeated shape | 161 | 1660.494 | 988.356 |

Substantial backing allocation time occurs even for repeated routing shapes.
The final fixed16 whole-run LayerKV backing-pool counters are 1386 allocations,
zero reuse, four releases and 10648 drops; grow17 has 1386 allocations, four
reuses, eight releases and 10634 drops. These include decode and installation,
not just profiled prefill. They corroborate ineffective reuse at this LayerKV
pool, not absence of caching in PyTorch's underlying host allocator.

By request 6, allocation is no longer dominant. Fixed16 self-time is 43.005 ms
in `torch.tensor`, 29.162 ms in Tensor `item`, 18.054 ms in the underlying stream
synchronize call and 17.899 ms in materialization control. These are CPU call
intervals, including implicit waits where applicable, not GPU kernel timings.
The Python stream synchronize wrapper's inclusive 18.250 ms includes the
18.054 ms underlying call; do not add those two.

## Reuse-order measurements and decision

Artifacts: `/mnt/vdb/chengzhi/layerkv_prepare_reuse_20260906_01/`.

| Arm | Request | Prepare wall | CPU backing allocation | Remap | H2D helper | Stream wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 2 | 1820.679 | 1541.320 | 15.769 | 38.994 | 22.546 |
| fixed16 | 3 | 966.099 | 720.016 | 13.987 | 34.914 | 18.235 |
| fixed16 | 4 | 873.033 | 643.039 | 13.122 | 31.339 | 17.863 |
| fixed16 | 5 | 209.551 | 1.937 | 12.612 | 28.880 | 12.721 |
| fixed16 | 6 | 192.856 | 1.733 | 11.832 | 26.913 | 11.686 |
| grow17 | 2 | 1706.196 | 1427.733 | 15.556 | 38.489 | 21.799 |
| grow17 | 3 | 960.861 | 716.204 | 13.918 | 34.915 | 18.346 |
| grow17 | 4 | 672.537 | 444.070 | 12.860 | 30.306 | 17.118 |
| grow17 | 5 | 207.364 | 1.818 | 12.483 | 27.848 | 12.695 |
| grow17 | 6 | 192.020 | 1.708 | 11.862 | 26.754 | 11.691 |

The prior request-3/4 prepare regression reproduces. Fixed16 request 4 rises by
391.341 ms with reuse ordering; its backing allocation rises by 415.025 ms,
while remap, H2D helper and explicit stream wait decrease. Grow17 request 4
similarly rises by 240.141 ms, with 262.329 ms more backing allocation.
Allocation call latency, not remap compilation, accounts for the regression
in these measured traces. The precise host-allocator/cache mechanism behind
the order-sensitive allocation latency remains an inference, not a driver trace.

Keep the default `input` order. The next bounded implementation candidate is
ownership-aware reuse of batched pinned backing, with an explicit host-memory
budget. A batch must not return to the pool until all expert views and all
in-flight copies release it; returning `dst[offset]` independently is unsafe.
This targets initial allocation spikes, not a claimed steady-state speedup.
Preserving extra immutable CPU expert copies is an alternative, but consumes a
different host-memory budget and should not be silently enabled.

For later requests where allocation is already cheap, separately target small
GPU index-tensor construction and scalar readbacks in materialization/control.
Do not remove existing synchronization without replacing its copy-lifetime
and readiness guarantees. Compare an unprofiled control after any optimization;
these ordered, instrumented traces alone do not establish stable end-to-end gain.

## Validation

Both final summaries report `valid=true`: all six paired requests per order
pass KV/expert guards, pointer/ownership checks, loan return and profile-count
checks. No growth-time physical page creation occurs. Across fixed/grow and
across the two orders, output IDs and token logprobs match exactly. All 1536
outputs in these four arms also match the corresponding earlier chunk-only
profiled runs, including logprobs.

The LayerKV suite passes **56 tests**, including shape bucketing, exception
cleanup, repeated profiler invocation accounting, immutable summary snapshots
and rejection of an already active thread profiler:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Black, isort and Ruff pass on the four focused profiling/controller/driver/test
files; `git diff --check` passes. Full-repository pre-commit is not a claimed
validation. Existing unrelated dirty changes were preserved; nothing committed.
