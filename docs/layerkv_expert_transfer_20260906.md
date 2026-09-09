# Expert CUDA batch transfer and CPU backing reuse

## Implementation

`--layerkv-expert-transfer-backend cuda-batch` submits independent expert
parameter copies using `cuda.bindings.runtime.cudaMemcpyBatchAsync`. This is
the same CUDA API used in the existing KVC C++ batch implementation, exposed
through the installed bindings rather than a new compiled extension. Expert
weights have heterogeneous parameter sizes and are not forced into K/V-specific
operator layouts. No package installation or compilation is required in the
tested environment. The explicit default remains `torch` for controlled rollout.

The new path covers demand/prefetch H2D materialization, sync demand D2H eviction,
async prefetch D2H eviction, and shared-slot recall's D2H backup. Initial compact
installation backing construction and KVC transfers are unchanged. API errors
are fatal rather than silently falling back after possible partial submission.
The CUDA backend requires matching contiguous tensors, pinned CPU buffers,
one GPU and direction per batch, non-overlapping destinations, and a runtime
exposing the batch API. Unsupported configurations remain on the explicit
`torch` backend, not an unreported fallback.

D2H copies go directly from expert slots into CPU backing, without the old GPU
`index_select` and GPU index-tensor construction. H2D uses one batch API call
for the list of expert/parameter copies rather than Python `copy_` calls for
each one. Descriptor construction and validation are still on the CPU.

## Ordering and lifetime

The batch API has no within-batch ordering guarantee. Input validation rejects
overlapping destinations and mixed directions before submission. Existing
materialization readiness events and demand D2H completion synchronization are
preserved. On a legacy NULL stream, a private CUDA stream waits on the caller,
submits the batch, records an event, and the caller waits on that event. This is
device-side ordering, not an added host synchronization.

Raw-pointer submission does not invoke PyTorch's `copy_` lifetime tracking.
`ExpertBatchTransfer` therefore retains both source and destination tensors
until an event reports completion. Per-host-storage in-flight counts also
defer pool returns: holding a tensor reference alone would not prevent our own
pool from reusing and overwriting it. Event collection happens at subsequent
copies, materialization finalization and stats reads. Blocking collection is
only used where the caller already requests blocking finalization. Copy errors
retain submitted owners and raise, without retrying the batch.

## Separate backing-layout optimization

`--layerkv-expert-batch-backing-layout batch` keeps batched CPU allocation and
per-expert views, isolating transfer changes from pool reuse. The existing
256 MiB per-parameter allocation chunk bound is preserved.

`--layerkv-expert-batch-backing-layout individual` allocates owning pinned tensors
per expert/parameter through the existing pool. These fixed-size owners can
return to that pool after copying completes. It requires `cuda-batch`; the
default layout is `batch`. This avoids refcounting slices of a shared batch
allocation and does not increase the existing free-pool capacity limit. It is
not a new persistent CPU weight cache: resident CPU backings are still trimmed
under the existing backing-cache policy.

New stats distinguish actual submission from logical batching:
`expert_cuda_batch_h2d_count`, `expert_cuda_batch_d2h_count`,
`expert_cuda_batch_copy_count`, `expert_cuda_batch_bytes`,
`expert_cuda_batch_pending`, and `expert_cuda_batch_deferred_release_count`.
Counts only increase after successful batch submission/event recording.

## Bounded experiment

H100 NVL GPU 0, Qwen3.6-35B-A3B local snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, batch=1. Full-model FP16
remains outside this experiment because of the previously observed GDN issue;
the transfer-only GPU tests cover FP16 and BF16. Six sequential requests per
fixed16/grow17 arm, 4096 synthetic input IDs and 64 greedy output tokens.
Input chunk order, donor cache on, detailed chunk and prepare profiling on.
All other memory, KV pressure, graph and scheduling settings match
[prepare attribution](layerkv_prepare_profile_20260906.md).

Three configurations, each in a fresh directory:

| Directory under `/mnt/vdb/chengzhi/` | Transfer backend | Backing layout |
| --- | --- | --- |
| `layerkv_transfer_torch_20260906_01` | torch | batch |
| `layerkv_transfer_batch_20260906_01` | cuda-batch | batch |
| `layerkv_transfer_pool_20260906_01` | cuda-batch | individual |

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_transfer_pool_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --profile-chunks --profile-prepare \
  --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual --execute
```

Use `cuda-batch`/`batch` or `torch`/`batch` in fresh directories for the other
controls. Omit `--execute` for the dry plan. `PATH` is solely for external
framework tools in the reused environment. Experiment options use CLI arguments.
The driver records all effective arguments and checks successful H2D/D2H batch
use and zero outstanding batch copies in addition to its existing correctness,
VMM ownership, pointer stability and profile-count guards.

## Results

All three final summaries report `valid=true`. All 2304 generated output IDs
and token logprobs (three configurations, two arms, six requests, 64 tokens)
match their corresponding reference outputs exactly. Each arm has 2901 profiled
prefill materializations in all configurations: neither expert ordering nor
miss count changed. Grow17 still lends and returns three 2 MiB pages in requests
1 and 3; all requests finish with no outstanding loans, blocked KV locations,
batch copies or profiling events. No growth-time physical pages are created.

Per-request prepare CPU wall, milliseconds:

| Arm | Request | torch | CUDA batch | CUDA batch + individual pool |
| --- | ---: | ---: | ---: | ---: |
| fixed16 | 2 | 2857.315 | 2915.006 | 493.244 |
| fixed16 | 3 | 827.102 | 856.203 | 366.735 |
| fixed16 | 4 | 479.682 | 509.255 | 322.838 |
| fixed16 | 5 | 261.091 | 291.586 | 345.567 |
| fixed16 | 6 | 243.583 | 272.640 | 301.180 |
| grow17 | 2 | 2770.755 | 2799.084 | 492.842 |
| grow17 | 3 | 832.065 | 851.084 | 366.849 |
| grow17 | 4 | 430.753 | 456.689 | 322.982 |
| grow17 | 5 | 261.613 | 291.757 | 346.382 |
| grow17 | 6 | 244.986 | 273.297 | 301.878 |

Request 1 has no chunked-prefill preparation and is excluded above. It is not
excluded from the full-request latency totals below. Reuse does not simply move
the measured cold cost into request 1: fixed16 first-request TPOT is 116.411 ms
with torch and 107.748 ms with pooling; grow17 is 119.513 -> 108.853 ms.

Client TTFT, seconds, torch -> CUDA batch + individual pool:

| Request | fixed16 | grow17 |
| --- | ---: | ---: |
| 1 | 1.914 -> 1.897 | 1.887 -> 1.877 |
| 2 | 3.593 -> 1.207 | 3.506 -> 1.218 |
| 3 | 1.753 -> 1.314 | 1.760 -> 1.336 |
| 4 | 1.639 -> 1.492 | 1.515 -> 1.414 |
| 5 | 1.453 -> 1.494 | 1.325 -> 1.407 |
| 6 | 1.418 -> 1.440 | 1.288 -> 1.350 |

Six-request client elapsed sums, excluding engine startup:

| Arm | torch (s) | CUDA batch (s) | CUDA batch + individual pool (s) |
| --- | ---: | ---: | ---: |
| fixed16 | 40.177 | 40.649 | 36.383 |
| grow17 | 40.255 | 40.425 | 36.581 |

Pooling reduces these instrumented short-run totals by 9.4% and 9.1%, driven
primarily by initial allocation savings. This is **not** a steady-state or
uninstrumented throughput claim. Requests differ in their prompts, the run order
is fixed, and CPU function profiling has configuration-dependent overhead.
Later-request preparation and some TTFT values regress, as explicitly shown.

Final whole-run counters:

| Arm/configuration | Backing allocations | Pool reuse | Pool bytes (MiB) | CUDA H2D/D2H submissions |
| --- | ---: | ---: | ---: | --- |
| fixed16 torch | 1386 | 0 | 12 | 0 / 0 |
| fixed16 CUDA batch | 1386 | 0 | 0 | 677 / 677 |
| fixed16 pooled | 534 | 10150 | 90 | 677 / 677 |
| grow17 torch | 1386 | 4 | 12 | 0 / 0 |
| grow17 CUDA batch | 1390 | 0 | 0 | 677 / 679 |
| grow17 pooled | 534 | 10140 | 90 | 677 / 679 |

These include installation and decode, not only profiled prefill. Pool bytes
mean reusable free buffers, not total mandatory CPU backing or total process
host memory. The existing free-pool limit remains 256 MiB. Pooled fixed16 and
grow17 defer 10180 and 10170 returns respectively until copy completion. CUDA
batch handles 21304 individual parameter copies in fixed16 and 21284 in grow17;
the extra grow17 D2H batches cover slot recall.

## Remaining bottleneck and decision

Batch submission by itself does not improve this trace. In fixed16 request 6,
the CUDA-batch stream synchronize call takes about 56 ms of CPU self time,
versus about 18 ms in the earlier torch attribution; the removed GPU index
construction is not enough to offset transfer/wait and descriptor overhead.
The pooled path adds ownership/pool control and still waits for demand D2H.

Keep both changes explicit, with `torch` as the default. The tested pooled
configuration is useful for studying initial residency churn, not a claim that
all later prefills are faster. The next optimization should replace the demand
D2H host barrier with explicit copy-stream/slot readiness dependencies, or
reduce descriptor/pool bookkeeping, with a matching **uninstrumented** control.
Do not merely delete the barrier: a slot cannot be overwritten until its backup
is safe, and a CPU backing cannot be recycled before H2D completes.

## Validation and limits

The complete LayerKV suite passes 66 tests. Ten transfer tests cover FP16/BF16
round trips on default and explicit streams, overlapping destinations and
pageable-memory rejection, no retry after submission errors, retention after
event-query errors, sync and async runtime D2H paths, and deferred owner-tensor
pool reuse. The final error-path hardening is covered by these tests; no
additional performance claim is made for the error-handling change.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Black, isort and Ruff pass on the new transfer module, tests and modified
benchmark driver. Modified integration modules compile, and `git diff --check`
passes. Full-repository pre-commit is not claimed. Existing dirty work is
preserved; no commit was created. Tested full-model execution is eager BF16 on
one H100 NVL, not graph capture, multi-GPU or full-model FP16 coverage.
