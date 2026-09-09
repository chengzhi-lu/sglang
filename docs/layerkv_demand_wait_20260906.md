# Stream-ordered demand D2H completion

## Change and scope

This follows [CUDA batch transfer and backing reuse](layerkv_expert_transfer_20260906.md).
The explicit option `--layerkv-expert-demand-d2h-wait stream` replaces the early
demand D2H **host** barrier with a GPU stream dependency. It requires the
`cuda-batch` transfer backend; default `host` preserves existing behavior.
The benchmark driver exposes the same choice as `--expert-demand-d2h-wait`.
No package installation, compilation, memory-budget, chunk-order or expert
selection change is involved. Pool management itself is unchanged in this step.

Only `_materialize_experts(..., reason="on_demand")` asks its backup helper to
return without waiting on the CPU. D2H remains ordered after prior work on the
caller stream. Before switching to the H2D stream, materialization records a
dependency on the caller stream, including that D2H. Slot overwrite therefore
cannot precede its backup. Before MoE consumes new weights, the original H2D
ready-event dependency is preserved. The legacy NULL-stream bridge still obeys
the same ordering. Same-stream H2D needs no additional wait event.

CPU backing returned by the deferred helper is **not yet safe for arbitrary
CPU reads**. The only internal caller uses it through the ordered materializer;
raw-copy owners and in-flight host-pool returns are still protected by
`ExpertBatchTransfer`. Shared-slot recall and ordinary callers retain the
helper's default blocking behavior. Prefetch/installation paths are unchanged.
This removes one early host barrier, not all synchronization: the later remap
guard's scalar readback can still wait for GPU work. It permits overlap of CPU
submission/bookkeeping with backup; it is not speculative expert prefetch.

New counters are `expert_demand_d2h_stream_batch_count` and
`expert_demand_h2d_dependency_count`. The benchmark requires both to be exercised
in stream mode, alongside existing batch completion, output and ownership guards.

## Regression feedback

The actual materialization call-chain regression was run before the fix:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/test_expert_transfer.py \
  -k demand_materialization -q
```

Both cases failed with `AssertionError: demand D2H blocked the host` at
`_materialize_experts -> _copy_slots_to_cpu_batched -> Stream.synchronize`.
After the change both pass. They cover default and non-default caller streams,
a distinct H2D stream, prior GPU writes observed by backup, new slot contents,
and rematerializing the evicted experts with exact values. The test forbids the
early host barrier and asserts the H2D stream actually waits on the correct
producer stream. A separate regression checks recall-style backup remains
host-ready even when the runtime's demand setting is `stream`.

## Bounded unprofiled comparison

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, batch=1. Six sequential
4096-input/64-output greedy requests in each fixed16/grow17 arm. All prior KV
pressure and residency parameters are retained. Both chunk profiling and prepare
cProfile are **off**; zero phase timers mean unmeasured, not zero work.
Client streaming, counters and correctness validation remain enabled.

Directories under `/mnt/vdb/chengzhi/`:

| Directory | Backend | Backing | Demand wait |
| --- | --- | --- | --- |
| `layerkv_wait_native_20260906_01` | torch | batch | host/default |
| `layerkv_wait_host_20260906_01` | cuda-batch | individual | host |
| `layerkv_wait_stream_20260906_01` | cuda-batch | individual | stream |

The native run began before the new CLI option was added; its default torch
path is unaffected. The two CUDA runs explicitly record the wait choice and
form the direct host-vs-stream comparison. Different requests use different
prompts. An ordered six-request sample is not a repeated steady-state workload.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_wait_stream_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --execute
```

Use `--expert-demand-d2h-wait host` in a fresh directory for the immediate
control. For native, use `--expert-transfer-backend torch`,
`--expert-batch-backing-layout batch`, and `--expert-demand-d2h-wait host`.
Omit `--execute` for a dry plan. `PATH` only exposes framework tools from the
existing environment; runtime settings use CLI arguments.

## Results and interpretation

All three summaries report `valid=true`. Corresponding generated IDs and token
logprobs match **exactly** across all configurations and both arms (2304 output
tokens in total). Each arm retains the same 2901 prefill materializations.
Whole-run materializations remain 5326 in fixed16 and 5321 in grow17; pooling
allocations/reuse also remain unchanged between host and stream modes.

Both stream-mode arms execute 672 deferred demand D2H batches and 672 H2D stream
dependencies. Total CUDA H2D/D2H submissions remain 677/677 in fixed16 and
677/679 in grow17; non-demand paths are intentionally not converted. Grow17
still lends and returns three 2 MiB KV pages in requests 1 and 3. Every request
finishes with no outstanding loans, blocked KV locations or CUDA batch copies,
and no growth-time physical page creation. Guard checks pass.

Client TTFT, seconds:

| Arm | Request | Native torch | Pooled, host wait | Pooled, stream wait |
| --- | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 1.924 | 1.917 | 1.921 |
| fixed16 | 2 | 3.497 | 1.099 | 1.075 |
| fixed16 | 3 | 1.695 | 1.200 | 1.156 |
| fixed16 | 4 | 1.571 | 1.428 | 1.389 |
| fixed16 | 5 | 1.386 | 1.419 | 1.399 |
| fixed16 | 6 | 1.353 | 1.367 | 1.389 |
| grow17 | 1 | 1.920 | 1.879 | 1.927 |
| grow17 | 2 | 3.396 | 1.113 | 1.088 |
| grow17 | 3 | 1.720 | 1.217 | 1.202 |
| grow17 | 4 | 1.441 | 1.352 | 1.282 |
| grow17 | 5 | 1.276 | 1.326 | 1.295 |
| grow17 | 6 | 1.269 | 1.259 | 1.252 |

Streaming TPOT, milliseconds:

| Arm | Request | Native torch | Pooled, host wait | Pooled, stream wait |
| --- | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 115.963 | 107.677 | 106.949 |
| fixed16 | 2 | 67.479 | 66.554 | 65.868 |
| fixed16 | 3 | 68.713 | 65.304 | 65.024 |
| fixed16 | 4 | 65.392 | 66.412 | 65.918 |
| fixed16 | 5 | 65.374 | 66.648 | 66.037 |
| fixed16 | 6 | 65.538 | 66.465 | 66.035 |
| grow17 | 1 | 117.607 | 108.857 | 108.243 |
| grow17 | 2 | 67.817 | 66.768 | 66.774 |
| grow17 | 3 | 68.569 | 65.422 | 65.462 |
| grow17 | 4 | 66.269 | 66.825 | 66.573 |
| grow17 | 5 | 66.265 | 66.699 | 66.927 |
| grow17 | 6 | 66.232 | 66.863 | 66.909 |

Six-request client elapsed sums, excluding engine startup:

| Arm | Native torch (s) | Pooled, host (s) | Pooled, stream (s) |
| --- | ---: | ---: | ---: |
| fixed16 | 39.680 | 36.092 | 35.787 |
| grow17 | 39.546 | 35.957 | 35.823 |

The incremental stream-wait change reduces these totals by only **0.8%/0.4%**.
Most of the larger difference versus torch comes from the **previous** backing
reuse change and its initial allocation savings, not this change. Later-request
regressions versus native remain in some cells. Even the fixed16 requests-2--6
TTFT median rises from 1.367 to 1.389 seconds despite several paired requests
improving, illustrating why these different prompts should not be treated as
identical steady-state repetitions.

Decision: retain default `torch`/`host`. Stream mode is a correctly ordered,
explicit experimental option, not an established stable-throughput advantage.
The early host wait was confirmed and eliminated, but its presence alone did
not prove it dominated end-to-end time. GPU copy dependencies, later scalar
readbacks and pool/control work remain. Future work should isolate those costs
instead of deleting readiness checks or claiming the whole previous pooling
gain for stream ordering.

## Validation notes

The final GPU-idle serial full suite passes **69 tests**:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

An earlier suite run during benchmark activity had one failure in the existing
`test_cuda_reload_after_slots_overwritten[dtype1-False]` KVC test. Its isolated
four-case rerun and the final full serial rerun both pass. The cause of that
intermittent failure is not established; no KVC code/test was changed or marked
fixed to hide it. New demand tests and the recall-style blocking regression pass.

Black/isort/Ruff pass for the touched driver and test file. Modified integration
modules compile; `git diff --check` passes. Full-repository pre-commit is not
claimed. Existing dirty work is preserved, no packages installed and no commit
created. Full-model coverage here is eager BF16 on one GPU, not FP16, graph
capture or multi-GPU validation.
