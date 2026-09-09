# CUDA batch submission metadata

## Change and safety boundary

Continue from the prefetch feasibility probe: optimize submission CPU work, not
chunk lookahead or residency policy. In `expert_transfer.py`, read stream device,
destination CUDA status, byte count and destination pointer once per appropriate
scope. Matching contiguous shape/dtype already establishes equal byte counts,
so destination overlap ranges use that validated count. Cache only CUDA copy
attributes, keyed by `(direction, torch.device)`, at most two entries per device.
These attributes contain enums/integers, not backing tensors or pointers.

In `_copy_materialized_experts_batched`, fetch the current parameter's `.data`
once per parameter per submission, rather than once per expert. Each destination
slot view is still built from the current parameter. No view survives in a new
persistent cache; only the existing pending-copy ownership retains destinations
until completion. Resize, recall and backing-pool reuse keep their old semantics.

Pinned checks, shape/dtype/contiguity validation, device/direction checks,
destination-overlap detection, inflight/deferred ownership and fatal submission
handling are unchanged. No pinned-attestation or pointer cache is introduced.
NULL-stream bridging and producer dependencies remain. No new synchronization,
GPU memory, pinned allocation, policy flag or package/build dependency is added.
The equivalent implementation replaces the existing cuda-batch submit path;
the server's transfer-backend default remains `torch`.

The preceding request-6 Kineto trace had about 33.7 ms in the prefill H2D submit
CPU region. Nested operator totals included 2.57 ms `is_pinned` and 1.83 ms
`select`; these are diagnostic, non-additive totals, not potential savings. This
change intentionally retains both operations. A separate 20-call CPU cProfile
in the new microbenchmark attributes most remaining cost to `copy` itself,
`is_pinned` and ownership collection; its timings are not used as speedup data.

## Correctness and isolated performance

`test_transfer_metadata.py` adds ten cases: FP16/BF16 roundtrips with alternating
H2D/D2H attribute reuse, weak-reference proof of no extra tensor retention,
post-warmup replacement with pageable storage, shape/dtype/stride mutation,
partially overlapping destinations, mixed directions, two-device key separation
and wrong-stream rejection, and empty batches. A valid zero-byte input uses an
empty view of pinned storage: a fresh zero-byte PyTorch allocation is not pinned
and must still fail the existing pinned guard.

Full LayerKV suite: **233 passed, 19 warnings**. Existing tests also cover failed
API submission, event inspection failure, deferred host-pool release, stream
ordering, VMM resize/recall, guard checks and materialization. Targeted Black,
isort, Ruff and `git diff --check` pass. No whole-repository pre-commit run.

Run from the worktree `python/` directory:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

`scripts/layerkv_transfer_submit_bench.py` loads the exact saved pre-edit copier
and current implementation in one process. It defaults to a dry plan. Both arms
perform real CUDA H2D and D2H, verify exact values and retirement of pending
owners, and alternate implementation order over four repetitions. Per repetition
and direction: 10 warmups, 100 measurements, 32 prebuilt tensor pairs (16 experts,
two parameters of 4 MiB and 2 MiB), 96 MiB per batch. Each measurement times only
`copy()` CPU wall time; the per-call GPU completion fence and value checks are
outside timing. Parameter view construction is not included. A separate cProfile
phase runs afterward. This is neither transfer bandwidth nor model throughput,
and its one-completed-batch regime does not represent a deep pending queue.

H100 NVL GPU 0; reuse `/mnt/vdb/chengzhi/agent_bench_sglang_venv` without installs.
Raw samples, effective arguments, device, Torch version, source hashes and CPU
profiles are in `/mnt/vdb/chengzhi/layerkv_transfer_submit_20260907/{fp16,bf16}.json`.
The exact pre-edit source is `baseline_expert_transfer.py` in the same directory.
Baseline SHA256: `27ebbbb25a14035697456eb8c4b330b8bc3062ea48bc535cf3e8775758c9f379`;
candidate: `502e59eadbaa84257b8bb30e43e379bb3370060186353de35e7de52a9b53d11e`.

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_transfer_submit_bench.py \
  --baseline-source /mnt/vdb/chengzhi/layerkv_transfer_submit_20260907/baseline_expert_transfer.py \
  --output /mnt/vdb/chengzhi/layerkv_transfer_submit_fresh_fp16.json \
  --gpu 0 --dtype float16 --experts 16 --param-elements 2097152 1048576 \
  --warmup 10 --iterations 100 --repeats 4 --profile-iterations 20 --execute
```

Omit `--execute` first to inspect the plan; use `--dtype bfloat16` and a fresh
output for BF16. No ordinary script configuration uses environment variables.

The values below are medians of four per-repetition medians, in microseconds;
the reduction range is calculated from each matched repetition, not profiles.

| Dtype / direction | Before | After | Per-repeat reduction |
| --- | ---: | ---: | ---: |
| FP16 H2D | 163.732 | 152.882 | 6.48–6.71% |
| BF16 H2D | 164.167 | 154.770 | 5.37–6.18% |
| FP16 D2H | 162.054 | 150.574 | 6.12–7.67% |
| BF16 D2H | 162.352 | 152.229 | 5.64–7.33% |

This establishes a small submission-side improvement without removing guards,
not an end-to-end inference speedup or attribution to any one edit in isolation.

## Model regression scope

A single candidate fixed16/grow17 family uses the existing bounded driver:
Qwen3.6-35B-A3B snapshot `995ad96eacd98c81ed38be0c5b274b04031597b0`, H100 NVL GPU 0,
BF16, six 4096-input/64-output greedy requests per arm, profilers off. Full-model
BF16 preserves the validated GDN setup; FP16 above is a transfer-only result.
Keep remap `batch`, CPU cache 128 MiB, idle pool cap 128 MiB, accounting `scan`,
individual backing, stream-ordered demand D2H, admission validation and CPU-known
input-order preparation. No concurrent GPU tests or microbenchmarks.

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_transfer_submit_fresh_model \
  --gpu 0 --dtype bfloat16 --rounds 6 --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual --expert-demand-d2h-wait stream \
  --prepare-path cpu-known --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 \
  --expert-backing-cache-accounting scan --expert-remap-update batch --execute
```

Dry plan was inspected before launch. `PATH` only exposes external framework
executables. Raw candidate artifacts are under
`/mnt/vdb/chengzhi/layerkv_transfer_submit_20260907_model/`. Comparing this run
against existing remap-batch artifacts can check exact outputs and work/budgets;
it is a historical-reference regression check, not a fresh paired A/B timing
experiment. End-to-end acceleration remains unestablished by this experiment.

Completed result: `summary.json` is valid. All 12 requests / 768 output tokens
and their logprobs match `layerkv_remap_20260907_batch_01` exactly. Plan arguments
differ only in output directory; engine arguments match exactly. Per-request
materialization, H2D/D2H counts/bytes, remap work, pool operations, host budgets
and KV loan work match the historical reference. All requests remain comparable.
The independent check and report are saved as `check_model_regression.py` and
`model_regression.json` under the transfer microbenchmark artifact directory.

| Arm | Six-request E2E sum (s) | Requests 2–6 TTFT sum (s) | Materializations | H2D / D2H batches |
| --- | ---: | ---: | ---: | ---: |
| fixed16 | 33.675767 | 5.646994 | 5326 | 677 / 4 |
| grow17 | 33.223420 | 5.388482 | 5321 | 677 / 4 |

These are candidate-only timings, not paired improvement estimates. Both arms
finish with 1440 MiB mandatory backing + 96 MiB cached backing, zero idle-pool
and pending-copy ownership. Grow17 lends and returns six 2 MiB pages in total;
final outstanding loans, blocked KV tokens and new growth allocations are zero,
with stable expert pointers. No residency budget or admission behavior changed.
