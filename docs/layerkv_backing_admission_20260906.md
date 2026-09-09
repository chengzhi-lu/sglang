# Validate deferred backing at pool admission

## Change and invariant

This implements the narrow CPU duplication identified by the
[caller attribution](layerkv_prepare_callers_20260906.md). Add server option
`--layerkv-expert-backing-release-validation eager|admission`; the performance
driver exposes `--expert-backing-release-validation`. Default `eager` preserves
the original validation order. `admission` requires `cuda-batch` transfers.
No ordinary runtime settings use environment variables.

In admission mode, a non-CPU tensor or a direct PyTorch view (`_base` present)
is still rejected immediately. This keeps normal batched backing views out of
the deferred queue. Otherwise, a tensor whose storage is still owned by a
pending/failed transfer is deferred **before** full storage/pinned checks.
Only when no copy still owns that storage does the release path perform the
existing full validation and attempt pool insertion. The completion callback
still goes through the release path; it is not a bypass into the pool.

Full validation reads the **current** tensor metadata and pool capacity.
Detached partial views cannot enter the pool even though `_base` is absent.
Ownership, storage offset/size, pinned-memory and pool-capacity checks remain.
No cached validity decision is carried across the asynchronous boundary.
Rejected detached views can be dropped later than in eager mode, but can never
be reused as owning buffers. Non-deferred tensors are validated immediately.

Copy events, owner references, multi-transfer storage reference counts,
failed-transfer fail-closed retention, pool limits, H2D/D2H ordering, final
remap guards and KV/expert residency policies are unchanged. This removes one
duplicate validation for a deferred eligible tensor; it does not remove all
release bookkeeping or the required copy work. The prior callback's entire
14 ms CPU interval is not a predicted saving.

## Regression loop

Before implementation, the following real-copy test gave two failures and two
passes: both FP16/BF16 admission cases observed `2 == 1` ownership validations,
while eager controls correctly observed two.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/test_backing_admission.py \
  -k validated_once -q
```

The regression submits an actual CUDA batch H2D copy, releases its CPU backing
while the completion event is uncollected, then retires owners and checks exact
copied data and subsequent pointer reuse. It never depends on a short DMA still
being physically in progress: uncollected ownership itself prevents reuse.
Admission now performs one full validation, eager still performs two.

The 15 new cases also cover metadata/offset/pinned-status/capacity changes before
pool admission, direct and detached partial views, two real transfers sharing
one storage, CUDA submission failure, event-inspection failure, and immediate
pinned/pageable releases. Metadata-mutation tests wait for DMA completion
**before** mutation but keep the event uncollected; arbitrary mutation during
active DMA is not a supported contract. The multiple-transfer test conservatively
delays one event's inspection so the other completion cannot admit storage early.

The full LayerKV suite passes 107 tests, including existing batch/individual
backing correctness paths. No packages were installed or CUDA extensions built.

## Matched unprofiled experiment

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1.
This retains the previously validated full-model GDN configuration; FP16 unit
coverage is not a full-model FP16 claim. Each invocation runs fixed16 then
grow17, with six sequential 4096-input/64-output greedy requests per arm.
Corresponding request inputs are identical between configurations/repetitions;
different request indices are different prompts and cache trajectories.

Two repetitions reverse configuration order: eager_01, admission_01,
admission_02, eager_02. Directories under `/mnt/vdb/chengzhi/` are prefixed
`layerkv_release_20260906_` with these four suffixes. Each records `plan.json`,
effective per-arm engine arguments, logs, raw request JSON and summary.
Both chunk and prepare profiling are disabled. Their zero timers mean
unmeasured, not zero work. GPU tests are not concurrent with model timing.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_release_admission_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known \
  --expert-backing-release-validation admission --execute
```

Use `--expert-backing-release-validation eager` in a fresh directory for the
control; omit `--execute` for a dry plan. `PATH` only exposes external framework
tools from the existing venv. Recorded defaults retain context 8192, KV capacity
16384, scratch 4095, block 2048, reclaim target 40 MiB, shared layer 0 and
initial expert slots 16, with extra slots 0/1 for fixed/grow. This is a backing
validation comparison, not request-admission or expert-selection tuning.

## Results

All four summaries report `valid=true`. Corresponding output token IDs and
logprobs match **exactly across all 48 requests** (3072 generated tokens),
between configurations, repetitions and fixed/grow arms. KVC/expert/ownership
guards and comparability pass. The effective engine arguments differ only in
`layerkv_expert_backing_release_validation` for each matched pair.

Per-request chunk counts, prefill materializations, whole materializations,
H2D/D2H batch counts, deferred release counts, KV loan/return counts, and all
backing-pool counters match the control. Each arm retains 295 CPU-known chunks
and 2901 prefill materializations. No extra growth-time physical pages are
created; every request returns to 16 expert slots with zero outstanding loans,
blocked KV locations or pending CUDA batch transfers. The experiment does not
obtain its result from a larger GPU/host budget or extra immutable expert cache.

Final cumulative counts, identical between modes/repetitions:

| Metric | fixed16 | grow17 |
| --- | ---: | ---: |
| Whole materializations | 5326 | 5321 |
| CUDA H2D / D2H batches | 677 / 677 | 677 / 679 |
| Deferred releases | 10180 | 10170 |
| Pool allocations | 534 | 534 |
| Pool reuses | 10150 | 10140 |
| Pool releases | 10180 | 10170 |
| Pool drops | 472 | 472 |
| Final pool size / limit (MiB) | 90 / 256 | 90 / 256 |

Client TTFT in seconds; E = eager validation, A = admission-time validation:

| Arm | Request | E repetition 1 | A repetition 1 | E repetition 2 | A repetition 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 1.8918 | 1.9304 | 1.8905 | 1.8763 |
| fixed16 | 2 | 1.0615 | 1.0520 | 1.0518 | 1.0528 |
| fixed16 | 3 | 1.1479 | 1.1452 | 1.1627 | 1.1564 |
| fixed16 | 4 | 1.3628 | 1.3740 | 1.3566 | 1.3583 |
| fixed16 | 5 | 1.3660 | 1.3685 | 1.3887 | 1.3786 |
| fixed16 | 6 | 1.3215 | 1.3346 | 1.3371 | 1.3534 |
| grow17 | 1 | 1.9268 | 1.8840 | 1.8789 | 1.9266 |
| grow17 | 2 | 1.0635 | 1.0648 | 1.0730 | 1.0720 |
| grow17 | 3 | 1.1771 | 1.1827 | 1.1825 | 1.1866 |
| grow17 | 4 | 1.2819 | 1.2899 | 1.2556 | 1.2526 |
| grow17 | 5 | 1.2875 | 1.2724 | 1.2605 | 1.2821 |
| grow17 | 6 | 1.2361 | 1.2319 | 1.2369 | 1.2641 |

E2E below is the sum of six client request durations, excluding engine startup.
Positive improvement means less time. Means over requests 2–6 describe this
matched sequence, not five identical steady-state samples.

| Arm | Repetition | Mean TTFT requests 2–6, E → A (s) | TTFT improvement | Six-request E2E, E → A (s) | E2E improvement |
| --- | ---: | --- | ---: | --- | ---: |
| fixed16 | 1 | 1.2519 → 1.2549 | -0.23% | 35.9029 → 35.6970 | 0.57% |
| fixed16 | 2 | 1.2594 → 1.2599 | -0.04% | 35.8853 → 35.8001 | 0.24% |
| grow17 | 1 | 1.2092 → 1.2083 | 0.07% | 35.6510 → 35.4509 | 0.56% |
| grow17 | 2 | 1.2017 → 1.2115 | -0.81% | 35.7009 → 35.6150 | 0.24% |

Mean late-decode TPOT over requests 2–6 (`tail_tpot_ms`, not a percentile):
fixed16 is 65.810 → 65.232 ms and 65.686 → 65.142 ms; grow17 is
65.294 → 64.741 ms and 65.649 → 65.312 ms in repetitions 1/2. The backing
release helper is used during decode as well as prefill, unlike the earlier
CPU-known routing optimization. Server prefill-forward timestamps remain null;
client TTFT is not a pure GPU kernel measurement.

Conclusion: the real-copy regression proves elimination of the duplicate full
validation, and the two matched repetitions show a small 0.24–0.57% reduction
in total request time. **No consistent TTFT improvement is demonstrated.** The
timing gain is small and two repetitions are not broad statistical evidence;
keep `eager` as default and `admission` opt-in. No new profiled model run was
used to claim these unprofiled gains or an exact amount of saved CPU time.

## Final validation

After the model runs, all **107 LayerKV tests pass**, including the server-args
to-runtime configuration path in the new regression fixture:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

The driver's incompatible `admission` + `torch` CLI combination was checked to
fail with exit 2 before starting an engine or creating an output directory.
Focused Black/isort/Ruff checks and `git diff --check` pass. No repository-wide
pre-commit rewrite was applied to the dirty tree. Both GPUs were idle with zero
process memory after validation; earlier LayerKV work is preserved and no
commit was made.
