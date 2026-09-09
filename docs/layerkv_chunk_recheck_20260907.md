# Input versus reuse ordering on the optimized transfer baseline

## Scope and controls

Recheck the existing `--chunk-order reuse` policy after CPU backing cache,
stream-ordered demand transfers, CPU-known preparation, remap batching and CUDA
batch submission metadata optimizations. This experiment changes no runtime
code or defaults. The exact first-fit token-group membership, whole-token native
MoE reduction, physical slot budget and KV/expert admission mechanisms remain.
Only group execution order differs between matched launches.

Reuse chooses the remaining group with the fewest missing experts against live
residency. Its changed final resident set can affect decode and later requests.
Consequently, total materializations, loans actually used and backing usage are
reported, not required to match across order policies; budget limits, outputs
and correctness guards must match/pass. Fewer prefill loads alone do not prove
better end-to-end latency.

H100 NVL GPU 0; local Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, **BF16**, TP=1, eager, batch=1.
This is the validated full-model/GDN setup, not a full-model FP16 experiment.
Each family launches fixed16 then KV-funded grow17, each with six different
4096-input/64-output greedy requests and per-token streaming. Shared expert
layer 0, initial slots 16, extra slots 0/1, KV tokens 16384, context 8192,
scratch 4095, block 2048, reclaim target 40 MiB. Other expert layers remain on
their existing path; this is not a multi-layer or production-trace benchmark.

Keep `cuda-batch`, individual backing, stream-ordered demand D2H, admission-time
backing validation, CPU-known preparation, remap `batch`, cache accounting `scan`,
CPU cache 128 MiB and idle pool cap 128 MiB. Chunk/prepare profilers, wait tracing
and cache debug checks are all disabled. No concurrent GPU tests or experiments.
Reuse the existing virtualenv; no package installation or extension compilation.

## Execution and artifact validation

Four fresh families, in order:

1. `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_input_01`
2. `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_reuse_01`
3. `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_reuse_02`
4. `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_input_02`

Both dry plans per repetition were inspected before execution. All four plans
and a pre-execution SHA256 manifest of LayerKV runtime and launch-driver source
are saved in `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907/`. Runtime source
is held fixed throughout the experiment; new CPU-only comparison code does not
participate in engine execution. Each family retains its effective engine args,
plan, logs, raw streaming responses/statistics and guard summary.

From `/home/chengzhi/github/sglang-perf-layerkv`, reproduce one family:

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_chunk_recheck_fresh_input \
  --gpu 0 --dtype bfloat16 --rounds 6 --input-tokens 4096 --output-tokens 64 \
  --expert-transfer-backend cuda-batch --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known \
  --expert-backing-release-validation admission --expert-host-extra-budget-mb 256 \
  --expert-backing-cache-mb 128 --expert-backing-cache-accounting scan \
  --expert-remap-update batch --chunk-order input --execute
```

Use `--chunk-order reuse` with a fresh directory for its mate, then reverse
family order in the second repetition. Omit `--execute` for the dry plan.
`PATH` only exposes external framework tools; ordinary configuration is explicit
CLI metadata. The first request precedes compact installation, so its zero
chunk count does not mean zero prefill work. Later requests use distinct prompts
and are not assumed interchangeable steady-state samples.

`scripts/layerkv_chunk_compare.py` is CPU-only. It requires exact matched plans
and engine settings apart from order/output location, profilers/debug checks
off, complete request sets, finite measured values, exact token IDs/logprobs,
unchanged group counts, actual reuse execution, comparable/KVC/expert/ownership
and host-budget guards, stable pointers and zero pending transfers/loans.

The report preserves each request and splits first versus later requests. Tail
TPOT is the average interval from output tokens 16 to 64, not p95. Null server
prefill-forward timestamps remain unavailable. `non_chunk_materializations` is
total minus compact-prefill materializations: it includes scheduler prefetch and
must not be relabeled a pure decode-demand miss count. Transfer MB comes from
the runtime's materialization counter (MiB units). No profiler timing is used.

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_chunk_compare.py \
  --pair /mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_input_01 /mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_reuse_01 \
  --pair /mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_input_02 /mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907_reuse_02 \
  --output /mnt/vdb/chengzhi/layerkv_chunk_recheck_fresh_comparison.json
```

Fifteen deterministic CPU-only comparison tests cover changed work accounting,
unmatched budgets/engine settings, profiled runs, output/logprob changes,
nonfinite or missing metrics, inconsistent counts/groups, disabled reuse and
failed comparability, host-budget or pending-transfer guards:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  test/registered/unit/layerkv/test_chunk_compare.py -q
```

## Results and decision

All four families completed, each with `summary.json: valid=true`. The final
two-pair report is `/mnt/vdb/chengzhi/layerkv_chunk_recheck_20260907/comparison.json`;
the first-pair report is retained as `comparison_01.json`. All 48 requests / 3072
generated tokens match their corresponding request across all eight arm runs,
including exact logprobs. Work counters also repeat exactly for each mode/arm.
Post-run source hashes match the pre-execution manifest.

Timings are unprofiled seconds, excluding engine startup. Positive reduction
means reuse is faster. Request 1 is retained separately because it does not
exercise compact prefill.

| Repeat / arm | Input all-six E2E | Reuse all-six E2E | Reduction | Input request 1 | Reuse request 1 | Input requests 2–6 E2E | Reuse requests 2–6 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / fixed16 | 33.795183 | 33.404016 | 1.157% | 7.657501 | 7.622754 | 26.137682 | 25.781262 |
| 2 / fixed16 | 33.663320 | 33.410911 | 0.750% | 7.622838 | 7.713551 | 26.040482 | 25.697360 |
| 1 / grow17 | 33.529959 | 32.888430 | 1.913% | 7.655732 | 7.564062 | 25.874227 | 25.324368 |
| 2 / grow17 | 33.172800 | 33.179068 | -0.019% | 7.588996 | 7.624795 | 25.583804 | 25.554272 |

| Repeat / arm | Requests 2–6 TTFT input → reuse | TTFT reduction | Requests 2–6 decode input → reuse | Median late TPOT input → reuse (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 / fixed16 | 5.588143 → 5.472134 | 2.076% | 20.548836 → 20.308429 | 64.737 → 63.889 |
| 2 / fixed16 | 5.695611 → 5.500846 | 3.420% | 20.344094 → 20.195784 | 64.140 → 63.363 |
| 1 / grow17 | 5.367314 → 5.240621 | 2.360% | 20.506132 → 20.083079 | 64.184 → 62.977 |
| 2 / grow17 | 5.323234 → 5.228699 | 1.776% | 20.259787 → 20.324900 | 63.583 → 63.714 |

The mechanism is repeatable: compact-prefill loads fall 29.16% / 29.33%, and
later-request TTFT improves 1.78–3.42% in all four matched comparisons. End-to-end
benefit is not equally stable. In repeat-2 grow17, later TTFT saves 94.5 ms but
decode adds 65.1 ms; request 1 adds another 35.8 ms, leaving the all-six E2E sum
6.3 ms worse (0.019%, effectively neutral at this scale). Late TPOT is also
slightly worse in that comparison. These are timing decompositions, not proof
that changed expert loads alone caused the decode variation; two repetitions
do not separate runtime/system noise or establish a latency distribution.

Work below is identical between repetitions. Non-chunk includes request 1 and
scheduler prefetch; it is not pure decode demand. Each materialization is 6 MiB
in this model/layer, consistent with the runtime's total materialization MiB.

| Arm | Compact prefill loads input → reuse | Non-chunk loads input → reuse | Total loads input → reuse | Total materialization MiB input → reuse |
| --- | ---: | ---: | ---: | ---: |
| fixed16 | 2901 → 2055 | 2425 → 2465 | 5326 → 4520 | 31956 → 27120 |
| grow17 | 2901 → 2050 | 2420 → 2464 | 5321 → 4514 | 31926 → 27084 |

Total loads/bytes fall about 15.1–15.2%, despite 40–44 additional non-chunk
loads. Both orders still make 677 H2D batches and four D2H batches per arm; the
weight batch count is unchanged, but each batch carries fewer experts on
average. Eviction D2H is 96 MiB, all before the later-request interval.

Per-request load details (identical in both repetitions):

| Request | Input prefill (both arms) | Reuse prefill fixed / grow | Non-chunk fixed input → reuse | Non-chunk grow input → reuse |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 (no compact prefill) | 0 / 0 | 379 → 379 | 378 → 378 |
| 2 | 619 | 454 / 454 | 453 → 457 | 453 → 457 |
| 3 | 617 | 440 / 440 | 240 → 298 | 236 → 288 |
| 4 | 550 | 389 / 385 | 449 → 441 | 449 → 455 |
| 5 | 578 | 401 / 401 | 452 → 446 | 452 → 443 |
| 6 | 537 | 371 / 370 | 452 → 444 | 452 → 443 |

Groups remain 63/62/56/59/55 for requests 2–6. Request 3 is the clearest changed
handoff: non-chunk loads rise by 58 in fixed16 and 52 in grow17; grow17's borrowed
slot uses fall from 43 to 18. Both orders still lend/return the same six 2 MiB
pages over the entire grow17 run (three on requests 1 and 3). Reduced use is a
different residency trajectory, not a failed loan. No additional physical
growth allocation occurs; final loans, blocked KV tokens and pending transfers
are zero. Every arm ends with 1440 MiB mandatory + 96 MiB cached host backing,
zero idle-pool/batch-owner storage and stable expert pointers.

**Decision:** retain default `input`; keep `reuse` as an opt-in candidate with
demonstrated transfer-volume and TTFT benefit for this bounded workload. Do not
claim a reliable grow17 E2E improvement or implement another scheduling policy
from this timing evidence alone. If continuing, inspect the prefill-to-decode
resident-set handoff, especially request 3, before adding a decode-aware final
group preference. The new policy would need a fresh controlled comparison.

After all GPU experiments finished, the full LayerKV suite passed **248 tests**
(19 warnings), including the 15 new CPU comparator tests. Targeted Black,
isort, Ruff and `git diff --check` pass. Full-repository pre-commit was not run;
no runtime defaults, runtime code or unrelated dirty files were changed, and no
commit was created.
