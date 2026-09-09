# Batched materialization remap updates

## Change and safety boundary

Add opt-in `--layerkv-expert-remap-update scalar|batch` (server) and
`--expert-remap-update` (performance driver). The default remains `scalar`.
This follows the wait trace in `layerkv_wait_trace_20260906.md`, where 537
prefill replacements produced 1074 scalar remap H2D copies/synchronizations
before bulk expert weight H2D.

The batch path collects only the logical IDs changed by `_materialize_experts`.
Evictions add `(old_id, -1)` and incoming experts add `(new_id, slot)` in the
same control flow as scalar updates. A dict gives last-write-wins semantics
without duplicate GPU scatter indices. Immediately before the existing D2H/H2D
copy path, publish one packed int64 ID/value tensor and one `index_copy_` on the
producer stream. The tensor constructor still uses a blocking small host copy;
this reduces many scalar synchronizations to one, not to zero.

No full remap-table rebuild, persistent mirror, reusable pinned staging, CUDA
extension build or pointer replacement is introduced. New temporary tensors
own their storage and PyTorch manages their current-stream lifetime. The packed
metadata uses 16 bytes per changed ID, versus 8 bytes per scalar value; its
purpose is fewer submissions, not fewer metadata bytes. It is not expert weight
backing and does not change the resident CPU-copy cache or GPU slot budget.

Unchanged entries are never rewritten, including erroneous entries that must
still be detected by the final remap guard. An all-hit call submits nothing.
The materialization order, protected IDs, LRU/slot decisions, hotness, backing
cache, copy descriptors and weight-copy backend are unchanged. The producer
dependency before demand H2D, readiness waits and final `any().item()` guard
remain. Existing scheduler prefetch uses the same batched publication option;
this does not add chunk lookahead or change prefetch policy.

If backing lookup or slot selection fails partway through, the `finally` path
publishes pending partial updates, including invalidation of evicted IDs, to
match the scalar path. A publication failure sets expert guard/comparability
false and is not retried. This does not make a partially failed materialization
transaction recoverable; the original failure still aborts the operation.

Counters `expert_remap_batch_count` and `expert_remap_batch_entries` report
actual batch publications and unique changed IDs. The driver verifies that
batch mode was exercised; scalar remains zero. Final runtime KVC, expert,
ownership, host-budget and stable-pointer guards remain required.

## Correctness and mechanism validation

`test_remap_batch.py` covers real FP16/BF16 transfers, cached/uncached backing,
demand/prefetch materialization, repeated replacements and duplicate IDs, exact
GPU maps/weights, LRU/backing order, transfer counters and pointer stability.
Delayed producer-stream weight reads are followed by the next replacement
without test-side CPU fences; results are checked after the sequence. Assertions
use the recorded logical-to-physical slot mapping, not an assumption that slots
remain in logical-ID order. All-hit and mixed hit/miss invalid GPU entries still
raise. Partial failure invalidations match scalar; failed publication is not
retried. Invalid server argument values are rejected.

A real CUDA profiler probe of two replacements with cached CPU backing verifies
**4 scalar `cudaStreamSynchronize` calls versus 1 batch call** inside actual
materialization. No model timing is inferred from this tiny probe.

`test_remap_compare.py` tests matched artifact comparison and rejects changed
runtime arguments, engine/host budgets, output tokens, transfer work, failed
guards and a candidate whose batch path was not exercised. The existing
comparison helper supports `--comparison-kind remap` in addition to its prior
cache/accounting modes. Remap comparison requires exact output IDs/logprobs,
identical materializations, H2D/D2H counts, pool operations, host storage and KV
loan work, with only the remap-update argument different.

## Bounded experiment

Reuse `/mnt/vdb/chengzhi/agent_bench_sglang_venv`; no package installations or
compilation. H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1. This is
the validated model/GDN setup, not a full-model FP16 result. Each launch runs
fixed16 then KV-funded grow17, with six different 4096-input/64-output greedy
requests per arm. First request and startup are retained separately.

Keep CPU cache 128 MiB and idle pool cap 128 MiB, `scan` accounting,
`cuda-batch` weight transfers, individual backing layout, stream-ordered demand
D2H, admission-time release validation, CPU-known preparation and input chunk
order. All profilers and accounting debug scans are off. Do not run GPU tests
concurrently with model timing. Inspect the dry plan first.

From `/home/chengzhi/github/sglang-perf-layerkv`:

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_remap_fresh_batch \
  --gpu 0 --dtype bfloat16 --rounds 6 --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual --expert-demand-d2h-wait stream \
  --prepare-path cpu-known --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 \
  --expert-backing-cache-accounting scan --expert-remap-update batch --execute
```

Use `scalar` and a fresh output directory for control. Omit `--execute` for the
dry plan. `PATH` only exposes external framework tools; ordinary experiment
configuration is explicit CLI metadata. Unchanged defaults: context 8192, max
KV tokens 16384, scratch 4095, block 2048, reclaim target 40 MiB, shared expert
layer 0, initial slots 16 and extra slots 0/1. Tail TPOT begins at output token
16 and is a late average, not p95; null prefill-forward time is unavailable.

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_cpu_cache_compare.py \
  --comparison-kind remap \
  --pair /mnt/vdb/chengzhi/layerkv_remap_20260907_scalar_01 /mnt/vdb/chengzhi/layerkv_remap_20260907_batch_01 \
  --output /mnt/vdb/chengzhi/layerkv_remap_fresh_comparison.json
```

Run regressions from the worktree's `python/` directory:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

## Results

Two matched repetitions, reversing family order: scalar_01, batch_01, batch_02,
scalar_02. Each family includes both fixed16 and grow17. Raw plan, engine args,
logs, streaming requests and guard summaries are preserved under
`/mnt/vdb/chengzhi/layerkv_remap_20260907_{scalar,batch}_{01,02}/`.
The final two-pair report is
`/mnt/vdb/chengzhi/layerkv_remap_20260907_comparison.json`; the first-pair report
`..._comparison_01.json` is also retained. To regenerate the final report, add
the second `--pair` to the command above and use a fresh output filename.

All timings below are **unprofiled**, seconds except the final table's tail
column. E2E sums include all six requests but exclude engine startup.

| Repeat / arm | Scalar E2E sum | Batch E2E sum | Reduction | Scalar request 1 | Batch request 1 | Scalar requests 2–6 | Batch requests 2–6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / fixed16 | 33.636966 | 33.492949 | 0.428% | 7.637625 | 7.746835 | 25.999341 | 25.746114 |
| 2 / fixed16 | 33.530414 | 33.495154 | 0.105% | 7.606891 | 7.627063 | 25.923522 | 25.868091 |
| 1 / grow17 | 33.450953 | 33.379227 | 0.214% | 7.635981 | 7.627605 | 25.814972 | 25.751622 |
| 2 / grow17 | 33.500543 | 33.215795 | 0.850% | 7.693927 | 7.596910 | 25.806616 | 25.618884 |

| Repeat / arm | Requests 2–6 TTFT sum scalar → batch | Requests 2–6 decode sum scalar → batch | Median late TPOT scalar → batch (ms) |
| --- | ---: | ---: | ---: |
| 1 / fixed16 | 5.672328 → 5.579959 | 20.326300 → 20.165533 | 64.062 → 63.381 |
| 2 / fixed16 | 5.709352 → 5.634834 | 20.213496 → 20.232480 | 63.713 → 63.617 |
| 1 / grow17 | 5.429480 → 5.364058 | 20.384852 → 20.386786 | 63.984 → 64.034 |
| 2 / grow17 | 5.399566 → 5.323163 | 20.406361 → 20.294955 | 63.955 → 63.707 |

Repeated-request TTFT improves 1.20–1.63% in all four matched comparisons.
Full E2E improves only 0.11–0.85%; decode and first-request changes are mixed.
This is a small measured benefit under this workload, not a broad or
statistically established speedup. Two repetitions are insufficient for strong
tail claims; late TPOT above is a median of per-request late averages, not p95.
Keep the default scalar and the batch path opt-in. No extra experiment families
were launched to chase a larger improvement.

All 48 requests (3072 output tokens) pass guards, and output IDs and logprobs
match exactly across all eight arms at corresponding request indices. Each
batch arm has 677 publications; fixed16 publishes 10652 changed entries and
grow17 10640. Unchanged per-arm whole-run work: 5326/5321 materializations
(fixed/grow), 677 weight-H2D batches, 4 eviction-D2H batches. The D2H batches
occur in the first request; subsequent requests have zero eviction D2H.

Final tracked CPU storage remains 1440 MiB mandatory + 96 MiB cached, no idle
pool or pending batch-owner storage. Grow17 lends and returns six 2 MiB KV
pages in total; final loan/blocked counts and new growth physical allocations
are zero. KVC stale/alignment counts are zero, KVC/expert/ownership guards pass,
and expert pointers stay stable. The incremental-accounting and CPU-cache
comparison modes also still pass their prior artifact pairs unchanged.

The final full LayerKV suite passes **204 tests** (19 warnings), including 23
new remap cases and 8 new comparison cases. No CUDA/model packages were installed.
Targeted Black/isort/F821 and
`git diff --check` pass. Full-repository pre-commit was not run over the large
pre-existing dirty worktree; no commit or unrelated cleanup was performed.

If proceeding further, test a bounded H2D overlap opportunity only after
establishing when a future chunk's destination slots are no longer in use.
This change does not justify removing the final guard or overwriting live
expert slots for lookahead.
