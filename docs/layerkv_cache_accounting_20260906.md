# Incremental resident CPU backing accounting

## Change and scope

The previous bounded CPU-copy cache removed almost all steady-request expert
D2H, but every positive-budget trim still scanned all layer backing entries
and summed mandatory bytes. Add opt-in server option
`--layerkv-expert-backing-cache-accounting scan|incremental`, default `scan`.
The performance driver exposes `--expert-backing-cache-accounting`.

Incremental accounting applies only to one installed shared expert layer, with
a positive CPU-copy cache budget and no global backing. Other paths, including
cache-zero, use the existing scan policy. Each eligible state lazily creates
a resident-ID-to-byte ledger once, then updates only the affected IDs when
materialization evicts/loads, async D2H publishes backing, backing is dropped,
or shared recall removes a borrowed slot. Offloaded expert backing is mandatory
and does not count toward this optional resident-copy ledger. Byte sizes come
from actual parameter tensors, not a uniform expert-size assumption.

Under budget, trim returns without enumerating backing or summing tensor sizes.
Over budget, it selects resident copies using the same `(backing_lru, logical_id)`
ordering as the scan control. It never drops a mandatory offloaded copy.
Leaving the eligible scope invalidates the ledger; reentry rebuilds it. Generic
compact reinstallation invalidates it as a cold path. Shared KV recall updates
it incrementally without changing synchronization, weights, pointers or loans.

`--layerkv-expert-backing-cache-accounting-check` (driver:
`--check-cache-accounting`) enables an independent full scan before each
incremental trim, checking both per-ID entries and the total. Drift raises and
marks expert correctness/comparability false. This diagnostic option requires
incremental mode and is disabled during performance timing. Independently,
every shared host-budget summary already scans actual backing and now checks
the incremental ledger. Model validation requires an active matching ledger in
incremental positive-budget runs. Summary scans are poll-time checks, not
hotpath bookkeeping. No ordinary configuration is read from environment vars.

This changes CPU accounting only: resident CPU-copy cache 128 MiB, idle pool
128 MiB, GPU slots 16/17, KV budget, transfer batch API and residency policy
remain unchanged. Host figures refer to tracked tensor storage, not RSS or
framework pinned-cache/transient peaks. Inference weights remain immutable.

## Diagnostic measurement before and after

Before editing runtime code, run the current cache-128 configuration with
`--profile-chunks --profile-prepare`; preserve artifacts under
`/mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_profile_scan/`.
Run the implemented incremental mode with identical profiling under
`..._profile_incremental/`. Both fixed16/grow17 summaries pass, and matching
request outputs/logprobs are exact across modes. The initial scan diagnostic
predates the new argument/counters; their absence there means unavailable.
Do not infer missing counter values as measured zeros.

The profile covers 295 preparation calls in requests 2–6 (63, 62, 56, 59, 55).
Request 1 has no chunk-preparation samples, not zero total work. Sum both
first/repeated routing-shape buckets; these are not cold/warm JIT categories.

| CPU inclusive time across profiled requests (ms) | fixed16 | grow17 |
| --- | ---: | ---: |
| Original scan trim | 169.912 | 165.920 |
| Incremental trim | 2.525 | 2.332 |
| New per-ID update helper | 20.701 | 20.343 |
| New ledger lookup/init helper, all callers | 2.976 | 2.831 |

Lookup time partly overlaps trim time; do not sum these nested inclusive rows
as an exact total. The update helper is new work and cannot be omitted from a
savings claim. These measurements contain cProfile overhead and are neither
DMA durations nor predicted E2E savings. GPU-stream timing around preparation
also includes submission gaps. Unprofiled matched runs determine E2E impact.

The same profile shows why less CPU work need not shorten the critical path:
Tensor `item` self time grows from 45.812 to 204.572 ms (fixed16) and 50.390 to
204.264 ms (grow17), while root prepare inclusive time is 680.439→683.856 and
668.580→685.318 ms respectively. Each mode still has 295 scalar reads. The
removed scan follows async H2D submission, so the observations are consistent
with CPU scan overlapping queued GPU work and the remaining wait becoming
visible at scalar readback. This is an inference, not an isolated H2D or remap
kernel measurement, and not justification for removing the correctness wait.

Incremental fixed16 ends with one ledger initialization, 10,652 per-ID updates,
and 677 under-budget fast returns; its 96 MiB accounted resident CPU copies
match the independent scan. The mandatory 1440 MiB plus cached 96 MiB remain
1536 MiB tracked storage, with no idle pool or pending batch-owner bytes.

## Reproduction

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1.
Retain the validated GDN model setup; FP16 transfer tests are not a full-model
FP16 result. Each invocation runs fixed16 then KV-funded grow17, six sequential
different 4096-input/64-output greedy requests per arm. Matching request indices
use identical inputs. No tests run concurrently with GPU benchmarks.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_cache_accounting_fresh_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known \
  --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 \
  --expert-backing-cache-accounting incremental --execute
```

Use `scan` for control, omit `--execute` for a dry plan, and use a fresh output
directory. `PATH` exposes external framework tools from the reused virtualenv;
normal experiment options are CLI arguments recorded in plan/engine metadata.
Defaults remain context 8192, max KV tokens 16384, scratch 4095, block 2048,
reclaim target 40 MiB, shared layer 0, initial slots 16 and extra slots 0/1.
First requests and engine startup are retained separately. Tail TPOT starts
at output token 16 and is a late average, not p95. Null prefill-forward time is
unmeasured, not zero. Both profilers and per-trim debug scans are off in timing.

Read-only matched-artifact comparison is available via
`scripts/layerkv_cpu_cache_compare.py --comparison-kind accounting --pair CONTROL INCREMENTAL --output FRESH_JSON`.
It requires equal settings except accounting mode, exact output IDs/logprobs,
unchanged expert materializations, H2D/D2H, pool operations, host usage and KV
loans. Existing default cache comparison behavior remains supported.

## Regression coverage

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

New tests cover no-rescan under-budget behavior, nonuniform tensor sizes,
partial/zero budgets and equal-age eviction ties versus scan, unchanged
mandatory backing, deliberately corrupted per-ID/total ledgers, untracked
residency changes, unsupported-scope fallback and reentry, CLI configuration,
real FP16/BF16 async backing publication, nine repeated demand swaps and
cached/uncached recall. Full scans remain the independent correctness oracle.

The final full LayerKV suite passes **159 tests**, including 27 new accounting
cases. Targeted Black/isort/F821 checks and `git diff --check` pass for the new
driver/test code and edited regions. Full-repository pre-commit was not run;
the previously documented unrelated static `LayerKVRuntime` helper in
`expert_backing.py` retains its pre-existing F821. No packages were installed,
CUDA extensions built, commits made, or unrelated dirty changes discarded.

## Final unprofiled results and decision

Four fresh invocations under `/mnt/vdb/chengzhi/`, prefix
`layerkv_cache_accounting_20260906_`, ordered `scan_01`, `incremental_01`,
`incremental_02`, `scan_02`. All four source summaries are valid. The exact
cross-mode comparison is saved as
`/mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_comparison.json`:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_cpu_cache_compare.py \
  --comparison-kind accounting \
  --pair /mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_scan_01 /mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_incremental_01 \
  --pair /mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_scan_02 /mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_incremental_02 \
  --output /mnt/vdb/chengzhi/layerkv_cache_accounting_20260906_comparison.json
```

Use another fresh output path to repeat the comparison; it refuses overwrites.
Across all eight arms, corresponding requests have identical token IDs and
logprobs (48 requests, 3072 generated tokens). H2D/D2H, materializations,
prefill groups, host storage and KV loans match per request. Incremental modes
initialize once and return early at all 677 trims in each arm. Update counts
are 10,652 fixed16 / 10,642 grow17 in both repetitions. Every request-boundary
independent ledger check passes. Both modes end at 1440 MiB mandatory backing
+ 96 MiB valid cache, 0 idle-pool bytes, 0 pending batch-owner bytes, for
1536 MiB tracked storage. The optional budget remains 128+128 MiB. No new
physical expert growth allocation, outstanding loans or blocked KV tokens.

| Arm | Repeat | Scan E2E sum (s) | Incremental E2E sum (s) | E2E reduction | Requests 2–6 sum, scan → incremental (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 33.6319 | 33.6338 | -0.006% | 25.9922 → 26.0158 |
| fixed16 | 2 | 33.8706 | 34.1614 | -0.859% | 26.2221 → 26.4547 |
| grow17 | 1 | 33.8797 | 33.6815 | +0.585% | 26.0997 → 26.0301 |
| grow17 | 2 | 33.7921 | 33.6877 | +0.309% | 26.0144 → 26.0136 |

Per-request E2E seconds (S = scan, I = incremental):

| Arm | Request | Repeat 1 S | Repeat 1 I | Repeat 2 S | Repeat 2 I |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 (first) | 7.6396 | 7.6180 | 7.6484 | 7.7067 |
| fixed16 | 2 | 4.9013 | 4.9079 | 4.9233 | 4.9686 |
| fixed16 | 3 | 5.0753 | 5.0855 | 5.1334 | 5.1610 |
| fixed16 | 4 | 5.3146 | 5.3289 | 5.3875 | 5.4423 |
| fixed16 | 5 | 5.3651 | 5.3472 | 5.3917 | 5.4248 |
| fixed16 | 6 | 5.3360 | 5.3462 | 5.3863 | 5.4580 |
| grow17 | 1 (first) | 7.7800 | 7.6514 | 7.7777 | 7.6741 |
| grow17 | 2 | 4.9695 | 4.9676 | 4.9593 | 4.9588 |
| grow17 | 3 | 5.1535 | 5.1358 | 5.1452 | 5.1314 |
| grow17 | 4 | 5.3246 | 5.3112 | 5.3029 | 5.3382 |
| grow17 | 5 | 5.3298 | 5.3104 | 5.3093 | 5.2970 |
| grow17 | 6 | 5.3223 | 5.3050 | 5.2977 | 5.2882 |

Later-request median TTFT is fixed16 1.2412→1.2662 / 1.2678→1.2863 s;
grow17 1.1681→1.1581 / 1.1569→1.1659 s. Later-request median late TPOT is
fixed16 64.0823→63.9346 / 64.6756→64.9992 ms;
grow17 64.9397→64.6604 / 64.7312→64.5515 ms. These are not kernel or p95
measurements. Most of grow17's small E2E reduction comes from the first
request; its second-repeat later-request sum differs by less than 1 ms.

**Decision: retain as opt-in, do not enable by default or claim a demonstrated
general E2E speedup.** The implementation removes repeated CPU byte scanning,
but profiling suggests much of that work overlapped GPU execution. Fixed16
is neutral/slower and grow17's later-request improvement is small/inconsistent.
Two repetitions cannot establish a general regression cause either. A better
next experiment should distinguish remaining H2D/remap/queued-compute waits
before attempting safe prefetch overlap; deleting the guard is not a valid
optimization. No prefetch or synchronization removal was implemented here.
