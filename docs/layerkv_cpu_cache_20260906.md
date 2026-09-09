# Bounded valid CPU expert backing versus idle buffer pooling

## Scope and implementation

Keep GPU expert capacity and KV lending unchanged. Reuse immutable CPU weight
copies already supported by `--layerkv-expert-backing-cache-mb`, and expose the
previously hardcoded idle pool limit as `--layerkv-expert-cpu-backing-pool-mb`
(default 256 MiB). Server arguments reject negative/non-finite limits. Defaults
remain zero valid-copy cache and 256 MiB idle pool; there is no unlimited cache.

The shared performance driver partitions an explicit optional host budget:
`--expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128` means 128 MiB
valid resident copies plus 128 MiB idle buffers. Its control uses cache 0/pool
256. These are static partitions, not a dynamic budget broker. Mandatory
offloaded expert backing is additional and must never be dropped. The tested
shared arena supports one compact expert layer; this is not a multi-layer
cache-policy evaluation. Weights must remain immutable during inference.

KV-loan recall now copies only tail experts without an existing CPU backing.
Previously it could overwrite a valid cached copy and account its bytes twice.
The tail is still synchronized and removed before physical pages return to KV.

`expert_host_budget` is poll-time telemetry in shared-mode summaries: logical
mandatory/cache bytes, idle pool bytes, partition limits, ledger consistency,
and deduplicated CPU storage held by live expert backing, idle pool and CUDA
batch owners (including failed/deferred ownership). Batch-owner storage overlaps
other categories and must not be added twice. Request-boundary guards require
an empty batch-owner set and tracked storage within mandatory plus optional
budget. This is **not** process RSS, framework pinned allocator cache usage, or
a transient/startup peak limit. Other runtime references outside these scanned
containers are not a whole-process memory inventory.

## Installation storage bug found by the new guard

Initial diagnostic artifacts are preserved at
`/mnt/vdb/chengzhi/layerkv_cpu_cache_20260906_cache0_01`. Both arms completed,
but `summary.json` is `valid=false`. Fixed16 had 1440 MiB logical mandatory
backing and only 18–90 MiB idle buffers, yet deduplicated backing/pool storage
was roughly 2.1–2.5 GiB. Batch owners were zero. Installation still created
batched CPU tensor views even under the `individual` layout; consuming some
rows retained the full underlying allocations through other rows.

The minimal failing regression invoked the real installation backup function:
one 8-byte row retained a 32-byte storage after discarding the other rows.
This reproduced without the model or the budget-summary code. Installation
now honors individual owning storage for synchronous and asynchronous copies;
the existing batched layout is unchanged. Installation continues to use its
existing PyTorch copy mechanism; this patch does not claim to batch those
startup submissions with CUDA's batch API. Demand transfers still use that API.
The failed diagnostic is not a controlled-memory performance baseline.

## Validation

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

132 tests pass. New cases cover FP16/BF16 real materialization at zero, partial
and full resident-copy budgets, exact GPU mapping and weights versus cache-zero
control, pool-zero operation, mandatory-copy retention, byte ledger and storage
guards, invalid server limits, cached/uncached tail recall, and sync/async
installation storage with both layouts. In the tiny 2-slot/6-expert test, nine
two-expert loads require 18 D2H copies without cache versus 2 with full resident
CPU copies, with identical GPU slot trajectories. No packages were installed.

## Matched model experiment

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1.
Full-model BF16 retains the validated GDN setup; FP16 unit coverage is not a
full-model FP16 result. Each invocation runs fixed16 then KV-funded grow17,
six different sequential 4096-input/64-output greedy requests per arm. No
profilers or concurrent GPU tests. Both cache configurations use the installation
fix, so this comparison isolates valid-copy retention rather than the old
installation-storage bug. Engine startup is excluded from request E2E sums.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_cpu_cache_fresh_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known \
  --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 --execute
```

Use cache 0 in a fresh directory for control. Omit `--execute` for a dry plan.
`PATH` exposes external framework tools from the existing virtual environment;
ordinary experiment configuration is entirely CLI-based and recorded in the
plan and per-arm engine arguments. Default context 8192, KV capacity 16384,
scratch 4095, block 2048, reclaim target 40 MiB, compact layer 0, initial slots
16, extra slots 0/1, and tail decode starts at output token 16. Tail TPOT is a
late-decode average, not p95; null prefill-forward time is unmeasured, not zero.

### Mechanism measurements

In both repetitions, both arms' cache-enabled runs issue only four
demand D2H batches, backing up the 16 initially GPU-only experts (96 MiB).
Requests 2–6 issue no demand/recall D2H. This does not eliminate the mandatory
installation backup or KV D2H; the counters below are expert eviction traffic.

| Metric, cumulative after six requests | Fixed control | Fixed cached | Grow control | Grow cached |
| --- | ---: | ---: | ---: | ---: |
| CUDA batch D2H submissions | 677 | 4 | 679 | 4 |
| Eviction D2H MiB, including recall | 31956 | 96 | 31926 | 96 |
| CUDA batch H2D submissions | 677 | 677 | 677 | 677 |
| Expert materializations | 5326 | 5326 | 5321 | 5321 |
| Idle-buffer allocations | 512 | 512 | 512 | 512 |
| Idle-buffer reuses | 10620 | 0 | 10610 | 0 |
| Idle-buffer drops | 0 | 0 | 0 | 0 |
| Final mandatory valid backing MiB | 1440 | 1440 | 1440 | 1440 |
| Final resident-copy cache MiB | 0 | 96 | 0 | 96 |
| Final idle pool MiB | 96 | 0 | 96 | 0 |
| Final unique tracked CPU storage MiB | 1536 | 1536 | 1536 | 1536 |

The optional host partitions sum to 256 MiB in every arm; final actual usage
is 96 MiB. During growth a seventeenth cached GPU resident can use 102 MiB,
still below its 128 MiB partition. This is inference from slot/weight sizes,
not a measured transient peak. Snapshot accounting is checked after recall.
The change saves unnecessary D2H and buffer recycling, **not** H2D loads or
GPU expert computation. D2H reductions must not be read as proportional E2E
speedup. There is no prefetch or new GPU residency policy in this patch.

### Artifact comparison

Fresh valid runs use prefix
`/mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_`, in order
`cache0_01`, `cache128_01`, `cache128_02`, `cache0_02`.
Every run retains plan, effective engine args, logs, raw requests and summary.
Use the read-only comparison driver to require identical engine settings except
the two host partitions, equal partition totals, complete and exact token IDs
and logprobs, and unchanged per-request expert loads/H2D counts/prefill groups
and physical KV lending. The JSON report retains per-request E2E, TTFT, decode,
late TPOT, host budget snapshots and transfer deltas, plus engine startup.

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_cpu_cache_compare.py \
  --pair /mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_cache0_01 /mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_cache128_01 \
  --pair /mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_cache0_02 /mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_cache128_02 \
  --output /mnt/vdb/chengzhi/layerkv_cpu_cache_owned_20260906_comparison.json
```

The comparison refuses existing output paths and invalid source summaries.

### Final timing and correctness results

All four valid invocations passed all guards. Across all eight arms, matching
request indices have exactly identical output IDs and logprobs: 48 requests,
3072 generated tokens. Per-request materialization/H2D counts, prefill groups,
and KV loan/return counts match between host-cache configurations. Grow17
borrows and returns six 2-MiB pages cumulatively, with no physical growth
allocation, outstanding loans, blocked KV tokens, or pending batch owners at
request boundaries. The final full unit suite passed with GPUs otherwise idle.

| Arm | Repeat | Control E2E sum (s) | Cached E2E sum (s) | Reduction | Later-request E2E sum, control → cached (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 34.5231 | 33.6905 | 2.41% | 26.8756 → 26.0518 |
| fixed16 | 2 | 34.5732 | 34.1109 | 1.34% | 26.9180 → 26.4251 |
| grow17 | 1 | 33.9735 | 33.7064 | 0.79% | 26.3711 → 26.0535 |
| grow17 | 2 | 34.5185 | 33.6072 | 2.64% | 26.7962 → 25.9581 |

Per-request E2E seconds (C = cache-zero control, V = valid-copy cache):

| Arm | Request | Repeat 1 C | Repeat 1 V | Repeat 2 C | Repeat 2 V |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 (first) | 7.6475 | 7.6387 | 7.6553 | 7.6858 |
| fixed16 | 2 | 5.0838 | 4.8928 | 5.0990 | 4.9594 |
| fixed16 | 3 | 5.2384 | 5.0937 | 5.2492 | 5.1750 |
| fixed16 | 4 | 5.5256 | 5.3648 | 5.5070 | 5.4146 |
| fixed16 | 5 | 5.5292 | 5.3377 | 5.5336 | 5.4418 |
| fixed16 | 6 | 5.4987 | 5.3628 | 5.5291 | 5.4343 |
| grow17 | 1 (first) | 7.6024 | 7.6529 | 7.7222 | 7.6492 |
| grow17 | 2 | 5.0638 | 4.9789 | 5.1491 | 4.9338 |
| grow17 | 3 | 5.1881 | 5.1405 | 5.2887 | 5.1312 |
| grow17 | 4 | 5.3639 | 5.3266 | 5.4645 | 5.3242 |
| grow17 | 5 | 5.4063 | 5.2933 | 5.4567 | 5.2783 |
| grow17 | 6 | 5.3491 | 5.3143 | 5.4372 | 5.2906 |

The first request is not consistently faster. Later-request median TTFT falls
in both repetitions: fixed16 1.3397→1.2656 / 1.3387→1.2819 s; grow17
1.2090→1.1642 / 1.2211→1.1668 s. These are client TTFT measurements, not pure
prefill kernel time. Later-request median late TPOT is fixed16
65.3568→63.9746 / 65.6283→64.8891 ms; grow17
64.6284→64.7075 / 65.8905→64.3849 ms. Thus grow17's late-decode gain is not
consistent across repeats. Two repetitions support a bounded small E2E gain,
not a universal percentage or a tail-latency guarantee.

Keep this opt-in at cache 128/pool 128 for this single-layer workload. Defaults
are unchanged. More GPU slots, larger batch/concurrency, other models/dtypes,
multi-layer caching and in-process mutation of expert weights are not validated
by these runs. The artifact comparison and focused Black/isort/F821 checks
pass for the new driver/tests. `git diff --check` passes. Full-repository
pre-commit was not run; `expert_backing.py` still has a pre-existing F821 at
the unrelated static `LayerKVRuntime` helper (also present in HEAD), which this
change does not fix or claim clean.
