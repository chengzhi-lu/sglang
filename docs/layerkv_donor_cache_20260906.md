# Shared expert donor miss cache

Historical report: the 2026-09-07 concurrent validation removed eviction
finalization from donor discovery after reproducing output divergence even
without a successful loan. Completion now remains in the normal KV lifecycle;
its publication still invalidates the miss cache. The original ordinary-free
donor mode disabled that cache because ordinary-free changes lacked its
generation contract. The follow-up adds generation bumps when native pages or
protected/allocated pages become free, so ordinary-free misses can use the
same negative cache without retaining donor addresses. See
[adaptive residency](layerkv_adaptive_residency_20260907.md).
The measurements below describe the earlier implementation, not current behavior.

## Scope and cause

After the [admission repair](layerkv_admission_20260906.md), the bounded
shared-VMM controller still scanned all potential donor pages on every decode
step until it could fund another expert slot. An unchanged set of insufficient
donors produced the same answer repeatedly.

New wall-clock instrumentation separates `_donors()` from the existing blocking
eviction-finalization call. In the uncached six-request replay, grow17 request 2
scanned 63 times with 63 misses: 406.425 ms in donor selection versus 15.369 ms
in finalization at this call site. Requests 4 and 5 also scanned 63 times, costing
371.782 / 300.800 ms. These are measured CPU control intervals, not GPU kernel
times or an exhaustive attribution of all LayerKV overhead.

## Implementation

Only an **insufficient-donor result** is cached. No positive donor addresses,
physical handles or ownership decisions are retained. The key comprises:

- The existing offloaded-index version.
- A new monotonic donor-availability version, advanced when free overwrite bits
  are published or live cleanup ownership is removed. This covers eviction,
  request completion and physical-page recall even if the offload index is
  unchanged.
- Current and target expert slot capacity.

The existing eviction-finalization call runs **before** checking the key: a
pending D2H completion must not be hidden by a previous miss. KV consumption
alone cannot increase eligibility, so it does not invalidate a negative result.
The donor predicate, exclusive ownership claim, lend/recall sequence, physical
budget and expert placement policy remain unchanged. Other free-list formats
remain ineligible for donation, as in the original conservative donor predicate.

Caching defaults on only within the opt-in shared expert controller. Server
flag `--layerkv-shared-expert-disable-donor-cache` disables it for an ablation;
the flag requires a configured shared expert layer. The benchmark exposes this
as `--disable-donor-cache` and records it in the plan and engine arguments.
There is no environment-variable configuration path.

Shared-VMM telemetry adds `donor_scan_count`, `donor_scan_ms`, `donor_miss_count`,
`donor_cache_hit_count` and `donor_finalize_ms`. The cached implementation also
reports `donor_cache_enabled`; effective engine arguments establish the mode
for the earlier instrumentation-only baseline. Timers add no CUDA syncs.

## Correctness tests

The original minimal test called `after_decode()` eight times with an unchanged
empty donor set. It failed with **8 scans instead of 1**, before the cache was
implemented. Added regressions cover repeated misses, the uncached control,
free-bit publication, offload-index changes, live-owner removal, and completion
of pending offload before a cache lookup. Config propagation is tested as well.

The CUDA VMM regression now exercises a partial donor set (two pages when three
are required), cache hits on that miss, publication of the missing token, actual
growth, recall and another successful loan with no new physical allocation.
All 48 LayerKV unit tests passed after the cache change:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

The controller, driver and changed test files pass Black/isort/Ruff checks;
`git diff --check` passes. Full-file Ruff on the four touched existing runtime /
configuration files has the same 21 findings as HEAD, with no new diagnostics.
The server CLI flag's parsing and rejection without a shared layer were checked
from the worktree's `python/` directory. No full-repository pre-commit was run.

## Bounded replay

H100 NVL GPU 0; local Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`; **BF16**, TP=1, batch=1. Each run
contains fixed16 followed by grow17, six sequential requests per arm, 4096
synthetic input IDs and 64 greedy output tokens per request. Context 8192,
KV capacity 16384, scratch 4095, block 2048, reclaim 40 MiB, expert layer 0,
initial slots 16. Graphs/overlap/radix/chunked-prefill remain disabled, tokenizer
enabled, per-token streaming enabled. Missing scheduler prefill times remain
null. Existing packages/environment were reused.

Instrumented uncached baseline:
`/mnt/vdb/chengzhi/layerkv_donor_baseline_20260906_01/` (`valid=true`).
Cached replay: `/mnt/vdb/chengzhi/layerkv_donor_cached_20260906_01/`
(`valid=true`). Both runs completed both arms and all six requests/arm.

### Results

For the same arm across runs, saved engine arguments differ only in
`layerkv_shared_expert_disable_donor_cache` (true -> false). Within each run,
fixed16 / grow17 differ only in extra expert slots (0 / 1).

Grow17 request-level results (before -> after):

| Request | Donor scans | Donor scan time (ms) | Mean TPOT (ms) | Pages lent/returned in each run |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 -> 1 | 5.973 -> 5.979 | 116.567 -> 120.025 | 3 / 3 |
| 2 | 63 -> 2 | 406.425 -> 13.500 | 73.389 -> 68.158 | 0 / 0 |
| 3 | 1 -> 1 | 6.069 -> 6.097 | 67.670 -> 69.554 | 3 / 3 |
| 4 | 63 -> 2 | 371.782 -> 12.129 | 70.921 -> 66.616 | 0 / 0 |
| 5 | 63 -> 2 | 300.800 -> 10.180 | 69.785 -> 66.590 | 0 / 0 |
| 6 | 63 -> 2 | 431.443 -> 14.821 | 71.991 -> 66.589 | 0 / 0 |

Requests 2/4/5/6 get 61 cache hits each. Their combined scan time falls from
1510.450 to 50.629 ms (96.6% reduction), and their TPOT median falls from 71.456
to 66.603 ms (6.8%). The observed cache invalidations still allow two scans per
such request; caching is not an unconditional once-per-request rule.
Mean TPOT is client arrival(last)-arrival(first), divided by 63. It is not GPU
kernel time. Existing finalization is still performed, not bypassed by a hit.

Fixed16 never scans donors in either run. Its TPOT medians on those same four
requests are 65.700 / 66.538 ms. Its third request has a large unrelated timing
variation (92.270 / 69.058 ms), despite no donor scans in either mode; do not
attribute a whole-run before/after speed change to caching. The first request
also includes lazy setup. Active-loan requests 1/3 show no observed speedup here.

Every run passes the driver's KV/expert, ownership, pointer stability and
cleanup guards. Each run's 384 paired fixed/grow output IDs and logprobs match
exactly. Across the cache modes, all 768 corresponding output IDs and logprobs
(both arms combined) also match exactly. Request 1 uses the extra slot 23 times
and request 3 uses it 43 times, unchanged by caching. Each loan borrows and
returns three 2 MiB pages, with no growth-time physical page creation.
All requests finish with zero host KV usage, no outstanding loans/blocked
locations and 12289 free arena tokens. No admission-stall warnings occur.

### Reproduce

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_donor_cached_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --execute
```

Remove `--execute` for a dry plan. Add `--disable-donor-cache` and choose a new
output directory for the uncached control. `PATH` only exposes existing external
tools such as Ninja; all experiment settings are explicit CLI arguments.

This is one ordered before/after run, not a randomized repeated-trial estimate
or a steady-state workload. Compare borrowed-slot uses and output correctness
as well as timing; not every growth-enabled request receives pages. A control
overhead reduction does not itself prove a benefit from extra expert residency.
