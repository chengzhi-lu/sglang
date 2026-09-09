# CPU-known demand for shared prefill groups

## Scope

Server option `--layerkv-shared-expert-prepare-path cpu-known` reuses the routing
snapshot already copied to CPU by shared prefill token grouping. The performance
driver exposes `--prepare-path generic|cpu-known`; the default stays `generic`.
No expert budget, grouping membership/order, admission, transfer backend, or
decode policy changes are included. Existing environments/packages are reused.

The controller extracts ascending logical IDs from each group's expert bitset.
On a miss, the existing materializer receives **all** those IDs, including hits,
in the same order as generic `torch.unique`. This preserves protection against
evicting needed experts, slot selection, LRU/backing-LRU and loading behavior.
On an all-hit group, materialization and LRU updates are skipped as before.

The fast path is restricted to the same shared controller state, non-decode
dispatch, an applied expert plan, optimized runtime, compact slots, calibrated
valid CUDA integer routing and consistent CPU forward/reverse slot maps. All
other cases fall back to generic preparation. Pre-plan hotness collection is
therefore unchanged; the applied-plan path does not introduce hotness updates.
If the CPU routing snapshot contains invalid IDs, the controller disables the
calibrated range shortcut before generic fallback, preserving sentinel IDs and
avoiding unchecked out-of-range indexing.

GPU remapping and the post-load scalar guard are **retained**, with an upper
slot-bound check as well. H2D ready-event waits remain before mapping/use, also
on all-hit groups. This removes redundant GPU demand discovery and
`unique().cpu()` per eligible group; it does **not** eliminate all host waits or
the initial batch routing `.tolist()`. The scalar guard still waits for mapping
and preceding stream dependencies. This is not a fully asynchronous scheduler.

Counters `expert_cpu_known_prepare_count` and
`expert_cpu_known_prepare_fallback_count` record coverage. Shared-VMM stats
include `token_chunk_prepare_path`. The benchmark checks that successful plus
fallback counts cover every token chunk and that eligible work actually runs.
Prepare profiling forwards the optional demand argument while retaining the
same root hook/call-count validation. Profiling is off in the performance runs.

## Regression coverage

`test_cpu_known_prepare.py` compares actual generic and CPU-known preparation
with real CUDA materialization, individual pinned backing, stream-ordered
demand backup and a separate H2D stream. It covers FP16/BF16, input/reuse group
order, profiler on/off, complete top-k reductions, row placement, input alias
protection, exact outputs, load order, slot maps, LRU and unchanged hotness.
Additional cases cover partial/all-hit demand, readiness invocation, invalid
routing fallback, unsupported states, pre-plan hotness collection, and retained
GPU guards for negative/out-of-capacity remaps. This complements the existing
transfer tests of real D2H/H2D stream dependencies and copy owner lifetimes.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

## Controlled experiment

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1.
Full-model BF16 retains the previously validated native GDN configuration;
unit coverage of FP16 does not establish full-model FP16 support.

Two matched repetitions, with configuration order reversed: generic_01,
cpu_known_01, cpu_known_02, generic_02. Each invocation runs isolated fixed16
then grow17 arms and six sequential requests per arm, retaining the same
per-request inputs across configurations/repetitions. Different request indices
use different prompts and residency trajectories; they are not six identical
steady-state samples. Both profilers are off, so phase timers of zero mean
unmeasured, not zero work. Streaming/correctness instrumentation remains on.

Directories are under `/mnt/vdb/chengzhi/`, prefixed
`layerkv_prepare_20260906_` and suffixed with the four names above. Each contains
`plan.json`, per-arm effective engine arguments, logs, request JSON and summary.
The process sequence is serialized on GPU 0; GPU tests are not concurrent.

Example command (choose a fresh output directory):

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_prepare_cpu_known_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --chunk-order input \
  --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual \
  --expert-demand-d2h-wait stream --prepare-path cpu-known --execute
```

Use `--prepare-path generic` in another fresh directory for the control. Omit
`--execute` for a dry plan. `PATH` only exposes required external framework
tools; ordinary settings are explicit CLI options. Recorded defaults include
context 8192, KV capacity 16384, scratch 4095, block 2048, reclaim target 40 MiB,
shared layer 0, initial slots 16, and extra slots 0/1 for fixed/grow arms.

## Results

All four summaries report `valid=true`. Across all 48 requests (3072 generated
tokens), corresponding output token IDs **and logprobs match exactly** between
configurations, repetitions and fixed/grow arms. Every request is comparable
and passes KVC/expert/ownership guards. The only difference in effective engine
arguments between configurations is `layerkv_shared_expert_prepare_path`.

Every CPU-known arm executes 295 eligible chunks with zero fallbacks. Per-request
chunk counts are `[0, 63, 62, 56, 59, 55]`; prefill materializations are
`[0, 619, 617, 550, 578, 537]`, total 2901, identical in all runs. Request 1 has
no shared compact prefill chunks and is not a measurement of this fast path.
Whole-run materializations are 5326 for fixed16 and 5321 for grow17. CUDA batch
H2D/D2H counts remain 677/677 and 677/679 respectively. These per-request
trajectory counters match the corresponding baseline, not merely the totals.

Grow17 borrows and returns three 2 MiB pages in requests 1 and 3 (six cumulative
page loans), with zero extra growth-time physical page creation. Fixed16 borrows
none. Every request ends with zero outstanding loans, blocked KV tokens and
pending CUDA batch copies, and with the original 16 expert slots restored.

Client TTFT in seconds; G = generic, C = CPU-known:

| Arm | Request | G repetition 1 | C repetition 1 | G repetition 2 | C repetition 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed16 | 1 | 1.9167 | 1.9328 | 1.8828 | 1.9213 |
| fixed16 | 2 | 1.0814 | 1.0485 | 1.0829 | 1.0528 |
| fixed16 | 3 | 1.1846 | 1.1543 | 1.1626 | 1.1562 |
| fixed16 | 4 | 1.3816 | 1.3789 | 1.3786 | 1.3698 |
| fixed16 | 5 | 1.3829 | 1.3791 | 1.3882 | 1.3723 |
| fixed16 | 6 | 1.3704 | 1.3362 | 1.3661 | 1.3727 |
| grow17 | 1 | 1.9240 | 1.8802 | 1.9238 | 1.8903 |
| grow17 | 2 | 1.0903 | 1.0714 | 1.0873 | 1.0619 |
| grow17 | 3 | 1.2068 | 1.1976 | 1.1890 | 1.1586 |
| grow17 | 4 | 1.2915 | 1.2959 | 1.2870 | 1.2791 |
| grow17 | 5 | 1.2813 | 1.3014 | 1.2665 | 1.2838 |
| grow17 | 6 | 1.2466 | 1.2667 | 1.2801 | 1.2457 |

Aggregates are descriptive for this matched workload, not steady-state estimates.
E2E is the sum of six client request durations, excluding engine startup.
Positive improvement means less time:

| Arm | Repetition | Mean TTFT requests 2–6, G → C (s) | TTFT improvement | Six-request E2E, G → C (s) | E2E improvement |
| --- | ---: | --- | ---: | --- | ---: |
| fixed16 | 1 | 1.2802 → 1.2594 | 1.62% | 35.8701 → 35.6298 | 0.67% |
| fixed16 | 2 | 1.2757 → 1.2648 | 0.86% | 36.0847 → 35.9000 | 0.51% |
| grow17 | 1 | 1.2233 → 1.2266 | -0.27% | 35.5323 → 35.7138 | -0.51% |
| grow17 | 2 | 1.2220 → 1.2058 | 1.32% | 35.7910 → 35.8181 | -0.08% |

Mean late-decode TPOT over requests 2–6 (driver `tail_tpot_ms`, not a percentile)
is 65.278 → 64.943 ms and 65.746 → 65.609 ms for fixed16 repetitions 1/2;
grow17 is 64.910 → 65.399 ms and 65.637 → 65.946 ms. This path changes prefill
preparation only: do not attribute these small decode timing differences to a
decode optimization. Server prefill-forward timestamps are unavailable (`null`),
so client TTFT must not be reported as pure GPU prefill time.

Conclusion: redundant demand discovery is eliminated on the covered path, with
a small fixed16 benefit, but **no demonstrated grow17 end-to-end speedup** and
mixed per-request TTFT changes. Two repetitions do not establish a broad stable
gain. Keep the option experimental and the default generic. Before further
changing synchronization, measure the remaining post-load guard, ready-event
and backing-ownership costs separately; this unprofiled comparison cannot
attribute their individual shares.

The follow-up [direct caller attribution](layerkv_prepare_callers_20260906.md)
separates remaining scalar waits, ready-event calls and backing release callbacks.

Final checks: **90 LayerKV unit tests passed**, including 21 new CPU-known
cases; import-level `layerkv_smoke.py` passed. Black/Ruff checks on the focused
controller/profiler/driver/test files and `git diff --check` passed. No repository-
wide formatting or pre-commit rewrite was applied to the preexisting dirty tree.
Both GPUs were idle with zero allocated process memory after validation. Work is
uncommitted on `perf/layerkv-allocation-fastpath`, based on
`6012837c636420f82901c90eefb55b291862a52e`, with earlier LayerKV work preserved.
