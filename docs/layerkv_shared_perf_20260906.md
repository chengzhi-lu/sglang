# Fixed 16 vs KV-funded 17: bounded performance check

## Outcome

No speedup established. The one request with an actual extra expert slot had
nearly unchanged total latency and a 1.6% slower late-decode window. This is one
paired trial, not a statistically significant regression or a steady-state
benchmark. The page mapping operation itself was small; the extra slot saved
only one expert materialization on this workload.

## Matched configuration

- H100 NVL GPU 0; local Qwen3.6-35B-A3B snapshot
  `995ad96eacd98c81ed38be0c5b274b04031597b0`; **BF16**, not FP16.
- TP=1, batch=1, two sequential requests/arm, 4096 synthetic input IDs and 64
  greedy output tokens/request. The second prompt shifts IDs by 7 in both arms.
- Shared expert layer 0, initial slots 16, 40 MiB KV reclaim target, VMM in both
  arms; context 8192, total KV tokens 16384, scratch 4095, KV block 2048.
- Triton attention/MoE, graphs/overlap/radix/chunked-prefill off. Metrics and
  per-token streaming enabled, detailed LayerKV profiling/debug-stat logging off.
  Direct token-ID output (`--skip-tokenizer-init`) avoids the detokenizer hop and
  preserves scheduler timing fields in the recorded responses.
- A field-by-field comparison of saved engine arguments found exactly one
  difference: `layerkv_shared_expert_extra_slots=0` versus `1`.
- Fixed arm ran first; no randomized ordering/repeated paired trials. Existing
  environment/packages were reused without installation or extension builds.

## Latency

All units below are milliseconds except the explicitly labeled seconds.

| Request | Arm | Scheduler prefill (s) | Client TTFT (s) | Client decode (s) | Mean TPOT (ms) | Tokens 17--64 TPOT (ms) | Client total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First | Fixed 16 | 1.925 | 1.929 | 7.323 | 116.23 | 89.39 | 9.252 |
| First | Grow to 17 | 1.937 | 1.942 | 7.360 | 116.82 | 90.81 | 9.302 |
| Second | Fixed 16 | 3.559 | 3.562 | 4.260 | 67.63 | 66.44 | 7.822 |
| Second | Growth enabled, no loan | 3.201 | 3.204 | 4.707 | 74.71 | 73.26 | 7.911 |

Engine construction/loading took 31.732/32.650 seconds (fixed/growth); these
times are excluded from request latency. The first prefill precedes installation
of the compact expert cache. First decode includes lazy expert installation
(metadata/backing timer 1.266/1.232 seconds), and potentially shape warmup.
The tail window excludes tokens 1--16 but is not proof of thermal/JIT steady state.

Scheduler prefill is `prefill_finished_time-forward_entry_time`, not pure GPU
kernel time. Client decode is last streamed output minus first output. Mean
TPOT divides that interval by 63; the tail divides arrival(64)-arrival(16) by 48.
Both first stream events contained exactly one token. Total includes IPC/client
handling; page return can occur before the final output reaches the client.

## Actual residency and measured overhead

- First growth request: three 2 MiB physical pages transferred and returned,
  16 -> 17 -> 16 slots, 23 MoE calls used the borrowed slot; no new physical
  pages were created during growth. Both arms allocated identical initial
  96 MiB expert backing (48 VMM pages), separate from KV-funded expansion.
- First-request expert materializations: **379 -> 378**, just one fewer.
  Corresponding H2D event sums were 45.39/45.44 ms; there is no observed copy-time
  saving. Do not equate event sums with end-to-end critical path.
- First-request page operation timers:

  | Operation | Existing synchronization wait | Mapping calls after wait |
  | --- | ---: | ---: |
  | KV -> expert | 0.077 ms | 0.567 ms |
  | Expert -> KV | 0.009 ms | 0.558 ms |

  Sum: **1.211 ms**. These timers exclude donor scans, CPU backing, allocator
  bookkeeping and expert metadata/view changes; no extra synchronization was
  introduced for timing.
- The second growth-enabled request got **no pages and no borrowed-slot uses**;
  both arms actually stayed at 16 slots. Both used 63 token-group prefill calls
  and 1072 expert materializations. It is not an active 16-vs-17 comparison.
- Second-request recorded LayerKV Python overhead was 212.75/656.70 ms: an extra
  443.95 ms, close to the extra 446.48 ms of client decode. Code still probes for
  donors each decode when target capacity is unmet. Repeated unsuccessful donor
  scans are a candidate bottleneck, not a separately timed causal attribution.

## Correctness and limitations

`/mnt/vdb/chengzhi/layerkv_shared_perf_20260906_02/summary.json`: `valid=true`.
All 128 output IDs and logprobs matched between arms, maximum logprob difference
zero. KV/expert/ownership/pointer guards passed; final host KV usage, loans,
blocked locations and physical-growth allocations were zero. The 38 LayerKV
unit tests, changed-script/static checks and `git diff --check` passed.

The originally planned four-request run was stopped in the fixed arm after its
first two requests completed: the third was stuck at admission, logging
`waiting=1 running=0 kv_available=0` even though the per-layer summary reported
12289 free arena tokens. No pending KV eviction explained the wait. Its process
group was terminated; no artifacts were deleted. Evidence is preserved under
`/mnt/vdb/chengzhi/layerkv_shared_perf_20260906_01/`, including the first two
responses and failure log. This admission/accounting limitation was **not fixed**
as part of the performance check. Do not treat the two-round retry as a sustained
serving pass. The earlier run also lacked scheduler prefill timestamps; missing
measurements were kept as null, not replaced with zero.

Follow-up: the admission issue was subsequently fixed and both arms completed
six consecutive requests; see the [admission repair report](layerkv_admission_20260906.md).
The original two-request timing conclusions above remain historical results.
Donor scanning was later separately timed and negative-result caching added;
see the [donor cache report](layerkv_donor_cache_20260906.md).

Before a larger matrix: extend admission validation to concurrency, instrument/cache
unsuccessful donor scans, and establish repeated windows with actual page loans.
Then test whether larger/hotness-directed residency gains amortize controller
and transfer cost. This experiment does not establish that physical KV offload
is faster, nor that remapping itself is the main cost.

## Reproduce

Use the command in [the shared VMM guide](layerkv_shared_vmm.md#fixed-capacity-performance-control)
with a fresh output directory. Remove `--execute` to inspect the dry plan.
Raw engine arguments, outputs, every stream-arrival event and per-request stats
are in `fixed16.json`, `grow17.json`, their `*.engine_args.json`, and `plan.json`.
`--summarize-only --decode-tail-start 16` recomputes the saved tail analysis with
the same execution arguments; it launches no model process.
