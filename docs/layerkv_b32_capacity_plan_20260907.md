# B32 capacity gate (dry plan, not a result)

The B8 mixed diagnostic is complete; do not rerun it for this gate. This plan
tests larger actual decode batches and queued long-context pressure with one
new workload, not a parameter sweep or the main300s trace evaluation.

## Capacity audit

The local Qwen3.6 snapshot text config has30 linear and10 full-attention layers.
`Qwen3_5MoeTextConfig` inherits the Qwen3Next Mamba cache shape. At TP1, runtime
FP16 convolution state and FP32 temporal state require, per request:

`30 * (8192 * 3 * 2 + 32 * 128 * 128 * 4) = 61.40625 MiB`.

With radix and overlap disabled, `handle_max_mamba_cache` defaults the Mamba
request capacity to `max_running_requests`. Memory-pool tensors allocate an
additional padding row. Changing8 to32 therefore adds1473.75MiB; the33-row
state tensors total2026.40625MiB, excluding auxiliary buffers. This is an
analytical state-storage estimate, not measured peak allocation.

Both new arms must use32 and identical graph settings. Comparing B32 adaptive
against old B8 fixed would confound residency with request-pool memory and
replay coverage. The previous device snapshot (~66.1GiB used of~93.1GiB) suggests
room for this gate, but does not prove B32 prefill/workspace peak safety.

Short workload:32 * (128 +64) =6144 logical tokens at completion.
Long workload:8 * (4096 +64) =33280 logical tokens, above the16384 configured
KV token pool. This establishes intended pressure, not actual batch capacity:
physical residency, offload scratch, scheduler reserves and admission all matter.
Do not report a computed resident-only floor as an observed running batch.

## Reproducible dry plan

Run from `/home/chengzhi/github/sglang-perf-layerkv`. The following command was
parsed successfully without loading a model or creating its output directory:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_01 \
  --dtype float16 --requests 32 --max-running-requests 32 --rounds 3 \
  --input-tokens 128 --output-tokens 64 --scratch-tokens 2048 \
  --expert-extra-slots 48 --expert-policy adaptive \
  --retain-experts-across-requests --native-moe-graph-max-batch-size 32 \
  --expert-benefit-horizon-steps 64 --expert-cpu-backing-mode selected \
  --round-workloads '[{"requests":32,"input_tokens":128,"output_tokens":64},{"requests":8,"input_tokens":4096,"output_tokens":64},{"requests":32,"input_tokens":128,"output_tokens":64}]'
```

Confirmed expanded engine arguments: both arms max_running32, max_total_tokens
16384, selected backing and native MoE replay limit32. Fixed has0 extra slots;
adaptive requests48. Both retain the same initial16 slots and physical layout.
Actual adaptive capacity must be read from results;49 from B8 is not a promise.

## Execution gates and interpretation

1. First run adaptive child only, after checking GPU process state and fresh
   output directory. Record actual batch histogram, queue/admission behavior,
   completed4608 tokens, guard/fault counters, physical pool and device memory.
   This single run tests the previously uncovered B32 path and pressure episode.
2. If the mechanism/outputs are incomplete, diagnose that artifact before any
   fixed/native launch. No automatic retry or capacity sweep.
3. If viable, fixed16 with the same selected backing/replay/memory configuration
   is the first performance comparator, not the older backing-none B8 run.
   Independent output validation remains required; different scheduled batches
   must not be excused solely by token-ID agreement.
4. Fixed16 gives adaptive no inherently greater KV capacity: it is the useful
   KV-heavy competitor. An expert-heavy retained split is separately needed to
   demonstrate admission recovered by returning expert loans; beating only that
   split does not establish superiority over reasonable fixed allocations.
5. Diagnostic flags and graph-memory acceptance guards remain unchanged. This
   gate cannot by itself certify20%, main-trace performance, or matched peak
   physical memory. Aggregate throughput includes all phase transition costs.

No model launch, new runtime behavior, or repeated unit tests in this planning
turn. The next model run has a different missing gate from the completed B8
chunk profiling, and should not add profiling overhead.

## First execution: scheduler prefetch failure, narrow fix

The planned adaptive child was run once in `mixed_b32_01`, with `--execute
--arm lend`. It terminated with exit137 after a scheduler exception, not a
completed benchmark. The log records native E36 expert kernel configuration
before failure. There is no complete result JSON and no throughput acceptance.

The actionable traceback is `_run_deadline_scheduler` ->
`_schedule_recovery_tasks` -> expert backend `recover` -> `_materialize_experts`
-> `_choose_expert_slot_for_materialize`: layer0 has no evictable expert slot.
Exit137 alone is not evidence of OOM; this log provides a concrete child error.
GPU process enumeration after termination was empty.

A CPU regression on the actual task builder reproduces the missing capacity
constraint: two slots generate a task for three missing experts. Recovery
protects every task ID at once, so this cannot fit; unlike demand execution,
recovery has no token-chunk execution between materializations.

`_build_recovery_tasks` now limits missing IDs to the current physical capacity
before constructing groups, byte estimates, signatures and demands. Candidate
counts still include all misses. Unselected IDs remain on the normal demand
path; no outputs, on-demand protections, KV ownership, or budget are changed.
This builder emits one expert task per layer. This fix does not claim a generic
capacity guard for arbitrary externally assembled/coalesced recovery tasks.

Targeted validation: the new capacity regression failed before the fix (3 >2).
After the fix it and the existing missing-demand test passed:2 passed,
16 deselected. The new test also drives the real slot chooser for each task ID
while marking previously chosen slots occupied/protected. `git diff --check`
passed. No full suite, fixed/native launch, or model retry in this turn.

Next gate is one same-settings adaptive retry in a fresh directory, to check
the original B32 failure path and complete the workload. The small test is not
proof of model correctness or20% throughput improvement.

## Corrective retry completed: donor availability is the next gate

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_02/lend.json`, same arguments
as the first child except output directory. Exit0. All72 requests completed64
tokens each (4608 total), with64 finite output-logprob entries each. This is
structural output validation, not independent native numerical validation.
No fixed/native run or repeated unit suite was launched.

| Phase | Actual decode forwards | Final expert slots | Output tok/s | Phase seconds |
| --- | --- | ---: | ---: | ---: |
| Short32 | B32:63 | 36 | 255.433 | 8.018 |
| Long8 | B3:126, B2:63 | 16 | 32.364 | 15.820 |
| Short32 | B32:63 | 16 | 180.434 | 11.350 |

Guards for KV, expert, ownership and zero reconstruction pass in every snapshot;
stale, alignment and physical-failure counters are0. Shared physical pool stays
456MiB. Long-phase admission recalls once, recovers6144 common KV tokens and
takes42.642ms. The120MiB loan is fully returned. Thus B32 execution and the
long-phase loan recall run successfully, but return to short-context expert
retention is incomplete.

Final-short donor scans increase8->15, misses7->14, while grow_count remains1
and donor_select_ms is unchanged. All seven new scans fail before page selection;
the final cost rejection is not sufficient explanation for the failed regrowth.
Aggregate free tokens alone cannot establish complete, unprotected donor pages.
Investigate physical page fragmentation/ownership eligibility using the current
allocator code before modifying the benefit horizon or running another model.

Final-short materializations12348 versus7698 in the initial short phase; prompts,
installation/cold behavior and history differ, so this is not a matched speedup.
Prefetch candidates/issued are0/0,45/30,16/16 by phase, with useful0/0/5 and
wasted0/30/11. The corrected scheduler handles the long-phase candidate overflow
without the previous exception. No claim that prefetch is performance-positive.

Torch lifetime peak reserved reaches73410805760bytes; final device-used snapshot
72651243520 of99951443968bytes. These are not a sampled device-wide peak or proof
of fixed/adaptive memory comparability. Results remain diagnostic-only under the
existing graph acceptance guard. Do not launch fixed solely to compare against
this incomplete short-phase regrowth mechanism. The next gate is donor diagnosis,
not another capacity sweep or a repeat of the successful B32 execution.

## Donor rejection observability

The saved B32 artifact cannot reconstruct the seven failed scans: it contains
aggregate misses, not live page eligibility. Its final14336 common-free tokens
are measured after requests finish. They do not prove complete free donor pages
existed during decode. No fragmentation or stale-ownership fix is justified yet.

Added `shared_vmm.donor_last_scan`, a snapshot of the latest completed scan:
mapped page count and first-rejection counts for edge, cleanup, not_free,
not_reserved, allocated, protected, plus eligible pages. Categories partition
mapped pages and are not cumulative. `not_free` includes any incomplete free
coverage; it alone does not establish fragmentation. In the offloaded-only path,
the existing combined eligibility bitmap remains unchanged and rejected pages
are classified not_free. Null means no completed scan, not a zero-page scan.

This adds CPU counter increments to existing scans, no CUDA synchronization or
extra allocator scan, and does not relax eligibility/ownership. The next missing
gate is observing these reasons during the same B32 transition; do not rerun
fixed/native or tune horizons to obtain that observation. This is targeted
instrumentation because the existing artifact lacks the distinguishing evidence,
not an attempted fix of an unproven allocator root cause.

Six CPU fixtures verify rejection accounting and snapshot replacement. Eight
existing differential cases were selected because the combined ownership check
was split into individually counted checks: outputs still match the independent
set-based reference.14 passed,25 deselected. Black, targeted Ruff and diff check
pass. No model run in this instrumentation turn.

## Rejection diagnostic completed

Artifact `/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_donors_01/lend.json`;
same B32 mixed adaptive settings, exit0. All4608 output token IDs and logprob
arrays match `mixed_b32_02` exactly. KVC/expert/ownership guards pass in all
three snapshots. No fixed/native launch, parameter sweep or new timing claim.

The last scan in phase3 reports180 mapped pages:40 edge pages and140 cleanup
rejections,0 eligible. Other first-rejection counts are0. Scans increase8->15
and misses7->14. Thus the latest phase3 donor failure is specifically at the
cleanup ownership check, before free/reserved/protected checks. This does not
prove the other checks would succeed, nor that all seven scans had identical
contents. Phase2 repeats phase1's snapshot because no scan occurred there.

`_track_per_layer_cleanup_entry` and `_track_per_layer_cleanup_append` register
physical locations of current KV entries, not merely already-completed requests.
Consequently cleanup rejection is not by itself stale metadata. Removing that
guard or publishing eviction completion here would violate live ownership.
`_remove_per_layer_cleanup_loc_bits` removes explicit owners and bumps donor
availability; the record must be checked against its request lifecycle before
being called a leak.

The next bounded CPU investigation should exercise allocation/free ordering:
`_alloc_common_per_layer_locs` pops `_per_layer_arena_common_free_order`, while
free/rebuild paths preserve or append historical order. Live locations can then
intersect every2MiB donor page even when total token use is low. This is a
candidate mechanism, not yet a demonstrated cause in the model. Construct a
real-allocator fixture of long-request reuse followed by short prefill and
decode, inspect live-page occupancy, and test page-packing only if it reproduces
the loss of whole-page donors. No further model run is needed merely to confirm
the cleanup rejection category now captured here.

## CPU lifecycle reproduction and address-ordered rebuild

Added `test_long_request_free_order_changes_short_request_donor_pages` using
the real common allocator, batch free and dirty common-free rebuild, plus the
real donor scanner. GPU backing/growth and stats refresh are stubbed; page size
is scaled to16 token rows. Two requests interleave32 decode allocations, finish
in sequence, then a short request allocates16 of the same64 available locations.
This is a controlled mechanism reproduction, not recovered model address data.

Historical free-order reuse spreads the16 live locations across two pages:
4 eligible K/V page handles. Address-ordered reuse leaves them on one page:
6 eligible handles. Both have48 free tokens, identical backing and no donor
overlap with live locations. The initial counterfactual fixture passed; after
replacing its manual ordering with the runtime feature path, it failed4 !=6
before the runtime edit and passed afterward.

`_ensure_per_layer_common_free_current` now sorts the already-computed common
free set by address when `shared_expert_free_kv_donors` is enabled. Its existing
dirty guard remains; there is no new per-decode sort or live data relocation.
The intersection still excludes allocated and protected locations. Other modes
retain historical ordering. Both shared fixed/adaptive arms inherit this common
allocation change. This is address packing, not a globally optimal page-density
allocator; fragmented free sets or future allocations can still touch many pages.

Only the new CPU case was run for red/green validation, no model launch. The
initial invocation from repository root failed at import collection; running
from the worktree `python/` directory reused the existing environment correctly.
Diff whitespace check passed. Whole allocator Ruff reports12 preexisting unused
imports in the untouched import block; no unrelated import cleanup was applied.

Next gate: one B32 same-settings validation of actual donor recovery, outputs
and physical guards. The CPU gain in free pages is not a20% performance result,
nor proof that historical free order fully explains the real model failure.

## Model packing check failed its mechanism gate; runtime change reverted

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_packed_01/lend.json`, same
settings, exit0. All4608 token IDs/logprob arrays match `mixed_b32_donors_01`
exactly, and all per-round KV/expert/ownership guards pass. Slots remain36/16/16;
final scan still rejects140 non-edge pages for cleanup ownership. Materialization
counts7698/5664/12348 are unchanged. Throughput256.714/34.437/182.638tok/s is a
single uncontrolled timing observation, not evidence of successful packing.

The added address-sort runtime branch was removed, preserving unrelated edits.
The CPU test remains explicitly counterfactual (manual ordering), not a passing
test that purports to verify a deployed optimization. No test/model rerun merely
to reconfirm the pre-change behavior. Diff whitespace check passes.

The existing allocation counters provide the missing path distinction without
another instrumented launch: final-short common allocation4096 tokens covers
prefill, while overwrite reuse handles all63 decode forwards/2016 tokens.
`alloc_per_layer_decode` attempts `_alloc_per_layer_overwrite_locs_by_layer`
before `_alloc_common_per_layer_locs`. The previous CPU fixture exercised only
the latter, so it did not cover the actual decode allocator. Long-phase counters
similarly show32768 common tokens and189 overwrite reuses/504 tokens.

Next: reproduce the overwrite-first decode path in a CPU lifecycle fixture,
including its ordering and cleanup registration, before implementing another
packing policy. Do not disable live cleanup checks or repeat the failed common-
queue sorting experiment. The20% goal and native correctness gate remain open.

## Overwrite-first decode regression and publication merge

New CPU fixture drives the real `allocate_decode_slots_for_batch`, not just the
common allocator. Two completed interleaved requests publish separate bit chunks;
eight two-request decode forwards consume16 addresses. Common allocation is
forbidden by the fixture. Actual cleanup registration must equal exactly those
16 addresses; remaining pending capacity must be48, with no donor/live overlap.
Before the fix it yields4 eligible K/V handles instead of6 (two versus three
whole free token pages). Thus this regression covers the observed allocator path.
It remains a scaled controlled history, not a recovered model-address trace.

When `shared_expert_free_kv_donors` is enabled, `_push_per_layer_overwrite_bits`
now merges already-published bit chunks at publication. The existing pop code
then selects highest addresses across requests rather than exhausting a strided
request chunk first. Counts track unique chunk addresses, including overlapping
publication. No new CUDA work, active KV relocation, ownership relaxation or
per-decode union is added. Other modes retain existing behavior. The failed
common-free ordering optimization remains reverted.

Validation: the new actual-decode regression failed4 !=6 before the edit, then
passed. Two existing reload-live-ownership checks passed (their unchanged mode
is regression coverage, not validation of the new branch). A separate new shared
publication test covers overlap, pop, republish and complete drain/counts. No
model or full-suite run. Next gate: same B32 transition, verify actual expert
regrowth, native/output comparison caveats and physical guards before measuring
against fixed splits. No20% result is claimed from these CPU page counts.

## Publication merge passes the B32 regrowth gate

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_union_01/lend.json`, same
adaptive settings, exit0. Expert slots now36 ->16 ->36, grow_count1/1/2.
Long-phase admission still recalls once and recovers6144 common KV tokens.
Physical pool remains478150656bytes with0 growth physical creates; all KV,
expert and ownership guards pass. All4608 token IDs and output-logprob arrays
exactly match `mixed_b32_donors_01`, the pre-merge/pre-common-sort reference.
This is regression agreement, not an independent native numerical reference.

Actual decode histograms remain B32:63 / (B3:126,B2:63) / B32:63.
Final-short materializations fall12348 ->9537. Its last donor snapshot has no
eligible pages after successful lending, which is not evidence of regrowth
failure: grow_count, current slots and removed mappings demonstrate the loan.

Phase times7.9832/14.2059/9.9871s yield256.539/36.041/205.064 output tok/s.
Aggregate4608 tokens divided by total time is143.211tok/s versus134.732tok/s
in `mixed_b32_donors_01`, about6.29% higher in this single before/after pair.
Final-short mean TPOT83.037ms and mean TTFT4.755s still expose substantial
transition/prefill cost. Do not average phase speedup percentages, discard
transition costs, or call the uncontrolled pair a20% result.

Keep the publication merge; the common-free sort remains reverted. No CPU tests,
fixed or native runs were repeated in this model-validation turn. Next is the
first same-settings B32 fixed16 comparator with selected backing, replay limit32
and the same publication code. Old backing-none/B8 fixed results are not that
comparator. Graph diagnostic/physical-memory acceptance gates remain unchanged;
main-trace and independent native validation remain outstanding.

## First matched fixed16 comparator: aggregate +4.01%, not20%

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_fixed16_01/fixed.json`, exit0.
Adaptive results reused from `mixed_b32_union_01`, no repeat. Saved engine args
differ only in extra slots48/0 and adaptive/fixed expert policy. Both use the
publication merge, selected immutable CPU backing and native replay limit32.

| Phase | Fixed16 tok/s | Adaptive tok/s | Relative difference |
| --- | ---: | ---: | ---: |
| Short32 initial | 236.165 | 256.539 | +8.63% |
| Long8 queued | 36.483 | 36.041 | -1.21% |
| Short32 return | 190.329 | 205.064 | +7.74% |

All4608 token IDs and logprob arrays agree exactly across arms, and actual
decode histograms agree in all phases. Guards pass. Shared pool456MiB and all
recorded graph memory snapshots (Torch allocated/reserved/lifetime peaks and
device used/total) match per phase. These snapshots are not sampled device-wide
peaks; graph diagnostic acceptance remains disabled. No independent native
reference or multi-order/repeated performance acceptance is supplied here.

Aggregate throughput uses4608 / sum(client_e2e_s): fixed137.6915tok/s versus
adaptive143.2114tok/s, +4.0089%. The long phase already achieves the same B3/B2
schedule under fixed16, so expert recall does not increase batch over this
KV-heavy comparator. Expert-heavy fixed comparison remains a separate admission
question, not a substitute for this stronger comparator.

Final-short TPOT94.420ms fixed versus83.037ms adaptive; TTFT4.811s versus4.755s.
Materializations10087/5659/12348 fixed versus7698/5664/9537 adaptive. The expert
decode benefit is real in this pair, but prefill/transition time still dilutes
it substantially. Do not extrapolate the earlier B8 prefill timing to B32 or
call TTFT an exclusive selected-layer prefill cost. Next investigation should
reuse existing counters to distinguish that remaining cost before another
targeted change; repeating this comparator alone is not the next optimization.

## Existing request timestamps redirect the next optimization to admission

No model launch or added timers were needed for this distinction. Response
metadata already contains queue_time, forward_entry_time and
prefill_finished_time. Per-request averages from the existing two artifacts:

| Arm/phase | Queue interval seconds | Forward-entry to prefill-finished seconds |
| --- | ---: | ---: |
| Adaptive short initial | 0.002657 | 1.786113 |
| Fixed short initial | 0.002666 | 1.791067 |
| Adaptive long | 4.650131 | 0.927385 |
| Fixed long | 4.499012 | 0.915278 |
| Adaptive short return | 3.356798 | 1.391464 |
| Fixed short return | 3.398222 | 1.406063 |

These are per-request averages, not additive phase totals: requests overlap and
the long phase contains three admission waves. No absent profiling values were
treated as zero cost; chunk/profile timers were disabled in these runs.

`ReqTimeStats.get_queueing_time` measures wait-queue entry to forward entry.
Scheduler sets forward entry after the admission loop (`init_next_round_input`
and `adder.add_one_req`) and before `ScheduleBatch.init_new` and
`prepare_for_extend`. Therefore the3.36s is pre-batch-construction time, not an
exclusive hardware wait. The1.39s interval includes batch preparation, forward
and result processing, not just selected-layer MoE kernels. The earlier4.8s
TTFT description must not be interpreted as4.8s of model prefill execution.

Final-short admission recall count and recall time are both0, so loan recall
alone cannot explain this delay. Actual batch32 is eventually assembled in one
extend, with no queued requests left at the prefill log. Both arms exhibit the
same large pre-forward delay despite different expert residency policy.

Next: distinguish request initialization/prefix matching from admission capacity
queries and scheduler-loop waiting, using their real call boundaries. Do not
optimize expert grouping or early prefill growth merely on the TTFT number.
This is a shared overhead candidate; any fix must apply equally to fixed and
adaptive arms and cannot itself be credited as adaptive residency benefit.

## Count-only admission fast path

`PrefillAdder.rem_total_tokens` repeatedly queries LayerKV prefill credit.
The credit branch previously called `len(_common_per_layer_reusable_locs())`,
expanding per-layer published bitmaps into address lists/sets although admission
only needs their intersection count. Added `_common_per_layer_reusable_token_count`:
intersect freshly read CPU ownership bitmaps and count bits. Reserved, ordinary,
pending-list, allocated and protected addresses retain the address-query semantics.
Tensor bitmap presence falls back to the existing address path. No capacity cache,
new sync, reclaimed credit, or ownership publication is introduced. Only the
`decode_prealloc_admission` count branch uses it; callers needing actual addresses
remain unchanged. Fixed and adaptive both receive this shared optimization.

Four randomized differential cases (five ownership mutations each) match the
address reference. Two integration cases cover the actual planner credit branch
and legacy CPU tensor-bitmap fallback; the nonlegacy case forbids address-query
fallback. No broad tests or model run in this turn.

CPU mechanism benchmark:10 layers,14336 reserved/free addresses held in published
bitmaps,32 queries, three repetitions. Old address expansion medians638.442ms
(639.067/638.442/638.308); new count-only median105.543ms
(105.820/105.543/105.408), all return14336. The configuration resembles the
post-long-request pool but is not a trace of the model's exact admission calls;
it does not prove the entire3.36s delay is this function or a20% end-to-end gain.
Next bounded model gate should examine saved queue/prefill times and all output,
admission, actual-batch and physical-budget guards under unchanged workloads.

## Count-only model gate and symmetric fixed comparator

Adaptive artifact `layerkv_fp16_20260907_mixed_b32_count_01/lend.json` and fixed
artifact `layerkv_fp16_20260907_mixed_b32_count_fixed_01/fixed.json`, both under
`/mnt/vdb/chengzhi/`, both exit0. Same workload/settings as the preceding pair.
No CPU tests, matrix, or repeated adaptive launch in this verification turn.

Adaptive final-short queue interval falls3.3568 ->0.5767s while forward-entry
to prefill-finished stays1.3915 ->1.3899s. Aggregate throughput143.211 ->158.135
tok/s (+10.42%) in the before/after pair. This supports the admission-query
attribution, not a claim that all queue time is removed. All4608 token IDs and
logprob arrays match the old adaptive artifact exactly. Slots36/16/36 and actual
batch histograms remain unchanged; KV/expert/ownership guards pass, pool456MiB.

The symmetric fixed16 run also benefits: final-short queue0.6079s, prefill
interval1.4016s. Current comparison:

| Phase | Adaptive tok/s | Fixed16 tok/s |
| --- | ---: | ---: |
| Short32 initial | 255.558 | 233.322 |
| Long8 queued | 36.882 | 36.778 |
| Short32 return | 282.725 | 255.032 |
| Aggregate (4608 / total seconds) | 158.135 | 149.955 |

Adaptive net aggregate advantage is5.4546%, not20%. All4608 cross-arm token IDs
and logprob arrays match exactly; actual batch histograms, physical pool and all
recorded memory snapshots match. Fixed guards pass. This remains a single ordered
controlled pair, not repeated acceptance, sampled peak-device-memory proof,
independent native verification, or the main300s trace result. Do not compare
new adaptive against old unoptimized fixed to claim an adaptive-only gain.

## Independent native output reference and expert-set provenance

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_native_01/native.json`, exit0.
Saved native engine args confirm LayerKV disabled and no layerkv_* options;
FP16, TP1, max_running32, max_total_tokens16384 and eager execution retained.
Native output IDs and logprob arrays match `mixed_b32_count_01` exactly across
the three workloads. This is independent numerical verification, not a physical-
budget-matched performance baseline; full resident native expert memory differs.

No dataset-specific expert-ID set is supplied by the experiment driver. Shared
installation chooses layer0, then `_select_initial_resident_experts` uses online
candidate ordering/hotness (ID tie-break fallback), and materialization responds
to routed demand with protected LRU replacement. Candidate ordering is generated
from runtime hotness snapshots, not a dataset whitelist. Selected CPU backing
means all experts of the selected layer have immutable host copies; it does not
mean a handpicked GPU expert subset. The synthetic token workload, single selected
layer and retained within-process history remain experimental limitations, not
evidence of generalization across real datasets or achievement of20%.

## Cross-layer prefetch policy follow-up

The scheduler already issues later-layer expert recovery on the copy stream
after the current layer's KV preparation. Existing B32 counters show why this
window must be policy-gated: in the long-context/small-batch phase, adaptive
issued 30 speculative expert copies and used 0; all 30 were later classified as
wasted. The new gate keeps ordinary demand-driven recovery but rejects
`allow_bounded_context` cross-layer expert lookahead in this regime, leaving
the reclaimed window to KVC. Short-context/large-batch cross-layer overlap
remains enabled, and no dataset-specific expert-ID set is introduced.

The first matched GPU follow-up is
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b32_hotnessprefetch_cap_01`.
It uses the same FP16 B32/long8/B32 workload, selected CPU backing, and graph
disabled. The result is `valid=true`: 4608 output tokens and logprobs match,
physical pools match, and KV/expert/ownership guards pass.

| Phase | Fixed tok/s | Adaptive tok/s | Relative difference |
| --- | ---: | ---: | ---: |
| Short32 initial | 158.382 | 160.087 | +1.076% |
| Long8 queued | 19.947 | 19.878 | -0.348% |
| Short32 return | 197.453 | 198.758 | +0.661% |
| Aggregate (4608 / total client time) | 94.097 | 94.321 | +0.238% |

The context gate is exercised: the long phase records 49 context skips per arm,
and neither arm issues speculative expert prefetch there. In the first short
phase adaptive issues 1021 bounded candidates (707 useful, 310 stale/wasted),
versus fixed's 626 (603 useful, 23 wasted); the return short phase is identical
at 323/309/12. This validates the route-fallback connection and the batch-sized
cap, but not a throughput win of 20%.

This run also exposes two scope limits. Only one expert layer is selected, so
`expert_prefetch_cross_layer_issue_count` remains zero; the cross-layer path is
not yet an end-to-end experiment. Further, actual decode histograms are equal
in both arms (B32, then B3/B2, then B32), so the run cannot demonstrate that KV
space recovered from experts admits a larger batch. The next model gate should
therefore increase offered concurrency and check admission/actual-batch growth
under the same physical budget, rather than repeat this B32 prefetch pair.

## B48 admission gate: larger batch is admitted, but no policy-driven batch split

The next non-duplicate gate used
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b48_admission_01`. It changed the
offered and maximum running requests to48 while keeping the FP16 model, physical
budget, selected backing, scratch2048, graph disabled and B48/long8/B48
workloads otherwise matched. The run is `valid=true`: 6656 output tokens and
logprobs match exactly, physical pools match, and all guards pass.

| Phase | Fixed tok/s | Adaptive tok/s | Relative difference | Actual decode batches |
| --- | ---: | ---: | ---: | --- |
| Short48 initial | 213.651 | 238.325 | +11.548% | B48 in both |
| Long8 queued | 20.100 | 19.851 | -1.240% | B3/B2 in both |
| Short48 return | 238.130 | 236.509 | -0.681% | B48 in both |
| Aggregate (6656 / total client time) | 126.176 | 128.814 | +2.091% | identical |

Adaptive grows to29 slots in the first short phase, then recalls 4096 common KV
tokens once in the long phase; fixed remains at16. The long-context gate records
49 context skips and no speculative expert copies in either arm. Thus the
context-aware preference and the larger offered B48 path are operational, but
this configuration still admits B48 in both policies. It does not demonstrate
expert-to-KV admission enlargement, and the aggregate result remains far below
the20% target.

The short-phase transfer counters explain why more slots do not translate
directly to20%: adaptive reduces first-phase expert materializations from15421
to13246 and ready-miss stall from1112.7 to1020.3ms, but still issues1248
prefetch candidates, of which239 are later wasted. The next admission experiment
must make the static KV/mamba limit nonbinding and compare against a retained
expert-heavy fixed split; otherwise a B48 run only measures residency at an
already-admitted batch.

## Expert-heavy fixed comparator: preconditioning is necessary but not sufficient

The concurrent driver now supports
`--precondition-expert-heavy-fixed`. With `--baseline-split expert-heavy`, this
keeps both arms at the same initial/extra configuration and runs an untimed
fixed-arm warmup. The warmup must reach `initial_slots + extra_slots` through
the normal KV donor loan path; new physical creation or an incomplete target
is a hard error. The unit coverage for the driver passes (`79 passed`), and the
dry plan keeps both engine arms at16+48 while recording the explicit warmup
arguments.

The first B48 *allocation-based* preconditioned run was
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_b48_expertheavy_preconditioned_01`.
It used the old allocation-based implementation and is deliberately rejected
by the strict summary guard: the fixed arm's physical pool grew to780,140,544
bytes, while the adaptive arm used478,150,656 bytes. Outputs/logprobs and
per-round guards still matched, but the fixed arm received an extra physical
allocation budget, so its throughput is not a valid comparison. The run also
confirms that fixed64 versus adaptive29->16 is observable, including one
adaptive long-context recall of4096 KV tokens, but that observation is
diagnostic only.

The next expert-heavy performance run must equalize the physical arena/pool
before timing (or make both arms explicitly preallocate the same capacity).
Do not weaken `physical_pool_matched` or reuse this run as a20% result. The
cross-layer prefetch path also remains unmeasured end-to-end because the current
model experiment selects only one expert layer.
