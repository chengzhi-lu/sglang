# Same-engine residency transition gate

The short B8 diagnostic currently reaches17.10% over the existing graph-enabled
fixed16 observation; see `layerkv_page_first_donors_20260907.md`. That does not
establish superiority to a suitable static split across short/long workloads.
Stop tuning the now-small donor scan and validate the intended transition.

## Driver support

`scripts/layerkv_concurrent_perf.py --round-workloads '<JSON list>'` accepts one
object per configured round. Required fields: requests,input_tokens,output_tokens.
Optional fields: initial_requests,late_input_tokens,late_after_seconds,
late_after_output_tokens. Unknown fields (including model/memory/policy knobs)
are rejected. Values and context bounds are validated before Engine launch.
Arrival fields default afresh per phase, not inherited from prior phases.
Mixed workloads cannot combine with separate round-output-triggers or profiling.

One Engine is constructed before the round loop; no memory/model reconfiguration
occurs between phases. Plans include expanded effective workloads, result rows
record the exact workload, and summaries validate counts against each phase and
reject mismatched workload metadata between arms. Existing graph-mode physical
budget/diagnostic acceptance restrictions remain in force.

## Dry plan completed, no model run yet

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_plan_01 \
  --dtype float16 --requests 8 --max-running-requests 8 --rounds 3 \
  --input-tokens 128 --output-tokens 64 --scratch-tokens 2048 \
  --expert-extra-slots 48 --expert-policy adaptive \
  --retain-experts-across-requests --native-moe-graph-max-batch-size 8 \
  --round-workloads '[{"requests":8,"input_tokens":128,"output_tokens":64},{"requests":2,"input_tokens":128,"output_tokens":64,"late_input_tokens":6000,"late_after_seconds":1},{"requests":8,"input_tokens":128,"output_tokens":64}]'
```

Dry plan exits0 without creating the output directory or starting GPU work.
12 new mixed-workload CPU tests passed;3 existing valid-summary compatibility
cases passed. Includes invalid types/ranges/context/arrival combinations,
forbidden engine-field overrides, per-phase reset, different request counts in
summary despite intentionally wrong global counts, and metadata-mismatch rejection.
No full test suite or model matrix was run.

## Next evidence required

One adaptive mechanism child first, not a new policy matrix. Check initial short
growth, actual queued KV shortage and recall/admission in the middle phase, then
renewed short-batch growth. Final slot counts alone cannot prove the transitions:
inspect recall/growth/admission counters, actual batch histograms and physical
ownership guards. The6000-token phase is a hypothesis about pressure, not proof
of queueing; absence of real shortage must be reported, not relabeled success.

Replay B8→B1/B2→B8 changes exact graph shape and can recapture; report capture
cost separately from warm throughput. Scratch2048 has only short-context model
validation so far. Reuse existing round1/round3 short outputs when comparable;
middle-phase output correctness needs a matched reference. Different actual
batch schedules can change FP16 rounding, requiring schedule-matched native
comparison rather than relaxing correctness based on throughput.

Eventually compare reasonable fixed splits on the identical mixed arrival
sequence and total physical memory budget with common optimizations enabled in
both arms. This transition probe is not the300s real-trace main evaluation and
does not complete the20% objective.

## First adaptive transition run: regrowth gate failed

Executed only the adaptive child using the dry-plan command above with fresh
directory `/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_adaptive_01` and appended
`--execute --arm lend`. Existing environment selected through PATH for external
framework executables. Child exited0; three result rows and effective engine
arguments are saved. No fixed/native process or parameter sweep was launched.

| Phase | Final expert slots | Actual decode forwards | Admission recall | KV credit recovered |
| --- | ---: | --- | ---: | ---: |
| Short B8 | 49 | B8:63 | 0 | 0 |
| Timed6000-token arrival | 16 | B1:24,B2:51 | 1 | 10240 tokens |
| Short B8 again | 16 | B8:63 | 0 | 0 |

Initial16→49 growth borrowed207618048bytes (198MiB); grow_count stayed1 for
the entire run. Middle admission recall took59.87ms, restored actual common KV
capacity, and virtual KV materialization count increased382. Long-request TTFT
was0.815s from its submission. Shared physical backing remained478150656bytes.
All three phases passed KVC/expert/ownership/pointer guards, with zero stale,
alignment and physical-failure counters. First/third phase1024 token/logprob
pairs exactly match the existing short B8 reference. Middle128 outputs still
require a schedule-matched reference; guards alone do not establish correctness.

The intended complete transition did NOT occur: final short B8 remained16slots.
The final budget reason is `cost-exceeds-benefit`, estimated_saved_ms27.096,
last_growth_ms135.899. Middle phase added7 cost rejections; final short phase
added8 more. Its expert materializations rose4092 versus2004 for retained49 in
the prior same-index short run. This is not a correct-regrowth success merely
because all requests finished.

Source inspection confirms the cause of the observed decision: `ResidencyBudget`
prices one observed interval (8samples) against the entire measured one-time
growth cost, then resets samples/misses. After recall it therefore never considers
benefits over a longer upcoming short-batch episode. Initial growth used the
uncalibrated-cost exception; regrowth must pass the measured-cost gate.

Next change should make the benefit horizon explicit and justified by available
request-lifetime/workload evidence, while retaining immediate admission-pressure
override. Do not fix by erasing the measured cost on recall, accumulating past
misses as future savings, or simply removing cost rejection. Validate the
decision rule with the recorded costs and short/long remaining-work cases before
another model launch. Also separate cold first growth cost from reusable warm
transition cost if the implementation can measure them meaningfully.

Graph captures changed39→156→195 and replay counters2379→5070→7449 across the
three phases. Shape-transition captures are real costs; none of these phases
constitutes a matched steady-state20% performance acceptance result.

## Bounded future-benefit implementation (model validation pending)

Added opt-in `--layerkv-shared-expert-benefit-horizon-steps` (driver spelling:
`--expert-benefit-horizon-steps`), default0 retaining historical interval pricing.
The scheduling hook reads CPU request metadata only. For ignore_eos requests
without stop strings/tokens/regex, remaining future steps are the minimum across
the active batch of max_new_tokens minus produced tokens minus this forward.
Unknown/early-stoppable requests retain single-interval pricing. Prefill clears
the observation; queued work disables horizon expansion. The forecast cap is
explicit and cannot exceed that observed remaining-work bound. Cancellation or
future arrivals can still shorten residency: this remains a forecast, not a
guaranteed residence interval or admission credit.

The budget projects observed per-step saved misses over the bounded future
window, preserves measured one-time growth cost, and still resets observations
after each decision. It does not accumulate past opportunities as future value.
KV pressure still overrides retention immediately. Nested budget diagnostics
report forecast_steps and remaining_decode_steps; nullable remaining work is
not inserted into generic scalar-counter deltas.

12 focused CPU cases passed using the recorded27.096ms/8steps savings and
135.899ms growth cost:56 future steps allow a proposal;8/2/0 or unknown horizon
do not bypass the cost gate. Tests cover uncertain stopping, earliest departure,
current-forward subtraction and pressure override. A scheduling-hook test also
covers clearing the horizon on prefill/uncertain stopping. No model run or
performance gain has been claimed for this change.

Next gate is the same mixed adaptive command plus explicit
`--expert-benefit-horizon-steps 64`, after remaining integration checks. Verify
actual regrowth and cost amortization rather than only a changed budget target.
The existing shadow estimator still models max64 although physical capacity
previously reached49; this proxy error must remain visible when interpreting
forecast accuracy. Broader early-stopping workloads and trace acceptance remain
unvalidated.

## Horizon64 model probe: regrowth observed, performance acceptance still open

Executed one child in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_horizon64_01`, identical to the
first mixed adaptive invocation plus `--expert-benefit-horizon-steps 64`.
Scheduling integration supplies real ScheduleBatch requests and uppercase DECODE
is normalized. Process exited0; no fixed/native reference rerun.

Inspection also corrected the stopping-condition field to actual
`SamplingParams.stop_regex_strs`. A test using the real SamplingParams constructor
passed; the model experiment has no regex stopping condition and is unaffected.

| Phase | Slots at end | Cumulative grows | Materializations | tok/s | Previous mixed tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short B8 | 49 | 1 | 1929 | 68.757 | 69.382 |
| Long arrival | 16 | 1 | 1497 | 23.227 | 23.455 |
| Short B8 again | 49 | 2 | 2846 | 97.012 | 95.299 |

The observed path now is16→49→16→49. Middle admission recall count is1 and
recovered common KV capacity10240tokens, as before. All KVC/expert/ownership
guards pass, pointers stable, stale/alignment/physical failures0, shared backing
478150656bytes throughout. Regrowth measured159.04ms versus initial137.38ms;
it is not cost-free. Third-phase materializations fall4092→2846, but throughput
improves only1.80% over the failed-regrowth mixed observation. These are separate
single launches, not a repeated/controlled performance acceptance comparison.

All1152 output token IDs match the previous mixed run. Short phases1024 logprob
pairs are exactly equal. Middle-phase128 logprobs have maximum difference0.04413,
above0.01: its batch schedule changed from B1:24/B2:51 to B1:30/B2:48. Do not
relax the tolerance or claim full mixed correctness from matching token IDs.
A schedule-matched native/reference run is still required for the middle phase.

Remaining decision-rule edge case to check before broader workloads: the future
horizon is for prospective growth, not a reason to undo already-resident loans
at a request's final step. The current three phases did not decide on exactly
zero remaining steps (final decisions used7/7/6), so this boundary is not proven
by this model run. Retention must remain subject to real KV pressure, not an
accidental zero-horizon cost comparison on sunk allocation cost.

## Final-step boundary correction and native reference

A CPU controller-level regression reproduced the final-step problem: with49
resident slots, remaining_steps0 and no KV pressure, the old caller passed a
zero benefit horizon and targeted16. The controller now supplies the expanded
future horizon only when slot_capacity equals base_slots, i.e. prospective new
growth. Existing loans retain historical sunk-cost treatment; actual pressure
still overrides them. Three controller cases passed after the one-condition
fix: keep existing49, prohibit new growth from16 at zero remaining work, and
recall existing49 under real queued shortage. No whole-model rerun for this
boundary-only change.

The horizon64 event artifact shows that middle-phase late submission occurred
at1.000033s, after receiving short token15 (0.975819s). Short token16 arrived at
1.023632s before the long request's first output. A single eager/native reference
was therefore launched with the same mixed workload but the middle trigger
changed from1second to `late_after_output_tokens:15`. Fresh output directory:
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_native_trigger15_01`.
The invocation uses `--arm native`; effective arguments remove all LayerKV
options and disable LayerKV, including the local expert replay optimization.
This is a progress-matched correctness probe, not matched arrival-time throughput.
An output-token trigger alone does not prove an identical scheduler trace;
inspect the actual resulting comparisons before interpreting it as a reference.

### Native reference completed with exact outputs

The native trigger15 child exited0. All three phases match the horizon64 adaptive
artifact exactly:512/128/512 token IDs and logprobs (1152total), finite values,
maximum logprob error0 in each phase. At the long request's first streamed output,
both arms had received16 short-request tokens. This closes the outstanding
bounded mixed-sequence native output comparison without relaxing the0.01
tolerance or rerunning adaptive/fixed.

The reference changes arrival triggering and disables LayerKV, so its timings
are not residency-policy performance evidence. The horizon64 model artifact
predates the final-step caller guard; that guard is validated by the three
controller regression cases, not by this native run (which has no LayerKV).

Next performance gate: matched mixed workload against reasonable fixed splits,
with the same common optimizations and physical-memory accounting. Preserve
phase/cold-transition costs and report aggregate tokens divided by total phase
makespan, not an average of phase speedup percentages. No20% claim follows from
the mechanism or exact-reference gates alone.

## Mixed fixed16 comparison: no demonstrated benefit

Added only the missing fixed child in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_fixed16_01`. Same horizon64 mixed
command, changing directory and `--arm fixed`. Effective engine args differ
only in fixed/adaptive policy and extra_slots0/48. The fixed child exited0.
No adaptive rerun, alternative-split matrix or profiler was launched.

| Phase | Fixed16 tok/s | Adaptive tok/s | Fixed TPOT ms | Adaptive TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| Short (cold) | 66.190 | 68.757 | 94.737 | 89.939 |
| Long arrival | 24.457 | 23.227 | 61.514 | 64.394 |
| Short again | 102.449 | 97.012 | 62.808 | 61.494 |

Aggregate1152tokens/summed phase makespan: fixed64.119tok/s versus adaptive
63.175tok/s, diagnostic difference-1.47%. Excluding the first phase yields
62.553 versus59.323tok/s (-5.16%), but the remaining shape transitions are not
steady-state warm intervals. Final-short TTFT is1.040s fixed versus about1.409s
adaptive, so a small decode TPOT advantage does not imply end-to-end gain.

All phase memory snapshots match field-for-field and shared physical bytes
remain478150656. Fixed keeps16slots, all guards pass, stale/alignment/physical
failure counters0, stable pointers. Short-phase token/logprob pairs are exact.
Long phase has token divergence and max logprob difference0.21089; batch
histograms differ (fixed B1:28/B2:49, adaptive B1:30/B2:48). Therefore the aggregate
timing comparison is NOT a valid performance-acceptance result. Adaptive already
has an exact native reference at its observed progress; fixed's differing
schedule still needs its own reference. Do not assume numerical drift merely
because batch histograms differ or because guards pass.

The evidence does not support20% mixed-workload improvement, even before that
correctness gap is resolved. Next actions: progress-matched fixed native output
check, then isolate transition/TTFT cost using available phase counters before
any new profile or larger experiment family. A claim based only on the favorable
short B8 result would not satisfy this mixed-workload research contract.

## Fixed native reference completed; backing-cache hypothesis

One native child used the same mixed sequence with middle
`late_after_output_tokens:14`, based on the fixed event artifact: token14 had
arrived before late submission and token15 before long first output. Directory
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_native_trigger14_01`, arm native;
other options match the trigger15 native command. Process exited0.

All fixed/native1152 token IDs AND logprobs match exactly (512/128/512 pairs),
finite, max error0 for each phase. Both have short-progress15 at long first
output. Adaptive separately matches its trigger15 native reference exactly.
Thus both observed schedules now have native-output evidence; their cross-arm
last-three-token divergence is reproduced by their respective native runs, not
merely dismissed as floating-point noise. The strict cross-arm equality check
still fails, and these native timings are not common-arrival performance data.
The original mixed comparison still shows no20% gain (-1.47% diagnostic aggregate).

Existing phase3 counters, without another profile:

| Counter delta | Fixed16 | Adaptive regrowth |
| --- | ---: | ---: |
| Expert materializations | 4093 | 2846 |
| Materialize ms | 523.757 | 340.514 |
| Main-stream wait ms | 298.380 | 230.472 |
| Backing cache hits | 4093 | 2191 |
| Backing cache misses | 0 | 622 |
| Backing cache evictions | 0 | 650 |
| LayerKV bookkeeping elapsed ms | 204.278 | 405.256 |

Bookkeeping elapsed time includes forward-end work/waits, not exclusive Python
CPU cost. Counters aggregate prefill+decode, so they cannot attribute the TTFT
gap to cache misses directly. Both arms have195 cumulative graph captures by
phase3 end; capture count alone also does not measure capture time variation.

Next hypothesis: regrowth increases resident-copy cache pressure and repeated
CPU backing reconstruction erodes saved expert transfers. Inspect reuse/lifetime
semantics before changing the cache; an immutable selected-layer CPU backing
option may avoid these misses without preloading all40 layers. The existing
`expert_cpu_backing_mode=all` preloads every discovered layer and must not be
enabled blindly as a small-memory optimization. Any controlled change must
report its host-memory cost and apply symmetrically to fixed/adaptive.

Counter semantics were checked at `expert_hooks.py`: backing-cache hit/miss is
recorded when replacing a GPU slot, for whether the EVICTED expert already has
a CPU copy. A miss schedules D2H reconstruction; it is not directly an incoming
expert H2D-cache miss. The hypothesis is specifically avoiding redundant D2H of
immutable weights, not treating these622 misses as622 missing model weights.
