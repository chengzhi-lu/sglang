# Selected-layer chunk attribution by forward phase

Capture timing ruled out recapture as the main final-short cost (79ms,1.51%).
The selected layer still handles final-short prefill at16slots before regrowth.
Existing chunk telemetry mixed prefill/decode and could not test that hypothesis.

Extended existing opt-in shared expert chunk profiling, not residency policy:
`--profile-expert-chunks` in the concurrent driver enables the existing server
`layerkv_shared_expert_profile_chunks` option. Summary `shared_vmm.token_chunk_by_mode`
contains prefill/decode/other wall and stream timing maps, chunked invocation
count, token rows and the slot-capacity histogram at invocation. Existing total
timing fields remain. Pending GPU events carry their submission-mode label,
so later polling in another mode cannot misattribute them.

This covers run_token_chunks, not unchunked selected-layer calls or whole-model
prefill. Wall subphases and current-stream event spans are not additive exclusive
kernel costs; stream spans can include waits/submission gaps. CUDA event recording
has overhead. All such driver runs/rows are diagnostic-only, and summarization
rejects acceptance from persisted diagnostic markers even if the summarizing
CLI omits the profile flag. Default runtime behavior and admission are unchanged.

Validation: two CPU tests passed (delayed event attribution/one-time collection,
disabled subphase clock bypass), three summary diagnostic-rejection cases passed,
and the existing CUDA event-collection check was selected to validate the changed
event tuple format. No full suite or model run for this implementation turn.

Next run only the existing selected-backing,horizon64 mixed adaptive sequence
with `--profile-expert-chunks`, fresh directory. Examine phase deltas of prefill
chunk totals/slot capacities and routing/group/prepare/MoE/scatter breakdown.
Do not rerun fixed/native merely to obtain these timings, or infer a20% gain
from an instrumented throughput value. Only then decide whether prefill-time
borrowing is justified and which safety boundaries it must preserve.

## Completed diagnostic and decision

Artifact: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_chunkphase_01/lend.json`.
The process is terminal and all three round records exist. Do not rerun it.
Engine arguments are saved alongside the result. FP16, selected CPU backing,
horizon64, native-core replay limit8; mixed short/long/short workload unchanged.

Final-short values below subtract round2 counters from round3, rather than
reading cumulative totals. Units are milliseconds, wall timing only.

| Selected-layer chunk phase | Prefill | Decode |
| --- | ---: | ---: |
| Routing | 0.262 | 0.039 |
| Grouping | 4.587 | 0.234 |
| Gather | 12.351 | 6.558 |
| Prepare | 187.932 | 125.298 |
| MoE | 49.244 | 28.449 |
| Scatter | 4.425 | 2.269 |
| Total invocation wall | 261.485 | 165.464 |

Prefill is one invocation of1024 token rows at16slots. Decode chunk coverage is
only17 of63 B8 forwards:9 at16slots and8 at49slots. Unchunked forwards bypass
`run_token_chunks`; their missing timing is not zero. Pending event count is0
at every round snapshot. Round1 prefill has no chunk invocation and cannot be
used as a zero-cost or matched prefill baseline.

All1152 output IDs and output-token logprob arrays exactly match the existing
`mixed_capturetime_01/lend.json` artifact. This is instrumentation regression
evidence, not a new independent native reference. Actual decode histograms are
B8:63, then B1:32/B2:47, then B8:63. KVC/expert guards pass, stale/alignment/
zero-reconstruct violation counters are0. Final slots49, physical pool456MiB,
growth physical creates0. Timing acceptance remains disabled.

The final-short phase takes5202.166ms. Even removing the entire measured prefill
chunk invocation, with all other costs held fixed, gives only about5.29% more
throughput, not20%. This is illustrative arithmetic on an instrumented run,
not a measured or general upper bound on earlier growth: earlier growth could
also affect decode residency. It does rule out treating188ms prepare as proof
that this local optimization alone meets the objective.

### Read-only implementation audit

- `SharedExpertController.after_decode` requires interval-length decode demand
  observations. `on_schedule_batch` clears remaining-decode horizon on extend.
  Calling the existing hook earlier therefore is not a prefill benefit policy.
- Token groups are built from the current slot capacity. Any earlier growth
  must precede grouping and use the actual achievable capacity, not the64-slot
  shadow-cache target (these artifacts physically reach49).
- Donors exclude live/allocated, protected, cleanup-owned and incomplete pages.
  Selection and a second check preserve common KV headroom; exclusive claims
  precede unmapping. No optimization may publish pending eviction completion
  from this path or count private scratch as admission capacity.
- Existing full decode profiling predates native replay and selected backing.
  Its host/kernel attribution cannot be relabeled as the current bottleneck.
- The current mixed test caps running requests at8 and observes8/1..2/8.
  It demonstrates transitions, not increased batch beyond8 or the main trace
  performance requirement. Repeated runs of this fixture cannot establish that
  missing research result.

Decision: do not move the hook or launch another prefill-only model probe yet.
Keep this diagnostic as completed evidence. Prioritize a reproducible workload/
capacity plan for the actual batch-growth requirement and matched fixed splits,
including Mamba request-state capacity and the replay batch-size limit. The main
evaluation still follows `layerkv_evaluation_plan.md`; this small diagnostic does
not replace it. No runtime behavior was changed or tests rerun for this audit.
