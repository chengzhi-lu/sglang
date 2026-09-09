# KVC prefetch capacity boundary (2026-09-07)

## Scope

This note isolates the long-context/small-batch side of the current goal:
keep KVC residency ahead of expert residency, while testing whether later KVC
layers can be loaded on the copy stream while an earlier layer computes.

Model and runtime were held constant:

- Qwen3.6-35B-A3B, FP16
- snapshot: `/mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
- 8 requests, 4096 input tokens, 64 output tokens
- `per-layer-arena`, `async-deadline`, block size 2048
- expert-heavy precondition, adaptive lend arm, 40 MB reclaim target
- launcher: `python3 scripts/layerkv_local.py`

## Evidence

The previous matched lend artifact used `scratch_tokens=2048`:

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_expertheavy_kvcfirst_02/lend.json`

It preserved the B8 schedule, but the 2048-token demand was split into two
1024-token buffers. Consequently each round reported:

- `virtual_kvc_prefetch_count=0`
- `virtual_kvc_prefetch_fallback_sync_count=382`
- `virtual_kvc_reload_sync_count=382`
- output throughput: 21.404 and 25.507 tok/s

The focused capacity probe used `scratch_tokens=4096`:

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_expertheavy_kvcfirst_scratch4096_01/lend.json`

The two 2048-token buffers enabled the existing async path:

- `virtual_kvc_prefetch_count=18`
- `virtual_kvc_reload_async_count=18`
- `virtual_kvc_prefetch_ready_before_use_count=18`
- `virtual_kvc_prefetch_fallback_sync_count=0`

However, the extra 2048 physical KVC slots changed admission under the same
`max_total_tokens=32768`: the decode histogram became B7 plus B1 instead of
B8, and the round throughput was 16.354 tok/s. This is not a matched
performance win and must not be used as a 20% result.

## Rejected fallback

An experimental fallback sent the next layer directly to its per-layer
physical arena when no virtual scratch buffer was available. It issued five
physical prefetches per round and reduced synchronous materializations, but
its responses diverged from the matched B8 native reference. The experiment
was kept only as a diagnostic artifact:

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_expertheavy_kvcfirst_fallback_01/lend.json`

The fallback was removed from the runtime. Physical per-layer prefetch must
not be re-enabled until its metadata/ownership path has a schedule-matched
correctness test.

## Current decision

The existing async virtual prefetch implementation is functionally reachable,
but its current capacity tradeoff is explicit:

```text
scratch_tokens < 2 * current_layer_demand  -> synchronous materialize
scratch_tokens >= 2 * current_layer_demand -> async virtual prefetch possible
```

For this long-context workload, increasing scratch is not currently an
optimization because admission loses one request. The next safe optimization
is therefore to make the scheduler/allocator account for a bounded prefetch
window without changing the physical KVC budget, then validate metadata and
per-arm native equivalence before measuring throughput.

## Safe single-buffer physical successor (2026-09-07)

The first bounded implementation is now connected at the single-buffer
boundary. When the current layer must synchronously materialize into the full
virtual scratch buffer, the runtime submits only the next layer's offloaded
entries to the existing per-layer physical arena with `strict=False`. The
current layer keeps its scratch mapping; the next layer waits on its own reload
event at the normal attention use point. The path does not allocate extra
physical KVC capacity and is separate from the rejected run-shadow fallback
above.

Focused regression coverage passed (`15 passed` in
`test_layer_prefetch_overlap.py`). A fresh Qwen3.6-35B-A3B FP16 lend run with
the same B32-short/B8-long settings as the earlier capacity probe recorded:

- long phase: `virtual_kvc_prefetch_fallback_sync_count=20` versus `764` in the
  earlier scratch-only lend artifact;
- `kvc_layer_prefetch_issue_count=18` and `kvc_layer_prefetch_token_count=36864`;
- actual decode batch stayed B4, with KVC/expert/zero-reconstruct guards and
  `comparable` all true.

This is mechanism evidence, not a 20% claim. The fresh lend long-phase
throughput was `20.255 tok/s`; the earlier artifact was `20.841 tok/s`, and
the launches were not a same-source matched pair. The extra physical reload
work is therefore not yet accepted as an end-to-end win. The next change should
be a runtime cost/overlap gate for this fallback, using the measured reload
EWMA and pending-copy window, before another model run.

## Runtime cost/overlap gate (2026-09-07)

The single-scratch fallback now keeps at most two unconsumed per-layer reload
events on the copy stream. After a per-layer reload EWMA is available, it
estimates the successor H2D duration from the selected token count and rejects
the fallback when that estimate exceeds an observed per-layer model window.
Cold-start calls without either measurement remain eligible so the runtime can
calibrate itself. The gate applies only to the new `allow_virtual_scratch`
fallback; the existing ordinary physical per-layer prefetch path is unchanged.

New counters distinguish pending-window skips, measured deadline rejects, and
cold/unprofiled allows. This is a scheduling guard, not a throughput result;
the next model check should compare fallback traffic, ready-before-use stalls,
and total output throughput against a same-source fixed control.

## Lightweight model-window measurement (2026-09-07)

The first model check showed that the overlap gate was otherwise always using
the cold/unprofiled allowance in normal runs: the detailed LayerKV profiler is
disabled for performance measurements, so its decode-forward timing fields are
zero. The runtime now lazily enables a lightweight decode-only host-window
sample after the physical successor path is reached. It records cumulative
model-forward time and count without CUDA synchronization or detailed per-layer
profiling, and the gate uses those samples before falling back to the old
detailed-profile fields.

Focused coverage passed (`17 passed` in `test_layer_prefetch_overlap.py`). The
post-fix Qwen3.6-35B-A3B FP16 artifact is:

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_expertheavy_kvcfirst_gate_04/lend.json`

Its long-phase round reported:

- `kvc_layer_prefetch_model_forward_ms=5095.923`
- `kvc_layer_prefetch_model_forward_count=63`
- `kvc_layer_prefetch_overlap_window_ms=8.262`
- `kvc_layer_prefetch_estimated_copy_ms=1.363`
- `kvc_layer_prefetch_unmeasured_allow_count=0`
- `kvc_layer_prefetch_deadline_reject_count=0`

Thus the measured gate is now active and the estimated copy fits the observed
window; this workload does not exercise a rejection case. The run kept the B8
decode shape, with `expert_ready_miss_stall_ms=152.8` and
`layerkv_python_overhead_ms=182.1` in that round. A CPU-only comparison against
the matched native B8 reference passed exact output IDs, log-probability
tolerance, decode shape, KVC/expert guards, and pending-transfer checks.
The result is correctness/mechanism evidence only, not a 20% speedup claim.

## Long-small exact expert successor probe (2026-09-07)

The first version of the long-small exact expert gate checked only whether
fewer than two KVC reload events were pending. A matched B32-to-long-B8 pair
was used to exercise that branch:

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_exactlong_pair_01`

The long phase did issue exact expert successors (`364` IDs in the lend arm,
`147` in fixed), and the paired outputs, physical pool and guards remained
valid. It did not improve the KVC-pressure phase: fixed/lend throughput was
`10.425/6.364 tok/s`, versus `10.931/6.365 tok/s` in the prior no-exact
long-window artifact. The lender's ready-before-use count was high, but the
additional expert churn did not reduce end-to-end time.

The runtime gate is therefore tightened so an existing physical KVC offload
reserves the copy window even after its current reload has been consumed.
Long-small exact expert prefetch remains available only for a genuinely KVC
idle window; it cannot consume capacity while KVC demand is resident in the
offload set. The new focused tests cover both the idle-window allow and the
existing-offload rejection. This is a policy correction based on matched
evidence, not a throughput claim.

## Partial target for soft context pressure (2026-09-08)

The previous decode policy treated every positive
`context_demand + reserve - capacity` deficit as a request to shrink experts to
`kv_min_slots`. That was too coarse: the B32-to-long workload had only a small
soft deficit, but the minimum target still caused tens of thousands of expert
materializations. The runtime now converts only the residual soft deficit into
physical KV pages and reuses `_select_kv_overflow_target` to choose the smallest
page-feasible expert target. The old minimum-target behavior remains the
fail-safe when physical layout information is unavailable.

The budget also distinguishes `context-pressure-partial` from hard
`context-pressure`. If queued admission or a real KV shortage wins the same
decision, the full reclaim request is preserved; the partial token cap is used
only when the budget selected the matching soft-pressure target. Focused budget,
admission, and shared-expert tests passed (`40 passed` for the first two suites,
plus the existing shared-expert coverage). No new model run is claimed here:
the prior long pair already had an active overflow segment, so this new
no-active-overflow branch was not reachable in that artifact.

## Scheduler-visible physical admission credit (2026-09-08)

The long/small admission path had one remaining capacity mismatch. The
scheduler's early `batch_is_full` guard compared a queued request only with
the native allocator's `available_size()`. With the per-layer physical arena,
valid common KVC locations can already be owned by LayerKV while not yet being
returned to the native free-page tensor. The prefill adder already exposed
this as scheduler credit, but the early scheduler guard did not.

The guard now evaluates:

```text
effective physical capacity = native free tokens + scheduler-visible arena credit
```

It clears the prefill-full state when that capacity is enough and recalls or
activates expert-backed KV only for the residual shortage. The PrefillAdder
uses the same accounting, so the early admission decision and the actual
allocation decision cannot disagree on this point. Focused scheduler,
admission, residency-budget, and shared-expert coverage passed (`80 passed`).

The single-arm FP16 Qwen3.6-35B-A3B validation used the same 4900-token,
three-request workload as the preceding lender-only artifact:

- previous: `/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_06/lend.json`
- post-fix: `/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_07/lend.json`
- throughput: `6.267 -> 7.573 tok/s` (`+20.8%`)
- third-request TTFT: `23.174 -> 17.484 s`
- decode batch mean: `1.500 -> 2.953`; post-fix histogram was B3 for 62 of
  64 decode steps
- post-fix admission recall counters stayed zero, while the final scheduler
  probe reported `4534` physical-credit tokens and the overflow tail held
  `4095` tokens

This is a same-arm before/after mechanism result, not a matched fixed/lend
comparison. It demonstrates that the 20% local target is reachable for this
long-context admission case, but does not establish a general 20% gain across
the adaptive residency workload. The next validation should be one matched
fixed/lend pair using this scheduler-visible capacity path.

## Matched pair and route-aware expert floor (2026-09-08)

The requested matched fixed/lend pair was then run with the same model,
physical pool, request arrival, and 4900-token workload:

`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_pair_01`

The pair passed physical-pool matching, exact token/log-probability guards, and
the KVC/expert ownership checks, but the lend arm was slower:

| arm | output tok/s | decode mean | TPOT ms | final expert slots |
|---|---:|---:|---:|---:|
| fixed16 | 10.193 | 1.500 | 121.6 / 121.6 / 84.4 | 16 |
| lend | 7.518 | 2.953 | 368.8 / 368.8 / 127.7 | 8 |

The lend arm's B3 decode shape was not beneficial because Qwen3.6 routes
`top_k=8`; shrinking to 8 resident slots caused `8141` expert materializations
and repeated route-group churn. The controller therefore gained a
model-independent route-aware floor derived from `top_k` and the scheduler's
actual request batch: a single request may donate down to one route width,
while a multi-request batch retains up to two route widths (capped at the base
allocation). The floor is enforced after the budget's hard `kv-pressure`
branch as well, so that branch cannot silently fall back to `kv_min_slots`.

Focused coverage passed (`86 passed`). A same-configuration lend rerun is:

`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_10_routefloor_final/lend.json`

It kept `current_expert_slots=36`, avoided the 8-slot overflow state, reduced
expert materializations to `6476`, and improved lend throughput to `8.966
tok/s` (`+19.3%` versus the old lend arm). It still remained `12.0%` below the
matched fixed arm, with decode shape B2/B1 rather than B3. Therefore the 20%
target is still not accepted. The next optimization should make the
long-context admission decision reclaim only the extra expert tail while
retaining the route floor, instead of choosing between full 8-slot churn and
no KV overflow.

That follow-up is now implemented as a long-context base floor: when
`context_live_tokens > 2048` and the current `batch * top_k` fits the base
allocation, adaptive growth is capped at `base_slots`; any already existing
extra tail remains eligible for the normal physical KV reclaim path.

The final same-workload validation artifact is:

`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_13_ctxbase_final/lend.json`

It used only E=16 for the selected layer, kept `current_expert_slots=16`,
reduced expert materializations to `6561`, and reported `10.687 tok/s`. Against
the matched fixed16 arm (`10.193 tok/s`) this is a `+4.85%` gain; against the
old unconstrained lend arm it is `+42.16%`. Exact output and physical guards
remain valid. The run still decoded as B2/B1 rather than B3, and no
expert-to-KV overflow activation occurred, so the result does not yet prove
the intended long-context admission benefit or the 20% target. The next
mechanism to isolate is scheduler admission of the third long request while
keeping the base expert route resident, rather than further changing the
residency floor.

## Common-free publication and block-aligned expert admission (2026-09-08)

The previous admission trace showed that the scheduler had counted per-layer
common-free KVC locations as credit, but had not returned those locations to
the native allocator before `PrefillAdder` performed its hard total-token
check. The residual shortage then reached expert admission as a sub-block
request (`488` tokens in the matched run), and the physical overflow selector
correctly found no page-feasible target, leaving E=16 and delaying the third
request.

The admission path now has two bounded steps:

1. Compute the candidate request's full reservation (`context + clipped
   max_new + page overhead`) and release only the residual common-free KVC
   locations to the native allocator.
2. If a physical shortage remains, round the expert-backed KV request to the
   configured `kvc_block_tokens` (2048 in this run), then select the smallest
   page-feasible expert target. The admission-only floor remains one `top_k`
   route; decode-time multi-request residency still uses its two-route floor.

Focused scheduler/admission/residency coverage passed (`89 passed`). The
matched-configuration lend validation is:

`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_18_blockaligned/lend.json`

- workload: Qwen3.6-35B-A3B FP16, TP1, 3 x 4900 input / 64 output, E=16 base,
  `kvc_block_tokens=2048`, retain-across-requests enabled;
- decode shape: B3 for 62/64 steps, B2 for 1, B1 for 1;
- admission: residual 431 tokens, block request 2048, target E=9, active KV
  overflow 2048 tokens, 7 expert slots reclaimed;
- throughput: `10.778 tok/s`, compared with the matched fixed16 reference
  `10.193 tok/s` (`+5.73%`), and the old unconstrained E=8 lend arm
  `7.518 tok/s` (`+43.36%`);
- correctness: exact token/log-probability match against both fixed and old
  lend references; max log-probability delta `0`;
- physical guards: KVC, expert, ownership, stale-entry, alignment, and
  physical-failure guards all pass.

This validates the long-context admission mechanism, not the general +20%
target. The remaining cost is B3 decode: the first two requests report about
`240.6 ms` TPOT and selected-layer materialization is `6948` events. The next
optimization should reduce repeated expert materialization for the B3 route
working set while preserving E=9 and the exact-output guard.

## Physical-page admission and post-MoE prefetch probes (2026-09-08)

Admission now queries the actual sparse-VMM destination pages before applying
the scheduler block fallback. This preserves smaller requests when a physical
page boundary is cheaper than `kvc_block_tokens`; a unit test covers the
1024-token boundary with a 2048-token scheduler block. The matched Qwen3.6
FP16 workload does not expose such a smaller boundary: the 2 MiB VMM page
corresponds to a 2048-token overflow here. The page-aware run
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_21_pagealigned`
therefore matched `lend_18` structurally (`request_tokens=2048`, E=9, 20
overflow pages, 6948 materializations). Its single-run 11.072 tok/s is not
counted as a speedup.

A decode-only post-MoE exact-successor prefetch probe used the completed MoE
event to reuse current slots. It increased useful exact prefetch IDs from 116
to 707 and reduced ready-miss stall from 83.2 ms to 53.6 ms, but increased
materializations from 6948 to 7113 and H2D busy time from 986.4 to 1021.6 ms;
the 10.825 tok/s result was not a reliable gain. The probe artifact is
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_softpressure_partial_4900_lend_22_postmoe_prefetch`;
the post-MoE change was reverted. The next useful work remains reducing route
group churn without increasing total expert transfer volume.

## Exact first-fit grouping index (2026-09-08)

The E=9/top-k=8 path was also spending CPU time linearly scanning all earlier
token groups. `group_token_experts` now recognizes the common
`capacity = route_width + 1` shape and indexes the exact first-fit candidates
by route subsets. It still selects the smallest original group index, so group
membership, token order, native MoE reduction boundaries, and invalid-route
handling are unchanged. Other capacities and mixed route widths keep the
original linear fallback.

The new indexed helper passed deterministic growth/stale-posting coverage and
500 randomized exact-equivalence cases against the old first-fit reference.
On a CPU-only synthetic 2000-row, 128-expert, top-k=8 workload, grouping fell
from about 193 ms to 40 ms. This is grouping CPU wall time, not end-to-end
throughput.

A single matched fixed/lend validation was run with the existing
`lend_18_blockaligned` settings. It remained `valid=true`, exact in output IDs
and logprobs, and all physical/KVC/expert/ownership guards passed. The lend arm
kept E=9, 2819 token-chunk calls, 6948 expert materializations, 2937 H2D batch
submissions, and 116 exact-prefetch IDs; its 10.794 tok/s versus the historical
10.778 tok/s is within single-run noise and is not counted as a throughput
claim. The candidate artifact is
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_groupindex_4900_pair_01`.

This removes one CPU-side scaling cost without changing the current transfer
volume. The next material optimization remains reducing E=9 route-group churn
or hiding its H2D traffic, with no additional model matrix launched yet.

## Bounded route reuse for E=9 (2026-09-08)

The low-capacity long-context adaptive branch now uses a bounded 32-group
lookahead. It selects the next group by resident/missing expert overlap while
keeping the token-group boundaries and row order intact; it does not define or
persist a dataset-specific expert set. The full reuse scan remains available
for larger capacities, while this bounded path avoids its full-forward
O(groups^2) CPU scan.

The one matched-settings model check used the Qwen3.6-35B-A3B FP16 snapshot
and the existing 3-request/4900-token E=9 workload. The lend arm selected
`window-reuse` on 62 batches and passed exact output/logprob, physical-pool,
KVC, expert, and ownership guards. Against the historical input-order lend
arm, expert materializations fell `6948 -> 6806`, H2D batches `2937 -> 2936`,
H2D bytes `43,731,910,656 -> 42,838,523,904`, and ready-miss stall
`83.2 ms -> 73.1 ms`. The fresh fixed/lend pair was `9.406/10.929 tok/s`
(`+16.19%`); this is below the 20% target and is not treated as a general
throughput claim because the fixed control was a single run and the reference
lend arm came from a separate artifact.

Artifact: `/mnt/vdb/chengzhi/layerkv_fp16_20260908_windowreuse_4900_pair_01`.
The next optimization should target the remaining on-demand H2D batches and
their overlap window, rather than widening the route-reuse scan.

## Decode route-snapshot reuse and KVC-credit probe (2026-09-08)

The current small-decode path now reuses the exact CPU route snapshot even
when CUDA grouping is skipped for a small row count. This snapshot is routing
metadata already needed by the CPU first-fit grouping path; activations remain
on GPU, and expert remap/readiness stays on the GPU/copy streams. In the
focused CUDA coverage, the decode path used CPU-known preparation for every
token chunk and no longer re-entered generic GPU demand discovery.

On the current worktree, a single lender-only 3-request/4900-token FP16 run
reported `expert_topk_gpu_remap_missing_count=5` versus `897` in the preceding
small-decode policy, while materialization/H2D volume remained about
`16887` experts / `2936` batches. This is a mechanism reduction, not an
end-to-end speedup claim; no fixed arm was rerun.

A bounded one-credit exception for physical-KVC offload was also probed: allow
one exact successor only when no KVC DMA was pending. The lender artifact
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_kvc_credit_4900_lend_01` had exactly
the same `115` exact-prefetch IDs, `16772` on-demand IDs and `2936` H2D
batches as the no-credit run, so the credit was removed. The next change should
make the exact successor enter the copy stream earlier or batch future demand,
using observed KVC/expert stream availability, rather than loosening the
KVC-priority gate without a mechanism effect.

## Successor-only future eviction hint (2026-09-08)

The previous implementation attempt extended the future-use map to every group
remaining in the lazy `window-reuse` deque. That was not a valid execution
order: adaptive ordering reselects later groups after each materialization, so
the pending deque is not the actual successor sequence. On the matched lender
workload this raised materialization to `16759` and ready-miss stall to
`826.1 ms` (`/mnt/vdb/chengzhi/layerkv_fp16_20260908_belady_windowreuse_4900_lend_01`),
and was removed.

The retained change is narrower. The lazy path uses only the already selected
immediate successor as a forward-local victim hint; the rest of the candidates
continue through the LRU chooser. Complete future positions remain enabled only
when the route execution order is materialized up front (input/indexed path),
where the order is stable. The focused route/eviction tests pass through the
repository launcher.

The matched lender probe
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_successor_windowreuse_4900_lend_01`
hit the hint on `2756` groups and `15467` slot choices, with `15918` expert
materializations, `2056.1 ms` expert H2D busy time, and `785.5 ms` ready-miss
stall. It is a mechanism result only: one run is not accepted as a general
throughput claim, and the remaining H2D/ready-stall volume is still the main
performance bottleneck.

## Dedicated cross-layer expert H2D stream audit (2026-09-08)

The scheduler path for cross-layer expert lookahead was still passing the
shared `_copy_stream` to `_materialize_experts`, even though the planner had
already created `_expert_h2d_stream`. This serialized speculative expert H2D
submission with KVC recovery. It now selects the dedicated expert H2D stream,
with the shared stream retained as a CPU/unit-test and older-runtime fallback;
the existing D2H-to-H2D dependency remains intact when a prefetch evicts a
slot.

The focused stream-routing test passed (`7 passed`). A single short-context,
large-batch lender probe exercised the changed path:
`/mnt/vdb/chengzhi/layerkv_fp16_20260908_dedicated_crosslayer_b32_lend_01`.
It issued `30` cross-layer expert tasks / `936` prefetch IDs and passed the
expert and KVC guards. Its single-run measurements were `3979` materialized
experts, `485.2 ms` expert H2D busy time, and `232.3 ms` ready-miss stall.
These are mechanism observations only; the run is not a matched throughput
claim. The next check should determine whether the newly overlapped queue
increases route churn, since the probe's materialization count remained above
the older short-batch artifacts.
