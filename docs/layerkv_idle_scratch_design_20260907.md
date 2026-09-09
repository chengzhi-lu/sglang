# Idle scratch lending: implemented, opt-in

## Current status

The design below is now implemented behind
`--layerkv-shared-expert-lend-virtual-scratch`. The concurrent model driver
exposes the matched-arm switch as `--lend-virtual-scratch`; the fixed arm keeps
the flag disabled. The implementation reuses `SharedVMM` page lending and
keeps virtual-scratch ownership separate from the ordinary per-layer arena
free list.

## Evidence and priority

See `layerkv_grouped_budget_20260907.md`. Reducing static scratch reservation
8192 to2048 exposes enough existing physical pages for expert29 to49 at unchanged
478150656-byte shared backing. This is useful capacity evidence, but adaptive
throughput increases only about2.2% across those separate launches. It does not
justify claiming20% as a general result.

The first matched physical-lending run uses Qwen3.6-35B-A3B FP16, 32
simultaneous requests, 128 input tokens, 32 output tokens, 8192 scratch tokens,
and two rounds. It passes token, logprob, stream-timing, physical-pool, KVC,
expert, and VMM ownership guards. Scratch lending moves six 2 MiB pages into
the expert tail (16 to 18 expert slots). The first round gains 17.57%, the warm
second round gains 4.49%, and the aggregate two-round client throughput gains
11.85%. This validates the mechanism and its decode-path effect, but is not a
20% acceptance claim.

The follow-up `extra-slots=8` run shows that simply exposing more scratch pages
is not the main bottleneck: the warm gain is 4.23% (176.02 versus 168.87
tokens/s), despite reaching 24 expert slots and lending 24 pages. The artifact
is `/mnt/vdb/chengzhi/layerkv_fp16_20260907_scratch_lend_b32_extra8_01`.

The next diagnostic run combines that capacity with the opt-in native MoE
replay path. For both fixed and lend arms, 39 unaffected resident MoE layers
(layers 1--39; selected layer 0 remains outside the graph) were captured and
replayed. Warm throughput is 184.83 tokens/s fixed and 191.88 tokens/s lend,
for a 3.81% matched-arm lending gain; all token, logprob, physical-pool, KVC,
expert, and ownership guards pass. The graph reports 2340 replays, 78
fallbacks, and no recapture. This is useful directional evidence, but the
artifact remains diagnostic-only because graph workspace accounting is
`torch_allocated_delta_not_total_physical_budget`; therefore it is not a
validated 20% result. The artifact is
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_scratch_lend_b32_extra8_graph_01`.

The layer20 cross-layer prefetch probe then reached the previously inactive
short-context trigger: the warm lending gain was 8.72% with 29/30 cross-layer
issues and 424/568 ready-before-use hits in fixed/lend. Adding native replay
and increasing the lend target to 32 slots improved the warm matched gain to
14.41% at `extra-slots=16`, but the graph artifact remains diagnostic because
its workspace is not included in total physical-budget accounting. The
artifact is
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_crosslayer_layer20_graph_extra16_01`.

The follow-up `extra-slots=32` probe is a negative capacity result: the lend
arm reached only 36 slots in round1 and 41 in round2, performed 20/25 growth
transactions, and warm throughput fell to a 0.13% gain. The 60 available
scratch pages were consumed one row at a time because the requested target
exceeded the scratch-funded capacity, causing repeated donor scans and
control overhead. Growth now batches all complete rows currently fundable by
scratch, while leaving any residual ordinary-KV growth for a later scan. Do
not use `extra-slots=32` as a performance point until this control path is
revalidated.

An adaptive/reuse exact-next-group prototype was tested once in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_crosslayer_layer20_graph_extra16_exact_reuse_01`.
It advanced the reuse generator after the current group was prepared and
submitted the next group's H2D before the current MoE call. The diagnostic
round-2 lend throughput was 184.024 tok/s versus 159.503 fixed (+15.37%), but
the paired outputs were not exact (`max_logprob_delta=1.4461`), so the summary
was invalid despite the physical/guard checks. Lend exact prefetch issued 120
groups / 792 IDs and increased materialization batches to 602; the source
change was reverted. This failure motivated the slot-lifetime ownership
protocol described below; the adaptive path is now re-enabled only with those
guards.

## Safe adaptive successor prefetch

The ownership protocol is now implemented for the adaptive/reuse path. The
next group is selected after the current group's preparation and its exact
expert IDs are submitted before the current MoE call. For every subsequent
group, both copy streams wait for the previous group's MoE completion event.
Pending H2D destination slots remain unavailable to the slot chooser until
their ready event is collected, and an asynchronous D2H eviction publishes an
event dependency to the H2D stream before the slot can be refilled. Completed
copies publish residency only when their original logical-to-slot mapping is
still current.

The focused regression covers the ordering, stale-publication guard and
pending-slot reuse guard. A same-source Qwen3.6-35B-A3B FP16 B32/layer20
sanity pair used 32 requests, 128 input tokens, 32 output tokens, 8192
scratch tokens, a 16-slot fixed arm and a 32-slot lend arm. Both arms had
physical bytes `478150656`; output IDs/logprobs, KVC/expert/ownership guards
and pending-clear all passed. Warm round-2 throughput was `144.946` fixed
versus `184.920` lend, a strict matched gain of `27.58%`. The lend run issued
118 exact groups / 777 exact IDs and recorded 133 D2H-to-H2D event
dependencies.

This is a single-launch-per-arm, short-context/large-batch mechanism result.
The native-MoE graph workspace is still not included in total physical-budget
accounting, so the pair remains diagnostic rather than a general 20%
acceptance claim. Independent repetitions and the long-context/KVC-priority
regime are still required.

## Minimal interface

Extend the existing `SharedExpertController` module; do not create another
allocator. Its donor records need explicit source ownership: ordinary per-layer
arena versus private virtual scratch. A proposed `restore_scratch()` interface
must return only after all borrowed scratch pages are mapped back and outstanding
expert users/copies have finished. Initially it can reuse full expert-tail recall;
selective tail recall is a later optimization, not a correctness prerequisite.

Maintain a separate scratch-loan ledger. Current `recall()` publishes every
`blocked` location into the ordinary overwrite free list. Applying that behavior
to scratch would create duplicate ownership/admission credit. Scratch restoration
must never publish scratch locations to ordinary free lists or count them as
common native admission capacity. Existing exact before/after admission credit
calculation remains authoritative.

## Borrowing eligibility and restore-before-use

- Explicit opt-in, per-layer backend, shared physical arena, no overlap/graphs as
  already required. No changes to default scratch size or physical buffer size.
- Offer only whole mapped pages entirely owned by the private scratch reservation;
  retain page0 and partial edges. Never infer ownership from a free-token count.
- Require no current offloaded KV demand and no pending eviction/reload or virtual
  materialization that can produce/use scratch demand. Check authoritative state
  without calling `_has_offloaded_kvc_entries()` as a supposedly pure predicate:
  that helper calls `_restore_impossible_per_layer_offloads()` and can mutate state.
- Existing VMM lending synchronizes before unmapping. Invalidate scratch caches
  only after proving no pending materialization; never clear pending work to make
  lending eligible. Failed transfers restore original ownership.
- Restore before scratch cache lookup/attention use, not just before H2D copies.
  `_prepare_virtual_kvc_attention()` calls `_ensure_virtual_scratch()` before
  lookup. Restoration there covers the ordinary attention path.
- Also protect direct `_materialize_virtual_kvc_sync()`,
  `_issue_virtual_kvc_prefetch()` and `_issue_all_virtual_kvc_prefetch_after_layer()`
  write entrypoints. The last one batches reloads and cannot be assumed to pass
  through the single-prefetch function. Restore before selecting buffers or
  constructing host-store transfer requests.
- A KV use overrides expert retention, including the diagnostic admission policy
  `retain`. Keep scratch VA/pointers stable. Scratch contents are disposable only
  while no cached/offloaded consumer is valid; restore mapping then reload data.

## Focused validation plan

Reuse current VMM/controller CUDA fixtures and output references. Do not rerun a
capacity matrix. Before a full model launch, targeted tests must cover:

1. Real physical scratch pages fund expert growth, with stable addresses and no
   additional physical creates; in-flight KV states prohibit borrowing.
2. Restore through the actual synchronous/single-prefetch/batched-prefetch entry
   interfaces precedes host-store writes and invalidates stale cache references.
3. Scratch recall contributes zero ordinary admission credit; arena loan recall
   still returns genuine common credit. Mixed loans preserve both owner types.
4. Mapping failure and repeat restore preserve ownership and immutable expert
   backing. No free-list publication of private scratch on cleanup or rollback.

The focused physical test and the model run now cover the first and third
items above. The remaining evidence gap is an end-to-end long-context run that
forces scratch restoration through an actual KVC use point, plus selective
recall for mixed ordinary/scratch loans. Use existing effective model arguments
and deterministic inputs. Additional runs require a concrete failing
hypothesis, not an unexplained performance fluctuation.
