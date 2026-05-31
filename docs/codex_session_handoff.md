# Codex Session Handoff

Date: 2026-05-31

Repository: `/sgl-workspace/sglang`

Branch: `layerkv/v0.5.12-integration`

Upstream: `chengzhi-lu/layerkv/v0.5.12-integration`

## Current Focus

LayerKV integration work is the active context. The staged changes cover
LayerKV CoResid runtime behavior, scheduler integration, KVC/expert validation
scripts, benchmark config, and local Codex agent skill files.

## Git State Before Push

The branch was ahead of upstream by one existing commit:

- `41331baa1 Optimize LayerKV CoResid dynamic pressure`

Additional staged changes were committed in the same session before pushing.

## Useful Validation Commands

Run targeted LayerKV checks before relying on performance or correctness
claims:

```bash
python3 scripts/layerkv_smoke.py
python3 scripts/layerkv_kvc_validation.py
python3 scripts/layerkv_expert_validation.py
python3 scripts/layerkv_pd_coresid_trace_replay.py
```

For wider repository checks:

```bash
pytest test/registered/unit/ -v
pre-commit run --all-files
```

## Resume Notes

- Start by checking `git status --short --branch`.
- Compare against the pushed upstream branch before adding new work.
- Keep LayerKV-specific validation first; avoid unrelated SGLang refactors.
- Important files in this handoff include:
  - `python/sglang/srt/layerkv/runtime.py`
  - `python/sglang/srt/managers/scheduler.py`
  - `python/sglang/srt/managers/schedule_batch.py`
  - `scripts/layerkv_*`

## CoResid Runtime Issues From WildChat E2E

Recorded: 2026-05-31T06:45:27Z

Command context:

```bash
PYTHONNOUSERSITE=1 /data/wenyan/conda/envs/moe-runtime/bin/python \
  scripts/layerkv_policy_eval.py \
  --model-path /data/hf-cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --output-dir /data/wenyan/tmp/layerkv_e2e_wildchat_default_vs_coresid \
  --gpu 0 --backend server --workload context-heavy \
  --policies full_residency coresid \
  --runtime-profile optimized --profile-detail --log-level info
```

Dataset/workload:

- WildChat-1M from `/data/hf-cache/hub/datasets--allenai--WildChat-1M/blobs`.
- 8 real requests, input token range 28886-31938, mean 30189.875.
- `policy_eval.csv`: `/data/wenyan/tmp/layerkv_e2e_wildchat_default_vs_coresid/policy_eval.csv`.
- Prompt metadata: `/data/wenyan/tmp/layerkv_e2e_wildchat_default_vs_coresid/fig4_prompt_metadata.json`.

Observed result:

- SGLang Default: valid, wall 16464.9 ms, decode0 1029.1 ms, throughput 7.77 tok/s, mean e2e 15730.4 ms, p95 e2e 16425.7 ms.
- CoResid: invalid only because of comparability guard, wall 24532.8 ms, decode0 1533.3 ms, throughput 5.22 tok/s, mean e2e 21909.0 ms, p95 e2e 24494.2 ms.
- CoResid was about 49% slower on wall/decode0, 33% lower throughput, 39% slower mean e2e.

Primary runtime problems to fix one by one:

1. Expert installation remains in progress during measured decode.
   - `comparability_reason=INSTALL_IN_PROGRESS_NOT_COMPARABLE`.
   - `expert_install_state=installing_slots`.
   - `expert_install_completed_layers=15`, `expert_install_pending_layers=33`.
   - `expert_install_not_comparable_step_count=15`.
   - Treat this first as a runtime scheduling/warmup problem, not a reclaim-target problem.

2. Controller overhead is large on the decode path.
   - `layerkv_python_overhead_ms=3693.0`.
   - `profile_controller_per_decode_step_ms=230.8`.
   - This is large enough to dominate the 16-token decode segment.

3. Expert slot install and CPU copy are blocking.
   - `profile_install_expert_slots_ms=1279.9`.
   - `profile_copy_expert_to_cpu_ms=1233.6`.
   - `expert_install_blocking_ms` is roughly the same scale as install time.
   - `layerkv_copy_stream_busy_ms=0.0`, `layerkv_copy_event_record_count=0`, `layerkv_copy_event_wait_count=0`, so this run did not meaningfully overlap copy work through the LayerKV copy stream.

4. KVC selection path spends time even when no KVC work is executed.
   - `profile_kvc_select_evict_ms=1573.2`.
   - `profile_kvc_evict_to_target_ms=2328.5`.
   - Runtime counters show no actual KVC eviction/reload: `kvc_evict_count_total=0`, `kvc_reload_count_total=0`.
   - This is wasted controller/planner time in the measured path.

5. Actual request batching collapses during long-context prefill/decode.
   - CoResid final stats show `observed_batch_size=1`, `native_schedule_batch_size=1`.
   - Server logs show long prompts entering chunked prefill one request at a time, then decode observations at small effective batch.
   - Compare scheduler behavior with Default before changing policy logic.

6. Expert hotness collection is expensive.
   - Detailed log stats show `profile_expert_hotness_record_ms=12682.2`.
   - `expert_hotness_sync_fallback_count=1620`.
   - Need check whether hotness recording is synchronizing or doing per-call Python-heavy work during prefill/decode.

Next debugging order:

1. Make expert install finish before measured decode or move it outside the timed path.
2. Remove/short-circuit KVC selection work when CoResid has no KVC action to execute.
3. Profile and reduce hotness recording sync/Python overhead.
4. Re-check scheduler batching behavior on WildChat long-context Default vs CoResid.
5. Only after the true runtime path is clean, revisit reclaim split/target behavior.

## CoResid Hotness/Copy Optimization Plan

Goal:

Move CoResid hotness collection and expert/KVC movement out of the decode
critical path. GPU should own hotness counting and candidate selection. CPU
should become a thin asynchronous copy submitter where CPU involvement is still
required. Planner decisions may use stale snapshots; do not synchronize on the
current step's freshly collected hotness.

Relevant upstream PR:

- https://github.com/sgl-project/sglang/pull/18589
- The PR optimizes SGLang's EPLB `per_token` expert-distribution recorder by
  replacing synchronous CUDA-to-CPU tensor conversion with a pinned CPU buffer,
  CUDA copy stream, events, deferred reconstruction, and bounded buffering.
- Direct cherry-pick is not enough for LayerKV because LayerKV hotness is an
  online runtime signal, not only an offline trace dump. Reuse the principle:
  no `.cpu()`, `.tolist()`, `.item()`, or equivalent CUDA synchronization in the
  hotness record path.

Phase 1: lock down baseline and counters.

- Keep the WildChat long-context E2E command as the main regression case.
- Compare only SGLang Default and CoResid first.
- Add or verify counters for:
  - GPU vs CPU hotness record path counts.
  - async snapshot issued/ready/dropped counts.
  - copy descriptor issued/enqueued/completed counts.
  - CPU copy enqueue time.
  - metadata commit wait/blocking time.
- Keep a runtime flag to switch between current CPU-sync hotness and new GPU
  hotness paths.

Phase 2: eliminate hotness synchronization first.

- Add per-layer GPU hotness buffers:
  - prefill counts: `counts_prefill[layer][num_experts]`.
  - decode counts: `counts_decode[layer][num_experts]`.
- In `_record_expert_hotness_for_layer()`, CUDA `topk_ids` must not call
  `.cpu()` or feed `torch.unique(...cpu...)`.
- Use GPU `scatter_add_` or `index_add_` to accumulate expert counts.
- Preserve CPU dicts only as stale/snapshot views for existing planner code.
- If no ready CPU snapshot exists, planner should reuse the previous snapshot or
  fall back to a static/cost-only policy. It must not block waiting for the
  current step's hotness.

Phase 2 acceptance:

- `expert_hotness_sync_fallback_count` should drop close to 0.
- `profile_expert_hotness_record_ms` should fall far below the WildChat E2E
  baseline of about 12682 ms.
- Existing correctness guards should still pass.

Phase 3: asynchronous hotness snapshots.

- Use pinned CPU tensors as bounded snapshot buffers.
- Use a dedicated CUDA stream to copy GPU counts to CPU with
  `copy_(..., non_blocking=True)`.
- Record CUDA events for snapshot readiness.
- Controller/scheduler should poll with `event.query()` and update CPU hotness
  dicts only when ready.
- Never call `event.synchronize()` in the decode critical path.
- Bound queue/buffer growth; drop or overwrite stale snapshots when full.
- Snapshot lag of one or more steps is acceptable.

Phase 3 acceptance:

- Snapshot issued/ready/dropped counters are visible.
- Planner records which hotness snapshot version it used.
- Missing ready snapshots do not block decode.

Phase 4: GPU candidate selection.

- Move hot/cold candidate scoring to GPU using hotness counts plus resident and
  offloaded masks.
- Emit small candidate tensors rather than full per-token traces:
  - evict layer ids.
  - evict expert ids.
  - install layer ids.
  - install expert ids.
  - score and sequence/version metadata.
- Asynchronously snapshot only candidate tensors to CPU.
- CPU should stop scanning full hotness or running heavy per-token unique/count
  logic.

Phase 4 acceptance:

- Hotness-related CPU planner time and `profile_planner_dp_*` overhead decrease.
- Candidate count/version is observable.
- Small deterministic cases match old policy behavior or have documented
  differences.

Phase 5: CPU as thin asynchronous copy submitter.

- Use fixed expert GPU slots and fixed CPU backing slots, preferably contiguous
  arenas or fixed descriptor tables.
- GPU/runtime emits copy descriptors:
  - direction.
  - source slot.
  - destination slot.
  - byte count.
  - sequence/version.
- A CPU background thread consumes descriptors and only enqueues
  `cudaMemcpyAsync` on a copy stream. It should not understand expert hotness or
  run planner logic.
- Record copy-done events.
- Commit expert slot/resident/remap metadata only after the copy event is ready.
- If a copy is not ready when needed, skip or use the old resident layout rather
  than blocking the main decode stream.

Phase 5 acceptance:

- `expert_install_blocking_ms` and critical-path `profile_copy_expert_to_cpu_ms`
  decrease substantially.
- `layerkv_copy_stream_busy_ms` and copy event counters reflect real async copy
  work.
- No metadata commit occurs before copy completion.

Phase 6: handle KVC separately.

- Prefer GPU-only virtual remap or arena metadata updates for KVC before adding
  CPU offload.
- Short-circuit KVC selection when no KVC task can be executed.
- Only reuse the copy-descriptor path for KVC after the GPU-only path is clean.

Phase 6 acceptance:

- `profile_kvc_select_evict_ms` is near zero when no KVC task is emitted.
- When KVC tasks exist, scheduler/copy counters reflect actual task execution.

Recommended first implementation step:

Implement Phase 2 and Phase 3 first: GPU hotness counting plus asynchronous CPU
snapshot. This directly targets the measured WildChat overhead
(`profile_expert_hotness_record_ms` around 12682 ms and
`expert_hotness_sync_fallback_count == expert_hotness_record_count`) while
keeping expert copy scheduling and metadata semantics mostly unchanged.

## Phase 2/3/4 Progress

Recorded: 2026-05-31T08:05:00Z

Committed baseline before Phase 4:

- Commit: `f0744a2e6 perf(layerkv): async hotness snapshots`.
- Phase 2/3 moved expert hotness accumulation to GPU and snapshots to async
  pinned CPU buffers.
- WildChat CoResid-only result:
  - `expert_hotness_record_count=1620`.
  - `expert_hotness_record_fast_count=1620`.
  - `expert_hotness_sync_fallback_count=0`.
  - `profile_expert_hotness_record_ms=281.6`.
  - `profile_expert_hotness_snapshot_ms=62.8`.

Stage A, GPU candidate order snapshot:

- Output: `/data/wenyan/tmp/layerkv_stage_a_wildchat_coresid`.
- Candidate snapshots are issued only from decode hotness and consumed as small
  per-layer expert-order tensors.
- WildChat counters:
  - `expert_candidate_snapshot_issue_count=192`.
  - `expert_candidate_snapshot_ready_count=192`.
  - `expert_candidate_snapshot_drop_count=0`.
  - `expert_candidate_order_hit_count=48`.
  - `expert_candidate_order_miss_count=0`.
- Performance:
  - `request_wall_ms=26644.6`.
  - `benchmark_decode0_latency_ms=1665.3`.
  - `output_throughput_tok_s=4.804`.
  - `profile_controller_per_decode_step_ms=338.5`.
  - `profile_copy_expert_to_cpu_ms=1292.9`.
  - `profile_install_expert_slots_ms=1333.7`.

Stage B, copy descriptor instrumentation:

- Output: `/data/wenyan/tmp/layerkv_stage_b_wildchat_coresid`.
- Descriptor schema records direction, reason, layer id, logical expert id,
  source slot, destination slot, byte count, parameter count, sequence, and
  decode step.
- Recorded descriptor counters:
  - `expert_copy_descriptor_count=132`.
  - `expert_copy_descriptor_d2h_count=127`.
  - `expert_copy_descriptor_h2d_count=5`.
  - `expert_copy_descriptor_install_count=122`.
  - `expert_copy_descriptor_evict_count=5`.
  - `expert_copy_descriptor_materialize_count=5`.
  - `expert_copy_descriptor_bytes=1245708288`.
  - `expert_copy_descriptor_param_count=264`.
- Stage B overhead is within run noise versus Stage A:
  - wall `26644.6 -> 26676.7 ms`.
  - decode0 `1665.3 -> 1667.3 ms`.
  - controller/decode step `338.5 -> 336.1 ms`.
  - copy-to-CPU `1292.9 -> 1269.1 ms`.
- Current bottlenecks remain:
  - install/copy still occurs on the critical path:
    `expert_install_blocking_ms=1311.9`,
    `profile_install_expert_slots_ms=1311.7`.
  - KVC selection still spends time without actual KVC eviction:
    `profile_kvc_select_evict_ms=3531.7`,
    `profile_kvc_evict_to_target_ms=3929.1`,
    `kvc_evict_count_total=0`.

Stage C.1, install backing D2H async attempt:

- Output: `/data/wenyan/tmp/layerkv_stage_c1_wildchat_coresid`.
- Install backing D2H is submitted on the copy stream and committed to
  `state.cpu_params` only after the ready event is finalized.
- Correctness/smoke:
  - `python -m py_compile` passed for runtime/eval scripts.
  - `scripts/layerkv_smoke.py` passed.
  - CUDA micro test verified pending install D2H finalizes into exact CPU
    backing tensors.
- WildChat counters:
  - `expert_install_d2h_async_count=122`.
  - `expert_install_d2h_async_finalize_count=122`.
  - `expert_install_d2h_async_mb=1098.0`.
  - `expert_install_d2h_async_wait_count=1`.
  - `expert_install_d2h_sync_fallback_count=0`.
  - `layerkv_copy_stream_busy_ms=1191.2`.
- Effect versus Stage B:
  - wall `26676.7 -> 27656.0 ms`.
  - decode0 `1667.3 -> 1728.5 ms`.
  - throughput `4.798 -> 4.628 tok/s`.
  - install blocking `1311.9 -> 1263.2 ms`.
  - copy-to-CPU profile `1269.1 -> 1209.1 ms`.
- Interpretation: the copy is now truly represented as async copy-stream work,
  but C.1 does not yet remove the CPU submit/allocation work from the install
  step and the D2H traffic overlaps with decode bandwidth. Stage C should next
  move descriptor submission off the decode critical path or defer/slice install
  D2H instead of launching full-layer backing copies during measured decode.

Fig4 prompt cache:

- `load_fig4_prompt_ids()` now caches tokenized Fig4 prompt IDs under
  `/data/wenyan/tmp/layerkv_fig4_prompt_cache` by default.
- Set `LAYERKV_FIG4_PROMPT_CACHE_DIR=0` to disable or point it to a different
  cache directory.
- Cache key includes dataset name/path, workload, batch size, min/max context,
  input length, seed, tokenizer id, and cache version.
- Validation for WildChat context-heavy:
  - first load wrote cache in `68.473s`;
  - second load hit cache in `0.021s`;
  - payload hash stayed `746dfb4a0f323fad`.

## Current Plan Snapshot

Recorded: 2026-05-31T08:25:00Z

Current branch state:

- Last committed optimization commit:
  `243b9f4e5 perf(layerkv): add copy descriptors`.
- Uncommitted work in progress:
  - Stage C.1 install backing D2H async attempt in
    `python/sglang/srt/layerkv/runtime.py`.
  - Fig4 prompt ID cache in `scripts/layerkv_eval_common.py` and eval logging
    in `scripts/layerkv_policy_eval.py`.
  - This handoff update.
- Do not treat C.1 as a successful perf change yet; it is evidence that naive
  async full-layer D2H is insufficient.

Reorganized phase map:

1. Phase 1, baseline and counters: complete.
   - WildChat long-context Default vs CoResid standalone was run.
   - Main bottlenecks were recorded: hotness sync, expert install/copy
     blocking, KVC no-op selection overhead, and batching behavior.

2. Phase 2, GPU hotness counting: complete and committed.
   - Commit: `f0744a2e6`.
   - Hotness record path uses GPU count buffers instead of CPU sync.

3. Phase 3, async hotness snapshots: complete and committed.
   - Commit: `f0744a2e6`.
   - Pinned CPU snapshot buffers, CUDA stream/events, no decode-path blocking.

4. Phase 4, GPU candidate selection: partially complete and committed.
   - Commit: `243b9f4e5`.
   - Expert candidate order snapshot is implemented and validated.
   - Not done: KVC candidate scoring/filtering on GPU.
   - GPU DP is explicitly out of scope for now.

5. Phase 5, copy submitter: in progress.
   - Stage B descriptor instrumentation is committed in `243b9f4e5`.
   - Stage C.1 install backing D2H async was tested but not committed as a
     standalone perf win.
   - Stage C must be re-centered around budgeted descriptor scheduling, not
     naive async copy.

6. Phase 6, KVC cleanup: queued.
   - Short-circuit KVC selection when no KVC task can execute.
   - Current evidence shows `profile_kvc_select_evict_ms` around 3.5s while
     `kvc_evict_count_total=0`.

7. Phase 7, standalone/PD alignment: queued.
   - Continue optimizing in standalone first.
   - Later rerun Default and CoResid under matching PD topology: one P GPU and
     one D GPU.

Updated Stage C plan:

- C.2 should be a descriptor-driven, chunked, budgeted copy scheduler.
- Install stage should generate descriptors rather than immediately submitting
  full-layer backing D2H.
- Per decode step, submit only a bounded copy budget, initially something like
  `64 MiB/step` with a `128 MiB` chunk cap, then tune from WildChat data.
- Preserve large-copy bandwidth when outside the decode critical path, such as
  prefill, idle windows, or explicit high-priority recovery.
- Maintain metadata correctness:
  - D2H ready before inserting into `state.cpu_params`.
  - H2D ready before marking a slot resident or pointing remap at it, unless
    the main stream has waited on the event.
  - Pending-not-ready experts can use wait fallback first, with counters.

Recommended next order:

1. Commit the Fig4 prompt cache separately, because it is clearly beneficial.
2. Decide whether to keep the C.1 code as scaffolding for C.2 or roll it back
   before implementing the budgeted queue.
3. Implement Stage C.2 budgeted descriptor queue for install backing D2H first.
4. Rerun WildChat standalone CoResid-only using the prompt cache.
5. Implement KVC no-op selection short-circuit if C.2 still leaves controller
   overhead dominated by KVC scanning.
6. Rerun standalone Default vs CoResid once the runtime path stabilizes.

## Stage C Revised Scope

Recorded: 2026-05-31T08:35:00Z

Stage C goal:

- Move expert copy out of the decode critical path.
- Allow intermediate demand to raise priority or insert urgent copy work.
- Preserve the original design boundary: the CPU copy submitter must not
  understand expert hotness, planner policy, or expert semantics. It should
  consume already ordered descriptors with source, destination, byte count,
  priority/deadline, sequence/version, and dependency metadata.

C.1: naive async install backing experiment.

- Status: implemented as an uncommitted experiment, not a successful perf
  optimization.
- Scope:
  - install backing GPU-to-CPU D2H is submitted on the copy stream.
  - event-ready finalization commits copied tensors into `state.cpu_params`.
- Result:
  - async path worked functionally (`122` experts, `1098 MiB`, `0` sync
    fallback);
  - wall/decode0 regressed because large D2H, CPU allocation, `index_select`,
    and submission still happen in the measured path.
- Use going forward:
  - keep/reuse pending event and finalize mechanics if useful;
  - do not treat naive full-layer async D2H as the final Stage C design.

C.2: D2H backing descriptor queue.

- Scope: GPU-to-CPU expert backing only.
- Includes:
  - install backing D2H;
  - evict backing D2H.
- Excludes:
  - CPU-to-GPU materialize H2D;
  - remap/topk use-point readiness;
  - expert core wait logic.
- Desired behavior:
  - install/shrink/evict stages generate D2H descriptors instead of immediately
    submitting full-layer copies;
  - descriptors enter an expert backing queue with priority, deadline, seq, and
    version;
  - each decode step submits only a bounded copy budget;
  - chunk size is bounded so urgent demand waits at most for the current chunk,
    not for an entire layer or full install queue;
  - D2H ready event must fire before inserting backing into
    `state.cpu_params`.
- Intermediate demand handling:
  - if demand needs an expert whose backing descriptor is queued, boost that
    descriptor priority/deadline;
  - if backing D2H is already pending, wait fallback is allowed first for
    correctness and must be counted;
  - if backing is ready, consume it directly;
  - if no descriptor exists, create an urgent descriptor or fall back sync.
- Acceptance:
  - `expert_install_blocking_ms` decreases;
  - `profile_copy_expert_to_cpu_ms` is no longer concentrated in install;
  - queue length, submitted bytes, finalized bytes, wait count, and fallback
    count are observable;
  - wall/decode0 do not regress versus Stage B.

C.3: H2D materialize descriptors and use-point guard.

- Scope: CPU-to-GPU expert materialization only.
- Includes:
  - on-demand materialize;
  - prefetch materialize;
  - dependencies on C.2 backing readiness;
  - readiness checks before expert core consumes remapped slots.
- Desired behavior:
  - generate H2D descriptors for materialize work;
  - allow urgent on-demand descriptors to outrank prefetch descriptors;
  - H2D ready event must fire, or the main stream must wait on it, before remap
    points execution at the target slot;
  - backing-not-ready materialize should boost or wait on the corresponding
    C.2 D2H descriptor.
- Acceptance:
  - materialize async counters increase;
  - ready-before-use improves;
  - copy-stream waits and deadline misses stay bounded;
  - no correctness guard failures from reading incomplete weights.

C.4: unified metadata commit rules.

- D2H complete before `state.cpu_params` insertion.
- H2D complete, or explicit main-stream wait, before resident/remap commit.
- Descriptor seq/version prevents stale copy completion from overwriting newer
  residency state.
- Pending/cancelled descriptors are cleaned up without leaking residency state.

C.5: tuning and regression.

- Tune chunk size and per-step copy budget, initially comparing values around
  `32`, `64`, and `128 MiB`.
- Track bandwidth as submitted/finalized MB over copy-stream busy time, but do
  not optimize bandwidth at the expense of decode0/wall latency.
- Regression order:
  1. micro tests for descriptor dependency and finalize;
  2. `scripts/layerkv_smoke.py`;
  3. WildChat standalone CoResid-only using prompt cache;
  4. standalone Default vs CoResid once expert path stabilizes;
  5. PD alignment later, after standalone behavior is stable.

## Stage C.2 Implementation Progress

Recorded: 2026-05-31T09:15:00Z

Implemented:

- Expert install backing D2H now uses a descriptor-style queue before physical
  slot install.
- New server/runtime knobs:
  - `--layerkv-expert-copy-budget-mb`, default `64`;
  - `--layerkv-expert-copy-chunk-mb`, default `128`.
- Each decode safe point submits only the configured expert D2H budget.
- Install queue waits for backing readiness before invoking the slot shrink and
  install path.
- D2H completion commits copied tensors into the install item's pending CPU
  backing map only after the CUDA ready event is finalized.
- CPU/non-optimized smoke paths keep the old synchronous fallback.
- Policy eval CSV now records install D2H queue, async, finalize, fallback,
  budget, and chunk counters. Fig4 prompt tokenization is cached for repeat
  WildChat runs.

Current limitation:

- C.2 currently budgets install-backing D2H. Expert eviction backing already has
  an async D2H event path, but it is not yet folded into the same budgeted
  install queue. That is acceptable for the expert-side first pass because the
  blocking regression we measured was dominated by install backing.
- H2D materialization/use-point readiness remains Stage C.3.

Validation:

- `python -m py_compile` passed for runtime/server args/eval scripts.
- `scripts/layerkv_smoke.py` passed.
- CUDA micro test verified queued install D2H finalizes exact CPU backing
  tensors.
- WildChat context-heavy prompt cache hit reduced prompt load to milliseconds
  in repeat runs.

WildChat standalone CoResid results:

- Stage B baseline:
  - wall `26676.7 ms`, decode0 `1667.3 ms`, throughput `4.798 tok/s`;
  - `expert_install_blocking_ms=1311.9`;
  - `profile_copy_expert_to_cpu_ms=1269.1`;
  - completed install layers `15`, physical expert reclaim `1098 MiB`.
- C.1 naive full-layer async D2H:
  - wall `27656.0 ms`, decode0 `1728.5 ms`, throughput `4.628 tok/s`;
  - not a perf win.
- C.2, `64 MiB/step`, `128 MiB` chunk:
  - output `/data/wenyan/tmp/layerkv_stage_c2_wildchat_coresid`;
  - wall `25059.2 ms`, decode0 `1566.2 ms`, throughput `5.108 tok/s`;
  - `expert_install_blocking_ms=129.0`;
  - `profile_copy_expert_to_cpu_ms=657.4`;
  - `layerkv_copy_stream_busy_ms=588.0`;
  - queued/submitted/finalized experts `57/53/45`, queue remaining `4`;
  - completed install layers `6`, physical expert reclaim `405 MiB`.
- C.2, `128 MiB/step`, `128 MiB` chunk:
  - output `/data/wenyan/tmp/layerkv_stage_c2_128mb_wildchat_coresid`;
  - wall `25790.0 ms`, decode0 `1611.9 ms`, throughput `4.963 tok/s`;
  - `expert_install_blocking_ms=282.8`;
  - `profile_copy_expert_to_cpu_ms=964.4`;
  - `layerkv_copy_stream_busy_ms=844.2`;
  - queued/submitted/finalized experts `69/69/57`, queue remaining `0`;
  - completed install layers `7`, physical expert reclaim `513 MiB`.

Interpretation:

- C.2 achieves the intended first-order effect: expert install blocking drops
  from about `1.31s` to `0.13s` at `64 MiB/step`.
- `64 MiB/step` is the better latency point in the current short 16-token
  decode window.
- The tradeoff is under-installing during short requests: reclaim drops from
  `1098 MiB` in Stage B to `405-513 MiB` in C.2. This means the next expert-side
  work should either add an idle/prefill drain mode or tune budget by remaining
  decode horizon, rather than blindly raising the per-step budget.

## Stage C.2.1/C.2.2 Progress

Recorded: 2026-05-31T09:45:00Z

Implemented after C.2 commit `05ad7980e`:

- Optional force-drain experiment:
  - flag: `--layerkv-expert-copy-force-drain`;
  - blocks after expert plan creation to drain install backing D2H and slot
    install work;
  - intended only for reclaim/blocking upper-bound experiments, not default.
- Horizon-aware expert D2H copy budget:
  - reuses `--layerkv-expert-install-target-steps`;
  - computes queued D2H bytes over remaining target steps;
  - raises the effective per-step copy budget above
    `--layerkv-expert-copy-budget-mb` when needed;
  - optional cap: `--layerkv-expert-copy-max-budget-mb`, where `0` means no cap.
- Optional install D2H lookahead:
  - flag: `--layerkv-expert-copy-lookahead-layers`;
  - default `0` to preserve the low-latency C.2 behavior;
  - when positive, later install items can enqueue backing D2H descriptors while
    earlier layers are still waiting;
  - physical slot shrink/install remains in original queue order.
- New CSV/stats:
  - `expert_install_d2h_effective_budget_mb`;
  - `expert_install_d2h_max_budget_mb`;
  - `expert_install_d2h_lookahead_layers`;
  - `expert_install_d2h_lookahead_queue_count`;
  - `expert_install_d2h_dynamic_budget_count`;
  - `expert_install_d2h_force_drain_count`;
  - `expert_install_d2h_force_drain_ms`.

Validation:

- `python -m py_compile` passed.
- `git diff --check` passed.
- `PYTHONNOUSERSITE=1 scripts/layerkv_smoke.py` passed.

WildChat spot checks:

- Dynamic budget without lookahead:
  - command added `--expert-install-target-steps 16
    --expert-copy-max-budget-mb 128`;
  - output `/data/wenyan/tmp/layerkv_stage_c2_dynamic_wildchat_coresid`;
  - completed layers `10`, reclaim `747 MiB`;
  - wall `26381.2 ms`, decode0 `1648.8 ms`, throughput `4.852 tok/s`;
  - install blocking `395.2 ms`;
  - conclusion: better reclaim than fixed 64 MiB but still descriptor-starved,
    because only the queue head was generating D2H work.
- Dynamic budget with lookahead `4`:
  - command added `--expert-copy-lookahead-layers 4`;
  - output
    `/data/wenyan/tmp/layerkv_stage_c2_dynamic_lookahead_wildchat_coresid`;
  - completed layers `14`, reclaim `981 MiB`;
  - wall `28552.8 ms`, decode0 `1784.5 ms`, throughput `4.483 tok/s`;
  - install blocking `261.3 ms`;
  - D2H queued/submitted/finalized `145/134/119`;
  - lookahead queued `17` layer backings;
  - conclusion: lookahead fixes descriptor starvation and nearly restores Stage
    B reclaim, but the extra copy traffic hurts latency. Keep it opt-in until a
    pressure-aware trigger is added.

Current recommendation:

- Keep default C.2 behavior focused on latency: lookahead `0`, copy budget
  `64 MiB`.
- Use `--expert-install-target-steps 16 --expert-copy-max-budget-mb 128
  --expert-copy-lookahead-layers 4` only as a reclaim-pressure experiment.
- Next optimization should be pressure-aware lookahead/drain, not unconditional
  lookahead.

## Stage C.2.3 Copy Path Experiment

Recorded: 2026-05-31T10:15:00Z

Implemented:

- Run-based install D2H coalescing:
  - expert ids are split into contiguous runs;
  - runs with length greater than one use `param[start:end]` slice copies
    instead of `index_select`;
  - singleton/non-contiguous ids are still batched through `index_select` to
    avoid many tiny D2H submits.
- Conservative CPU backing pool:
  - released CPU backing tensors are reused only when they own their storage;
  - batched backing views are not pooled, because reusing one view could
    corrupt other experts sharing the same underlying storage;
  - default internal pool cap is `256 MiB`.
- New counters:
  - `expert_d2h_slice_run_count`;
  - `expert_d2h_slice_expert_count`;
  - `expert_d2h_gather_batch_count`;
  - `expert_d2h_gather_expert_count`;
  - `expert_cpu_backing_pool_*`.

Validation:

- `python -m py_compile` passed.
- `git diff --check` passed.
- `PYTHONNOUSERSITE=1 scripts/layerkv_smoke.py` passed.

WildChat default C.2 rerun:

- Output: `/data/wenyan/tmp/layerkv_stage_c23_coalesce_wildchat_coresid`.
- Command used default `64 MiB/step`, lookahead `0`.
- Result:
  - wall `25155.9 ms`;
  - decode0 `1572.2 ms`;
  - throughput `5.088 tok/s`;
  - install blocking `146.9 ms`;
  - completed layers `6`;
  - reclaim `405 MiB`;
  - `profile_copy_expert_to_cpu_ms=567.5`;
  - `layerkv_copy_stream_busy_ms=496.5`.
- Copy breakdown:
  - slice runs `10`;
  - slice experts `22`;
  - gather batches `26`;
  - gather experts `84`;
  - pool alloc `36`;
  - pool reuse/release `0/0` in this short run.
- Comparison against C.2 default before coalescing:
  - wall `25059.2 -> 25155.9 ms` (roughly flat);
  - decode0 `1566.2 -> 1572.2 ms` (roughly flat);
  - copy profile `657.4 -> 567.5 ms`;
  - copy stream busy `588.0 -> 496.5 ms`;
  - install blocking `129.0 -> 146.9 ms`.
- Interpretation:
  - slice coalescing reduced copy-path measured time without changing reclaim;
  - the end-to-end latency change is within noise and still dominated by KVC
    no-op selection plus general controller overhead;
  - CPU backing pool does not help this particular short run because backing is
    not released/reused during the same request.

## Stage C.3 Expert H2D Reload Path

Recorded: 2026-05-31T11:05:00Z

Implemented:

- Split expert copy streams:
  - D2H install/evict backing uses `_expert_d2h_stream`;
  - H2D expert reload/materialize uses `_expert_h2d_stream`;
  - the existing `_copy_stream` remains the fallback/common stream.
- On-demand expert reload now issues async H2D copies when optimized profiling
  is enabled, instead of forcing host-side sync.
- Ready-before-use guard:
  - materialized experts are marked `materializing` with CUDA events;
  - before top-k/slot use, the decode stream waits on the matching ready event;
  - ready/miss/wait counters are shared with the scheduler-facing metrics.
- New CSV/stats:
  - `expert_d2h_stream_launch_count`;
  - `expert_h2d_stream_launch_count`;
  - `expert_d2h_stream_busy_ms`;
  - `expert_h2d_stream_busy_ms`;
  - `expert_h2d_on_demand_async_count`;
  - `expert_h2d_prefetch_async_count`;
  - `expert_h2d_sync_count`.

Validation:

- `python -m py_compile` passed.
- `git diff --check` passed.
- `PYTHONNOUSERSITE=1 scripts/layerkv_smoke.py` passed.

WildChat default C.3 rerun:

- Output: `/data/wenyan/tmp/layerkv_stage_c3_default_wildchat_coresid`.
- Command used default `64 MiB/step`, lookahead `0`.
- Result:
  - wall `25348.8 ms`;
  - decode0 `1584.3 ms`;
  - throughput `5.050 tok/s`;
  - install blocking `24.3 ms`;
  - completed layers `6`;
  - reclaim `405 MiB`;
  - D2H stream launches/busy `13 / 597.7 ms`;
  - H2D stream launches/busy `0 / 0.0 ms`.
- Interpretation:
  - default workload did not trigger expert reload, so C.3 only validates the
    separated D2H stream path in this configuration;
  - end-to-end latency remains roughly flat versus C.2.3 default.

WildChat lookahead C.3 spot check:

- Output: `/data/wenyan/tmp/layerkv_stage_c3_h2d_lookahead_wildchat_coresid`.
- Command added `--expert-install-target-steps 16
  --expert-copy-max-budget-mb 128 --expert-copy-lookahead-layers 4`.
- Result:
  - wall `28077.2 ms`;
  - decode0 `1754.8 ms`;
  - throughput `4.559 tok/s`;
  - install blocking `273.5 ms`;
  - completed layers `14`;
  - reclaim `981 MiB`;
  - H2D on-demand async/sync `2 / 0`;
  - H2D stream launches/busy `2 / 0.387 ms`;
  - ready-use checks `2`, ready-before-use `0`, stream waits `2`;
  - deadline misses `2`, exposed wait `0.013 ms`.
- Comparison against the pre-C.3 lookahead run:
  - wall `28552.8 -> 28077.2 ms`;
  - decode0 `1784.5 -> 1754.8 ms`;
  - throughput `4.483 -> 4.559 tok/s`;
  - materialize host sync `2 -> 0`;
  - async materialize `0 -> 2`.
- Interpretation:
  - C.3 removes the host-synchronous expert reload path;
  - the two reloads still reached use before the copy had completed, so decode
    had to wait on the CUDA event;
  - the remaining bottleneck is still install/KVC/controller scheduling, not
    H2D copy time itself.

Known caveat:

- These policy-eval rows still exit non-zero with
  `INSTALL_IN_PROGRESS_NOT_COMPARABLE`; this is expected for the current
  long-context CoResid experiment and does not invalidate the collected runtime
  counters.

## Phase 5 Expert On-Demand D2H Priority

Recorded: 2026-05-31T11:45:00Z

Implemented:

- Intermediate on-demand expert backing now feeds back into the D2H descriptor
  queue.
- If on-demand materialize needs an expert whose D2H backing descriptor is
  still queued:
  - the requested expert id is split out of the original install job;
  - it is requeued as a single-expert `demand_backing_async` job;
  - priority is raised above normal install/lookahead work;
  - deadline is moved to the current decode step;
  - the job can mirror finalized backing into both the original install target
    and the live layer `state.cpu_params`.
- If no queued descriptor exists but the module is still full-sized, the runtime
  can enqueue a new urgent demand descriptor.
- If D2H is already pending, it cannot be preempted; the use point waits and the
  demand wait is counted.
- D2H submit ordering is now priority-first, then deadline, then sequence. With
  equal priority this preserves the previous deadline/queue order.

New CSV/stats:

- `expert_d2h_demand_ready_hit_count`;
- `expert_d2h_demand_boost_count`;
- `expert_d2h_demand_urgent_enqueue_count`;
- `expert_d2h_demand_async_count`;
- `expert_d2h_demand_finalize_count`;
- `expert_d2h_demand_pending_wait_count`;
- `expert_d2h_demand_unavailable_count`.

Validation:

- `python -m py_compile` passed.
- `git diff --check` passed.
- `PYTHONNOUSERSITE=1 scripts/layerkv_smoke.py` passed.
- A local queue-promotion construction test confirmed that the demanded expert
  is split into a high-priority single-expert job at the front of the queue,
  while the original install job keeps the remaining expert ids.

Important limitation:

- CUDA copies already submitted to the D2H stream still cannot be preempted.
  The priority mechanism only changes ordering for descriptors that have not
  been submitted yet; bounded chunk size is still required to keep urgent waits
  short.

## Phase 5 Ready-Miss Stall Accounting

Recorded: 2026-05-31T12:20:00Z

Implemented:

- Added CUDA-event accounting for use-point ready misses.
- Expert H2D ready misses now record events around the main/decode stream
  `wait_event()` and finalize the elapsed time later without synchronizing at
  the use point.
- KVC ready-miss fields were added to the shared stats/CSV path as well.
- New CSV/stats:
  - `expert_ready_miss_stall_ms`;
  - `expert_ready_miss_stall_count`;
  - `kvc_ready_miss_stall_ms`;
  - `kvc_ready_miss_stall_count`;
  - `layerkv_main_stream_wait_ms`.

Validation:

- `python -m py_compile` passed.
- `git diff --check` passed.
- `PYTHONNOUSERSITE=1 scripts/layerkv_smoke.py` passed.

WildChat spot check:

- Output:
  `/data/wenyan/tmp/layerkv_stage_c4_wait_stall_chunk128_wildchat_coresid`.
- Command used lookahead `4`, target steps `16`, max copy budget `128 MiB`,
  default chunk `128 MiB`.
- Result:
  - wall `27626.9 ms`;
  - decode0 `1726.7 ms`;
  - throughput `4.633 tok/s`;
  - expert ready checks `2`;
  - expert ready-before-use `0`;
  - expert ready-miss stall count `2`;
  - expert ready-miss GPU stall `0.0705 ms`;
  - scheduler CPU exposed wait submission time `0.0334 ms`;
  - H2D on-demand async count `2`;
  - H2D stream busy `0.386 ms`.

Interpretation:

- The observed expert ready misses are real, but their measured GPU stall in
  this run is small: about `0.035 ms` per miss.
- The larger decode latency gap is not primarily from the two H2D use-point
  waits; controller/KVC no-op selection and install/shrink bookkeeping still
  dominate.
