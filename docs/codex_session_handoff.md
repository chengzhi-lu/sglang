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
