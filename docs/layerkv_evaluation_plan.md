# LayerKV Evaluation Plan

This note records the evaluation plan to run after the current end-to-end real-trace experiment finishes.

## Main Evaluation

Use real trace as the main end-to-end experiment.

- Trace source: AzureLLMInferenceTrace_conv.csv arrival time and input/output lengths.
- Prompt source: ShareGPT first, WildChat fallback when matching context length is unavailable.
- Primary setting: 300s trace window, SGLang max running batch size 256.
- Do not truncate with a small max request count for the main result.
- Report actual workload pressure, not only configured pressure.

Required workload diagnostics:

- Request count in the replay window.
- Average request rate and burst rate.
- Context length mean, p50, p95, max.
- Output length mean, p50, p95, max.
- Average and max running batch size.
- Pending queue mean, p95, max.
- Actual reclaimed memory, planned reclaimed memory, and physical reclaimed memory.

Primary metrics:

- Output token throughput.
- Request throughput.
- TTFT p50/p95.
- TPOT p50/p95.
- End-to-end latency p50/p95.
- Success/failure count and OOM count.
- KVC reload count and expert materialize count.

Policies:

- expert-first.
- kvc-first.
- ratio-25-75.
- ratio-50-50.
- ratio-75-25.
- layer-aware-joint-dp.

Important interpretation:

- Natural real trace may not create GB-level overload. If actual reclaim is small, DP and simple policies may be close.
- For memory-pressure claims, use the same real trace workload with controlled pressure levels.

Pressure settings to test:

- Natural pressure.
- 2GB controlled pressure.
- 4GB controlled pressure.
- 8GB controlled pressure.

## Controlled Workload Evaluation

Keep controlled workloads as explanation experiments rather than the only main result.

Workloads:

- Batch-heavy: high batch size, short context, e.g. bs 128/256 and ctx around 1K.
- Context-heavy: lower batch size, long context, e.g. ctx 8K/16K/32K.
- Balanced: medium batch and medium context.

Purpose:

- Show batch-heavy prefers expert residency.
- Show context-heavy prefers KVC residency.
- Show fixed policies cannot adapt across workloads.
- Explain why layer-aware joint planning is needed.

## Ablation Studies

### A1. Planner Ablation

Purpose: isolate C1, value-aware joint KV/expert decisions.

Compare:

- kvc-only.
- expert-only.
- ratio-50-50 or ratio policies.
- joint-greedy.
- joint-dp.

Metrics:

- TPOT and throughput.
- Selected KVC MB and expert MB.
- Exposed KVC stall and expert stall.
- KVC reload count and expert materialize count.

Questions to answer:

- Is joint planning better than KV-only or expert-only?
- Is DP better than greedy?
- Is layer-aware selection better than uniform ratio?

### A2. Cost Model Ablation

Purpose: show the final cost model is necessary.

Compare:

- Raw copy cost.
- Exposed stall cost.
- No-overlap model.
- Uniform layer cost.
- Layer/use-point-aware cost.
- With and without expert churn cost.

Metrics:

- Predicted vs measured ranking.
- Final selected plan.
- TPOT and throughput.
- Exposed stall breakdown.

### A3. Runtime Layout Ablation

Purpose: isolate C2, controlling plan-induced layout irregularity.

Compare:

- Destructive KVC eviction.
- Arena without dirty workspace reuse.
- Arena with coalesced packing.
- Arena with dirty-region workspace reuse.
- Final arena backend.

Metrics:

- Per-step latency.
- Allocation churn counters: cat, contiguous, empty_like, set_kv.
- Destructive eviction and pointer replacement counters.
- Reserved memory growth.
- Workspace pack time.
- Correctness match.

### A4. Scheduler Ablation

Purpose: isolate C3, deadline-aware copy-side recovery.

Compare:

- Synchronous recovery.
- Naive prefetch.
- Deadline scheduler.
- Deadline scheduler plus coalesced KV reload.
- Deadline scheduler plus batched expert materialize.
- Deadline scheduler plus bounded expert backing cache.

Metrics:

- Ready-before-use ratio.
- Deadline miss count.
- Visible KVC stall.
- Visible expert stall.
- Copy stream busy time.
- TPOT and throughput.

### A5. Full System Ablation

Purpose: show cumulative system contribution.

Compare:

- Base SGLang offload.
- + joint planner.
- + arena stable residency.
- + deadline scheduler.
- Full LayerKV.

Metrics:

- Output throughput.
- TPOT p95.
- OOM/failure rate.
- Actual reclaimed memory.

## Runtime/Correctness Checks

For valid policy rows, record:

- guard_pass for KVC and expert.
- kvc_cat_count.
- kvc_contiguous_count.
- kvc_empty_like_count.
- kvc_set_kv_count.
- kvc_destructive_evict_count.
- kvc_pointer_replace_count.
- expert_slot_resize_count.
- resident_group_state_error_count.

The arena-backed runtime should keep destructive KVC counters at zero in steady state.

## Suggested Evaluation Order After Current E2E Run

1. Parse and summarize the current 300s, max-running-requests=256 real-trace run.
2. Decide whether natural pressure is sufficient from actual reclaim, running batch, and pending queue.
3. If natural pressure is low, run controlled pressure on the same real trace.
4. Run controlled batch-heavy/context-heavy workloads for explanation.
5. Run planner ablations.
6. Run runtime layout and scheduler ablations.
7. Build final paper figures and tables.
