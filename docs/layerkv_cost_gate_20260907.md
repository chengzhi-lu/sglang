# Measured transition-cost gate for adaptive expert growth

Positive shadow-LRU saved misses alone no longer authorize repeated growth.
`ResidencyBudget.decide()` accepts measured transfer milliseconds per materialized
expert and the most recent successful growth transaction time. It estimates one
observed interval's savings as saved misses times transfer cost. That estimate
must strictly exceed known transition cost to choose the upper target.

The physical controller obtains transfer cost from runtime materialization
milliseconds/count and measures successful growth from donor discovery through
capacity/stat publication. `last_growth_ms` remains unknown until calibration;
one positive-benefit growth may calibrate it. Unknown transfer cost cannot repay
a known transition cost. Nonfinite/negative supplied costs are rejected rather
than silently treated as zero. Already-grown slots pass zero transition cost:
the policy must not repeatedly charge a sunk growth cost. Admission and KV
headroom pressure still override expert retention immediately.

Telemetry includes `cost_rejections`, nullable `estimated_saved_ms` and nullable
`last_growth_ms` inside `shared_vmm.residency_budget`. No new environment switch
or performance constant is introduced. The observation horizon is the existing
explicit decision interval.

Limitations: this is a measured-cost **growth gate**, not an optimal cost model.
Shadow LRU uses deduplicated batch demand, not exact token-group replacement.
GPU transfer busy time is not necessarily exposed critical-path time. The growth
measurement excludes later kernel-shape JIT, recall and continuing per-step
headroom checks. One interval may underestimate useful long-term residency.
It cannot predict batch-admission benefit, which remains a separate requirement.

## Validation

- Pure tests cover cheap/prohibitive growth, unknown transfer measurements and
  negative/NaN/infinite cost rejection.
- The real shared controller test proves an expensive growth decision returns
  base capacity without donor discovery or page loans.
- Full suite before the final runtime test: 340 passed; that additional runtime
  test passes separately. Targeted Ruff and `git diff --check` pass.

## Bounded model run

Dry plan inspected; exact common settings are the same as
[batched growth](layerkv_batch_growth_20260907.md), except **three rounds** and
output directory `/mnt/vdb/chengzhi/layerkv_adaptive_20260907_costgate_b8_01`.
Three rounds test behavior after the calibration growth, not just its initial
cost. BF16 Qwen3.6-35B-A3B/H100 NVL/TP1, B8, input128/output32, base16/max24,
adaptive interval8/headroom16. This remains fixed-first, without independent
native output validation or reversed-order acceptance repeats.

The three-round pair passes exact outputs/logprobs for all 768 tokens, physical
budget matching and recorded guards. Fixed/adaptive output tok/s:

| Round | Fixed | Adaptive | Relative throughput |
| --- | ---: | ---: | ---: |
| 1 | 43.4781 | 43.4949 | +0.04% |
| 2 | 90.8313 | 84.0570 | -7.46% |
| 3 | 85.4404 | 80.0052 | -6.36% |

There is one calibration growth in round 2, measured at 66.95 ms. Round 3 has
zero predicted saved misses and does not grow. **Cost rejection count stays zero**:
this full-model trace validates integration, not actual cost-gate rejection.
The rejection path is exercised by the real-controller test, not claimed from
this run. All rounds retain actual B8. No performance target is met.

Round 3 materializations are exactly 2735 in both arms, H2D busy time is
349.70/349.59 ms, but LayerKV Python overhead is 71.06/136.62 ms. This motivates
the follow-up: loan-free benefit/cost decisions defer KV discovery until growth
is actually proposed. Existing loans still check pressure; all new loans still
pass exact post-selection headroom guards. `free_tokens=None` means a deferred
proposal, never authorization to claim physical pages. Regression tests cover
no-benefit decisions avoiding allocator reads and zero-cost donors being rejected
when common/native capacity is nevertheless insufficient for decode headroom.

The deferred-query adaptive-only follow-up is saved at
`/mnt/vdb/chengzhi/layerkv_adaptive_20260907_deferred_b8_01`, with the same three
rounds and arguments plus `--arm lend`. All 768 outputs/logprobs match the saved
fixed reference exactly, physical pools match, and KV/expert/ownership guards
pass. Round-3 Python overhead is 102.93 ms versus the earlier adaptive 136.62 ms.
Follow-up throughput is 43.38/84.89/80.54 tok/s; this is not a new paired estimate
of speedup. Cost rejection remains zero on this trace, so no measured benefit is
attributed to the gate itself.

A final pure-policy correction clears interval miss/sample counters during KV
pressure, while retaining the bounded cache contents. Otherwise a long pressure
episode could be incorrectly priced as one future decision interval. Its test
checks repeated pressure and fresh post-pressure observation. This correction
was made after the model follow-up and is covered by unit validation, not another
full-model run. Cross-request expert retention and the fixed expert-heavy
admission baseline remain unimplemented; the research goal is still incomplete.

Final full LayerKV validation: **344 passed, 19 warnings**; targeted Ruff and
`git diff --check` pass. No full repository pre-commit run.
