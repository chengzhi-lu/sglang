# Adaptive shared residency budget: first integrated controller

## Interface and policy

`ResidencyBudget` separates CPU-only budget selection from physical ownership and
page movement in `SharedExpertController`. Opt in with:

```text
--layerkv-shared-expert-policy adaptive
--layerkv-shared-expert-free-kv-donors
--layerkv-shared-expert-decision-interval 8
--layerkv-shared-expert-headroom-steps 16
```

The existing shared layer/slot arguments still bound capacity. `fixed` is the
default and preserves the prior always-target-extra-slots behavior. The server
rejects adaptive mode without a shared layer/free donors, unknown policies and
nonpositive intervals/headroom. Effective arguments flow through LayerKVConfig
and benchmark metadata; no environment-based feature switch is introduced.

The existing decode chunk-capacity check already obtains CPU logical expert IDs.
Adaptive mode reuses those IDs once per decode step, without another GPU routing
readback. Two bounded shadow LRU caches estimate misses at base and maximum
capacity. After an observation interval, positive saved misses choose the upper
target; otherwise choose base. This is a **reuse proxy**, not measured transfer
benefit, and does not model the exact materializer replacement or chunk ordering.

KV shortage with queued admission, or inadequate decode-growth headroom, overrides
expert reuse. Headroom uses actual batch size times configured steps. Before a
loan, remove the exact donor union from common free capacity and preserve the
required remainder. Unreserved native free tokens are counted separately; both
ledgers are re-read after arena preparation, which can move capacity between them.
No planned reclaim MB is credited. Existing admission recall remains immediate.
Recall imposes a fresh observation/cooldown interval; failed donor/headroom scans
back off for one interval. The controller can shrink previously grown capacity.

Current limitations remain explicit:

- Only one selected expert layer and base/max budget choices; no partial recall.
- Prefill and request completion still recall loans. Cross-request retention is
  not yet implemented.
- Context pressure uses current CPU-visible KV demand plus the batch headroom
  reserve against allocator capacity, not a hard context-length cutoff. A large
  batch alone is not sufficient evidence to retain more experts.
- Donor eligibility now uses [per-layer free ownership](layerkv_per_layer_donors_20260907.md)
  and prioritizes low incremental KV-capacity cost. Capacity/performance scaling
  still requires full-model evidence.
- Saved misses do not yet pay for scan/remap/transfer costs in the decision.
- No larger actual batch or 20% throughput improvement is established.

## Tests and first model evidence

Pure policy tests cover reusable working sets, no benefit, duplicate observations,
pressure override, cooldown, invalid values and argument propagation. CUDA tests
exercise the controller's real loan path: deny a loan that exhausts headroom,
allow it when distinct native capacity remains, and recall under increased batch
pressure. The native-capacity test first failed on the expected missing loan, then
passed after the accounting fix. The complete LayerKV suite: **326 passed,
19 warnings**. Targeted Ruff, formatting and `git diff --check` pass; full repository
pre-commit was not run.

First matched pair (before the native-headroom accounting follow-up):
`/mnt/vdb/chengzhi/layerkv_adaptive_20260907_stagger_01`.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  ../scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_adaptive_20260907_stagger_01 \
  --requests 2 --max-running-requests 2 --rounds 1 \
  --input-tokens 128 --output-tokens 32 \
  --late-input-tokens 6000 --late-after-output-tokens 16 \
  --expert-policy adaptive --execute
```

Dry plan inspected before execution; Qwen3.6-35B-A3B BF16/H100 NVL/TP1. `fixed`
has 16 slots, `lend` opts into adaptive 16/17. All 64 output tokens and logprobs
match exactly, guards and physical pool comparison pass. Adaptive records five
decisions, one growth, four donor scans totaling 78.56 ms. Actual batches remain
32 B1 and 15 B2 forwards in both arms. Fixed/adaptive throughput is 8.9386/8.3531
tok/s (-6.55%). Output-triggered arrival timing is diagnostic, not fixed offered
wall-clock load; do not compare its ratio directly with other arrival triggers.

The headroom-accounting follow-up runs only the adaptive child in
`/mnt/vdb/chengzhi/layerkv_adaptive_20260907_headroom_01` with identical requests.
It is a current-code correctness check against the saved fixed response, not a
new matched performance pair.

The follow-up finishes with all 64 token IDs/logprobs exact against that fixed
reference, matching 478150656 physical bytes, zero growth physical creates and
all KV/expert/ownership guards passing. It still performs five decisions, one
growth and four donor scans (78.67 ms), and records one admission recall restoring
4096 KV tokens. Actual batch is unchanged. No native reference for this changed
arrival trigger was run in this turn.

Lint scope: new policy/controller/driver/tests pass targeted Ruff. Existing
unused imports in config/hooks and the pre-existing `full_bytes` F841 in
expert_hooks remain; they were not cleaned up as part of this change.
