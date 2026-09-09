# Adaptive residency: implementation and acceptance gates

Current correctness update: the B2→B1 reload-overwrite bug is fixed and verified
with exact native replay ([evidence](layerkv_native_replay_20260907.md)). The
previous FP16 limitation is also resolved for the bounded Qwen3.6 scenario:
FP16 native and adaptive outputs/logprobs are exact across256 tokens with real
expert16→20→16 borrowing ([evidence](layerkv_fp16_20260907.md)). Historical BF16
and invalid performance numbers below remain historical;20% is not yet accepted.

## Objective and measurement contract

At a fixed physical GPU memory budget, favor expert residency for short-context,
large actual decode batches; favor resident KV and return expert backing to KV
when long-context admission is constrained. Re-evaluate after admission changes.
These are workload hypotheses, not an unconditional context-only threshold.

### Workload policy contract

- Short contexts and large actual decode batches favor retaining the measured
  expert working set when this reduces repeated transfers, subject to active KV
  growth headroom and queued-request admission needs.
- Long contexts and small batches favor returning expert backing to KV when a
  queued request is genuinely KV-capacity-limited. Preserve live KV and count
  only physically remapped, allocator-visible capacity as admission credit.
- Context length alone does not determine the split: track actual batch, live
  KV bytes, queued capacity demand, expert misses and transfer time. With no
  waiting work, freeing memory does not itself increase batch or throughput.
- Add hysteresis and a minimum residence interval to avoid repeatedly reversing
  a loan. An admission shortage overrides retention; subsequent expert regrowth
  requires renewed headroom and evidence of benefit.
- Evaluate output throughput and per-token latency separately: a larger actual
  batch can improve aggregate decode throughput while increasing step latency.

This is the intended controller contract, not a claim that the current fixed
extra-slot lending controller already implements these decisions.

The adaptive shared controller now includes a bounded context-pressure guard:
at a decision point it compares the current CPU-visible KV demand (the sum of
active request context lengths plus current writes) plus
`batch_size * headroom_steps` against **effective** KVC capacity. Effective
capacity starts from the native allocator total and subtracts virtual scratch
tokens and locations currently lent to expert weights. If that projected demand
exceeds capacity, it selects base expert residency and recalls outstanding
expert loans. This uses context length through actual KV demand rather than a
hard context threshold; physical donor eligibility and native scheduler
admission remain authoritative.

A context-admission experiment also tested converting virtual scratch into
persistent KVC. It reached the native allocator ceiling but did not create
overflow capacity: a 4096-token workload remained at decode batches `3/3/2`
because four requests also need output reservation. The conversion added
synchronous reload cost and was removed from the runtime path. Future long-
context capacity work must add real overflow backing or lower scratch before
startup; it must not count scratch conversion as newly created KV memory.

The first shared compact resident set is also calibrated from the current
prefill hotness snapshot before installation. This is a one-time synchronization
of online routing metadata, not a dataset-specific expert whitelist; the
summary records the selected IDs and whether the source was online hotness,
candidate ordering, or the deterministic ID tie-break fallback.

The provisional performance target is at least 20% higher output tokens/s than
a fixed residency split, with identical model, offered input/output workload and
physical memory budget. Report TTFT and TPOT separately, including regression.
The user has not specified a latency SLA or one required fixed-split baseline.
Do not count microbenchmark gains, increased CLI limits, or an invalid-output
run as acceptance. Compare reachable actual batch, not just configured batch.

## Implemented prerequisites

- Opt-in `--layerkv-shared-expert-free-kv-donors` admits wholly free, reserved KV
  pages as expert backing, in addition to the existing backed-offload donors.
- Donor claims fail closed on missing ownership, live/protected/cleanup locations
  or partial claims. Claimed addresses leave scheduler KV credit before unmapping.
- Recall remaps the same physical pages before restoring KV allocator credit.
  FP16 and BF16 CUDA roundtrips exercise exact credit removal/restoration, stable
  pointers, physical-create counts and rollback after injected mapping failure.
- Actual decode ForwardBatch CPU metadata now records forward count, request
  steps, maximum and a batch-size histogram. No GPU metadata read is added.
- `scripts/layerkv_concurrent_perf.py` submits multiple requests, records per-request
  streaming events, batch makespan throughput, output logprobs and runtime guards.
  It is a bounded fixed/lend feasibility driver, not yet an adaptive-policy test.
  Dry planning is the default; effective engine arguments are recorded explicitly.

The module interface separates page ownership/allocator credit from budget
selection. A loan cannot simultaneously count as resident expert capacity and
available KV capacity. Native SGLang retains request admission ownership.

## First concurrent feasibility run: NOT accepted

Artifact directory:
`/mnt/vdb/chengzhi/layerkv_concurrent_free_20260907_b4_01`.

Qwen3.6-35B-A3B BF16, H100 NVL GPU0, TP1, input 2048, output 16,
4 offered requests / max-running 4, 2 rounds, max-total-tokens 16384,
scratch 8192, block 2048, reclaim 40 MiB, selected expert layer 0,
initial 16 slots, fixed extra 0 versus lending extra 1. CUDA batch transfer,
individual backing, stream-ordered D2H, admission-time backing release validation,
batch remap, CPU-known prefill preparation, input group order; cache/pool 128 MiB
each. Both arms use the same settings except extra slots. Plan JSON stores the
exact launch commands. Full-model FP16 remains separately unvalidated due to the
previous GDN dtype issue; FP16 mechanism tests do not resolve that limitation.

| Round | Actual decode batches | Fixed output tok/s | Lend output tok/s | Verdict |
| --- | --- | ---: | ---: | --- |
| 1 (cold) | 15 steps at B3, then 15 at B1 | 8.9583 | 9.0496 | Request 0 diverges; invalid |
| 2 | 30 steps at B2 | 11.5367 | 7.5612 | Outputs exact, 34.46% throughput regression |

Both rounds/arms have mean actual B=2. Round 1 output divergence first appears
in request 0's fourth output logprob and fifth token ID. The other three requests
match. All round-2 outputs/logprobs match. KV/expert/ownership guards pass, proving
that aggregate guards alone are insufficient for output correctness.

Both arms report the same VMM physical backing: 377487360 bytes at startup and
478150656 after the first round (including lazily materialized backing). Lending
does not add physical creates: one 3-page loan is returned, peak expert capacity
17. The second round has no additional successful loans but performs 30 donor
scans totaling 2982.84 ms. This is performance evidence for investigating the
scan path, not a claim of a verified fix or an adaptive-policy result.

The initial driver's per-round histogram subtraction assumed JSON string keys;
Engine returns integer keys in-process. Raw cumulative runtime counters are
correct; the initial round-2 derived histogram incorrectly retained the previous
B3/B1 entries. A CPU regression test exposed this; key normalization fixes the
driver. The table above uses differences of raw cumulative histograms. Existing
raw artifacts are preserved, not silently overwritten.

## Concurrent correctness isolation and scan fixes

The reduced command uses the same driver/settings with `--requests 3 --rounds 1
--output-tokens 6`. It reproduces the same fourth-logprob/fifth-token discrepancy
in request 0, with **zero successful page loans**. Full model setup takes tens
of seconds; generation itself takes approximately 4 seconds. This differential
loop is retained because aggregate memory guards did not detect the symptom.

| Artifact suffix (under `/mnt/vdb/chengzhi/layerkv_concurrent_free_20260907_`) | Single change | Output result |
| --- | --- | --- |
| `b3_min_01` | Reduced workload | Same divergence |
| `b3_no_reserve_01` | Omit donor arena preparation | Same divergence; hypothesis rejected |
| `b3_poll_01` | Poll instead of wait for eviction completion | Same divergence |
| `b3_no_finalize_01` | Do not publish eviction completion from donor path | Exact tokens and logprobs versus fixed reference |

The last diagnostic runs only the lending child, retaining the fixed reference
from `b3_min_01`; it is an output diagnostic, not a new matched performance pair.
Polling still publishes ready entries, so avoiding a blocking wait alone did not
remove the trigger. `after_decode()` is invoked after KV eviction submission;
publishing newly submitted eviction state there changes the normal KV lifecycle.
The fix removes donor-side publication and consumes only previously published
free pages. This isolates the triggering call site; it is not a proof that all
arbitrary KV-finalization reorderings elsewhere would be safe.

`test_donor_path_does_not_publish_pending_kv_evictions` failed with the old call,
then passed after removal. The older miss-cache test now proves that a completion
published by the normal lifecycle invalidates the cache, without requiring donor
discovery itself to finalize. No temporary runtime probe flags remain.

CPU profiling of a fragmented 10-layer allocator (`test_donor_scan.py:controller`)
found redundant encoding of allocated addresses and repeated wide-bigint expansion.
Ten real `_donors()` calls took 618 ms under cProfile; avoiding duplicate exclusion
and expanding dense bitsets by byte reduced this to 74 ms. This is a Python-path
diagnostic, not end-to-end model speedup. Large location sets now also use packed
byte encoding; small encodings and sparse/limited expansions retain their cheap
original behavior.
Tests cover unordered inputs, duplicates, filtering, sparse high locations and
allocation limits, plus active/protected/cleanup exclusions for backed donors.

The original 4-request, 2-round workload was rerun as `b4_fixed_01` after the
publication fix, redundant-exclusion removal and dense bitset expansion (before
packed encoding): **valid=true**, all 128 output tokens and logprobs exact,
matching physical pool bytes and all guards passing. Actual batches remain B3/B1
then B2; they have not grown. Fixed/lend throughput is 10.2267/10.0856 tok/s in
round 1 and 19.2208/14.2088 in round 2. Lending still regresses 26.08% in round 2;
30 unsuccessful donor scans consume 1143.58 ms. This result does not satisfy the
research objective. The packed-encoding follow-up is recorded separately.

### Latest matched follow-up

`b4_packed_01` includes packed large-location encoding. It is another fixed-first
single pair, not a reversed-order repeated acceptance study. All 128 output tokens
and logprobs are exact across arms, `valid=true`, physical pools match, and all
recorded guards pass. The final VMM physical backing is 478150656 bytes in each
arm; three pages are lent and returned with no growth physical creates and no
outstanding loans. Batch sizes are unchanged from the earlier pair.

| Round | Fixed output tok/s | Lend output tok/s | Lend relative throughput |
| --- | ---: | ---: | ---: |
| 1 (cold) | 10.4114 | 10.1781 | -2.24% |
| 2 | 19.2134 | 15.1354 | -21.22% |

Round-2 TPOT is approximately 69.6–70.2 ms fixed versus 98.9–101.3 ms lending.
Its 30 unsuccessful scans still consume 901.09 ms, down from 2982.84 ms in the
first implementation but not eliminated. Improving an always-attempt-lending
path is not the desired workload-adaptive policy. In particular, this B2 phase
should not repeatedly pay to seek more expert backing without a demonstrated
working-set benefit or available whole pages.

## Remaining gates

An initial [adaptive budget controller](layerkv_residency_budget_20260907.md) now
uses observed expert reuse, projected context demand, actual KV headroom and
cooldown, with physical CUDA tests and a valid model pair. It still regresses
throughput and does not enlarge actual batch. Cost-aware selection, page-capacity
scaling and cross-request retention remain required; a shadow-cache miss
benefit alone is insufficient.

The staggered short/long recall/regrowth correctness failure is now fixed:
write-time address translation was deleting valid cyclic mappings. The original
32-output-token pair passes exact outputs/logprobs with regrowth enabled; see
[the isolation and regression evidence](layerkv_staggered_20260907.md).
The complete LayerKV suite passes 316 tests. This resolves that specific blocker,
not the adaptive-controller or throughput requirements below.

1. Preserve the now-passing concurrent fixed/lend output regression; supplement
   it with an independent native-reference comparison for this B3 workload before
   treating it as complete model correctness. Eliminate remaining
   unsuccessful free-page scan overhead with an
   ownership-safe correctness test; do not cache positive donor addresses.
2. The [admission recall interface](layerkv_shared_admission_20260907.md) is now
   implemented with real CUDA page/admission tests. Prove more requests actually
   run in the full model, not merely higher credit or a unit-level decision.
3. Keep safe expert residency across suitable short-context request boundaries,
   with enough KV headroom for active decode and queued admission. Current code
   still recalls all loans before prefill/request completion and grows one slot
   per decode step; this is not the desired adaptive controller.
   [Per-layer proven-free eligibility](layerkv_per_layer_donors_20260907.md) now
   allows reuse of addresses already lent in another layer, with strict claims,
   cleanup exclusions and nonduplicated admission credit. A valid B8 model pair
   reaches 24 slots. [Batched target growth](layerkv_batch_growth_20260907.md)
   reduces eight transactions to one, but the current B8 pair still regresses
   6.79%. Cost-aware acceptance and the expert-heavy admission baseline remain.
4. Add workload/budget selection with hysteresis and real demand feedback; compare
   short/large-B and long/small-B traces at matched physical budgets and several
   fixed residency splits. Expand selected-layer coverage only after correctness.
5. Repeat matched unprofiled runs with reversed arm order, report cold/warm phases,
   actual batch distributions, output tokens/s, TTFT/TPOT, swap volumes, physical
   budget and guards. Only then evaluate the 20% target.

## Validation at handoff

From `/home/chengzhi/github/sglang-perf-layerkv/python`:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

293 passed, 19 pre-existing/environment warnings. Targeted Black/isort/Ruff and
`git diff --check` pass. Full-repository pre-commit was not run. Existing packages
were reused, no CUDA extension compiled, no commits/staging or unrelated dirty
changes discarded. All experiment/test processes from this work have completed.
