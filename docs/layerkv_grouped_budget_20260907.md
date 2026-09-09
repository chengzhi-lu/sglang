# Group-aware residency budget: correctness and B8 regression

The budget now replays CPU-known routing rows using the same first-fit token
grouping as execution, at base and maximum expert capacities. Shadow replacement
protects all experts required by the current group. This fixes the unique-ID
stream's inability to model within-forward group reuse, but remains a proxy:
production replacement uses decode-step timestamps and slot-ID tie breaks.
The input-order grouping model does not model the optional reuse group order.

Decode chunk checks and execution share a single routing readback, in both fixed
and adaptive policies. The snapshot is consumed by tensor identity and cleared;
prefill falls back to its existing readback. No extra GPU routing readback is
introduced. Pressure and measured growth-cost gates are unchanged.

Validation command (from the worktree's `python/` directory):

```
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest ../test/registered/unit/layerkv/ -q
```

Result: **396 passed, 19 warnings**. New tests cover the unique-stream blind spot,
first-fit grouping, invalid IDs, duplicate observations, KV-pressure priority,
and single-readback snapshot behavior for fixed/adaptive policies.

## Full-model measured result

Artifacts: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_grouped_01`.
The directory contains effective arguments, logs, responses, stats and summary.
Qwen3.6-35B-A3B FP16, H100 NVL GPU0, TP1, selected expert layer0, simultaneous
B8, input128/output64, five rounds. Fixed16 versus adaptive16..24, KV-heavy
baseline, retain experts across requests. Other settings match
`layerkv_shortb8_retention_20260907.md`. Rounds1–2 are excluded as startup/warmup.

| Warm rounds3–5 | Fixed16 | Adaptive16..24 |
|---|---:|---:|
| Aggregate output tokens/s | 99.29531 | 93.39447 |
| Token-chunk materializations | 11409 | 9421 |
| End-of-round expert slots | 16 | 24 |

Throughput **decreases 5.9427%**, despite 17.42% fewer materializations. These
are end-to-end output throughput and prefill-plus-decode chunk materializations,
not isolated decode kernel throughput or decode-only transfer counts. All five
rounds have exact cross-arm tokens and logprobs, guards pass, physical pools
match, and actual decode batch is8 for63 steps each round. There is no independent
native B8 reference yet. Single-launch pilot, not20% acceptance.

Adaptive grows once in round1, retains24 slots through all round-end snapshots,
and reports positive shadow miss savings. Thus the observation blind spot is
addressed, but the performance objective is not. Do not count fewer swaps as
a throughput win or relax correctness gates to accept this result.

## Next diagnostic

Matched fixed24 versus adaptive16..24, expert-heavy baseline, same workload and
warmup convention, planned/executed in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_fixed24_01`.
This distinguishes adaptive control overhead from the24-slot execution shape;
both policies retain loans and no late requests are submitted. Fixed capacity
grows during startup rather than being preconditioned. Inspect achieved capacity,
outputs and physical guards before interpreting warm performance.

Candidate control costs include grouped shadow observation and per-step exact
common-KV-address discovery while a loan is outstanding. A synthetic CPU B8
observation loop measured about137us/step (10000 iterations, random8-of256 rows);
it is only an order-of-magnitude check, not attribution from the model trace.

The fixed24 diagnostic completed with strict summary valid and matched pools.
Warm aggregate throughput: fixed24 **102.70427**, adaptive **91.75305** tokens/s.
The fixed24 warm outputs/logprobs also match the preceding grouped adaptive run
exactly. Warm materialization counts are identical at the24-slot capacity. This
points to adaptive control overhead rather than simply an unfavorable24-slot
MoE shape, although independent/reversed repetitions remain outstanding.

## Bounded headroom predicate

Added `_has_common_per_layer_reusable_tokens(count)`, used only by the adaptive
post-decode headroom check. The previous path expanded every layer's lazy free
bits into address lists and sets on every decode step with an outstanding loan.
The new pure-bitset path intersects the free bits and verifies only `count`
candidate addresses against current reserved/allocated/protected ownership.
Mixed representations or an unsuccessful ownership witness fall back to the
existing exact query. Insufficient raw common bits prove insufficiency directly.
Native free capacity is a sufficient lower bound when it alone covers reserve;
it was zero in this B8 experiment, so that shortcut alone is not the solution.

This predicate neither allocates addresses nor changes admission credit, donor
selection, physical claims or recall. The budget only uses free capacity for a
threshold comparison; it receives a proven reserve-sized lower bound or zero.
Queued admission shortage still overrides retention independently.

Tests first failed on unnecessary exact discovery and the missing predicate,
then the full LayerKV suite passed **401 tests**. Additional deterministic
fragmented-ledger coverage checks the predicate against the old exact oracle.
A CPU mechanism comparison with10 layers and8192 lazy free addresses/layer,
reserve128,100 queries measured **13.4413ms exact vs0.3570ms bounded**. This is
not GPU-model performance attribution. The original full B8 workload is rerun
in `/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_headroom_01`; inspect that
artifact's terminal result before claiming the regression is resolved.

The rerun is now terminal: strict summary valid, pools matched and all output
tokens/logprobs exact across arms, but warm throughput is96.81278 fixed versus
92.03297 adaptive tokens/s (-4.9372%). **The original full-model regression is
not resolved.** The synthetic
pure-bitset improvement does not establish that the real model takes that path.
Next distinguish mixed-list fallback and failed ownership-witness fallback from
other adaptive control costs with scoped runtime evidence; do not claim the
headroom predicate is a successful model performance fix yet.

## Real representation diagnosis and list-backed witness

A bounded3-round adaptive diagnostic completed in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_headroom_probe_01` (direct child invocation;
engine arguments/responses/stats persisted, console probe captured by the command
session). At09:03:06 UTC the one-shot probe reported:

```
[DEBUG-headroom-path] mixed layer=3 free=3000 pending=0 bitmap=False count=128 exact_ms=3.447
```

This directly establishes that the pure-bitset shortcut missed a real ordinary
free-list representation. The diagnostic's timings are not performance acceptance.
All temporary probe code was removed before the next performance run.

The witness now unions at most `count` entries from each ordinary/pending list
with that layer's lazy free bits, then intersects across layers and checks
reserved/allocated/protected ownership. It is a sufficient lower-bound witness:
truncated lists and legacy bitmaps can only cause an exact fallback, never a
false insufficient-capacity verdict. No address cache or dirty-generation
assumption is introduced. Positive capacity evidence never authorizes a physical
claim; existing exact donor/admission checks still do that separately.

Three new ordinary/pending/mixed3000-entry regression cases failed before this
change and pass afterward. Full LayerKV validation: **405 passed,19 warnings**.
The previous deterministic fragmented-ledger oracle comparison remains passing.
The original5-round B8 pair, without probes, is rerun in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_headroom_lists_01`.

The uninstrumented list-witness rerun completed: strict summary valid, physical
pools matched, all2560 cross-arm tokens and logprobs exact. Warm rounds3–5
aggregate throughput **97.96833 fixed16 vs100.69787 adaptive24 tokens/s (+2.7861%)**.
Warm chunk materializations remain11409 vs9421, actual batch stays8 throughout,
and adaptive retains24 slots. Thus the observed B8 regression is resolved in
this single-launch rerun; it is not20% acceptance, an independent native check,
or proof of stability across launch order. A CPU ordinary-list mechanism check
(10 layers,3000 entries/layer,reserve128,100 queries) measured2.4196ms exact
versus0.7333ms bounded, consistent with the targeted control-cost explanation.

Next bounded capacity probe: same workload and physical budget, adaptive maximum
64 (base16 plus48 extra), in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_slots64_01`. The planner must
still enforce actual post-loan KV headroom; requested maximum is not proof of
achieved capacity. This is one capacity point, not a matrix or main trace result.

## Requested64, achieved29: capacity limit

The64-upper-bound run completed with strict summary valid and matched pools.
Warm throughput: **97.99962 fixed16 vs101.91767 adaptive tokens/s (+3.9980%)**.
All rounds ended at **29**, not64, with39 borrowed2MiB pages (78MiB), one growth,
and no physical creates during growth. There were40 donor scans,39 misses,
494.90ms cumulative scan time over5 rounds. Warm chunk materializations were
11409 fixed versus8422 adaptive. The budget still estimates benefit at64, so
its estimate must not be interpreted as realized29-slot savings.

The terminal stats show8192 virtual scratch tokens reserved and0 scratch tokens
used in this short-context scenario. `planner._ensure_virtual_scratch` reserves
these addresses separately from the ordinary per-layer arena; the donor path
only considers arena-reserved addresses. Scratch has real reload consumers and
cannot simply be donated without a restore-before-use lifecycle.

Next diagnostic changes only the existing explicit scratch reservation argument
to2048 in both arms, leaving KV buffer size/physical allocations unchanged:
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_scratch2k_01`. Same B8, input128,
output64,5 rounds, adaptive16..64. This tests the cost of idle scratch reservation,
not a final adaptive scratch implementation or validation of long-context reload.

The scratch2k pair is terminal: all5 rounds have exact cross-arm tokens/logprobs,
strict summary valid, physical backing478150656 bytes in both arms throughout.
Warm throughput **95.18488 fixed16 vs104.17639 adaptive49 tokens/s (+9.4464%)**.
Adaptive reaches49 slots once and retains them. This supports the idle-scratch
capacity hypothesis (29 to49), but not20% acceptance or dynamic scratch safety.
The previous adaptive29 run was101.91767 tokens/s: the cross-launch increase to49
is only about2.2%, with no independent repetitions. Do not credit the entire
9.45% matched-arm gain to scratch reclamation alone.

Warm total expert materializations are11433 fixed versus5741 adaptive49. The
chunk-only counter falls to1939 but excludes non-chunked forwards at larger
capacity and therefore overstates the overall reduction if used alone. Measured
expert materialize time is1470.32ms versus631.48ms across the three warm rounds;
main-stream wait is808.10ms versus530.85ms. These timings overlap and must not be
added as an end-to-end decomposition. Adaptive still spends344.76ms on24 donor
scans after capacity saturates. Existing telemetry does not explain the rest of
roughly15 seconds of warm generation; absent profile fields are not zero cost.

Per user direction, no additional capacity sweeps or duplicate full-model tests
are launched. Before implementing the more involved scratch-sharing lifecycle,
use one bounded CPU/GPU profile of the already established warm workload to
identify the dominant remaining decode cost. Reuse these output references and
matched baselines; profiling timings themselves are not performance acceptance.
