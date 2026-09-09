# Native references for different decode concurrency regimes

The release-fix timed pair completes all rounds but differs in output tokens in
rounds2–3; its original cross-arm `valid=false` is preserved. Performance numbers
from that pair are not an accepted20% result.

## Serial reference

`/mnt/vdb/chengzhi/layerkv_retained_20260907_serial3_01/native.json` runs the
same three input rounds, BF16 model, output64, input128/6000 and1-second arrival
with native SGLang, `max-running-requests=1`. All LayerKV options are removed;
native is a correctness reference, not a matched-memory performance baseline.

Fixed retained rounds2–3 each match the native serial reference exactly for both
requests' token IDs and logprobs (256 total tokens). Round1 does not match the
short request: fixed ran B2 while this reference ran B1. Its short-request token
IDs differ and maximum logprob difference is1.284662; the long request is exact.
This independently reproduces concurrency-dependent token differences in native
execution, but does not yet validate the adaptive warm rounds.

## Diagnostic per-round trigger replay

The concurrent driver adds `--round-output-triggers`, one integer per round.
Zero inherits the base arrival settings; a positive value overrides that round
to submit the second wave after the first request reaches that output count.
Arguments are copied per round; overrides cannot leak into subsequent rounds.
Each output row records the effective time/output trigger. Invalid length,
negative/out-of-range triggers and an invalid initial-wave size fail before
launch. This is explicitly a diagnostic, not a fixed-arrival benchmark.

For two equal64-output requests, the adaptive histograms B1:16/B2:55 and
B1:26/B2:50 motivate trigger proposals8 and13 respectively. Histogram agreement
alone is not proof of identical ordered scheduler execution. Independent output
comparison and, if needed, scheduler trace validation must check the proposal.
The replay uses base1-second arrival with triggers0,8,13, native max-running2,
under `/mnt/vdb/chengzhi/layerkv_retained_20260907_native_replay_01`.

Driver unit tests:35 passed. Targeted Black/Ruff and `git diff --check` pass.
The diagnostic skill keeps original failing artifacts and their thresholds
unchanged while isolating native versus LayerKV behavior.

## Replay result: one localized failure remains

Native replay is terminal and its artifacts are complete. Against adaptive
`releasefix_01/lend.json`, rounds1 and3 match all tokens/logprobs exactly. Round2's
short request also matches exactly. Its long request differs first in token ID
at zero-based index58 (59th output): adaptive248068 versus native846. Native
round2 long output/logprobs match the fixed serial side exactly.

Near the B2→B1 boundary, both adaptive and native streams show the short request
finishing alongside long output56. This supports the proposed switching point,
but stream timing is not a complete ordered scheduler trace. Long output57
(index56) has identical token198 and logprob-4.768370445e-7. At output58
(index57), both choose248045 but logprobs differ (-0.000760862 versus-0.000423699).
At output59 the token choice diverges. This failure cannot yet be dismissed as
the previously demonstrated native serial/concurrent difference.

Next: isolate state/metadata/cache behavior immediately after the short request
finishes and before the long request's first differing B1 result. Preserve both
the token and pre-divergence logprob signal. Native replay is an output reference,
not a new performance result; the original timed pair remains invalid.

Full LayerKV validation with the new replay tests:372 passed,19 warnings.
No temporary runtime probes or GPU jobs remain from this turn.

## Deterministic reproduction and KV-path isolation

The following bounded runs retain the first two input rounds,64 output tokens,
and arrival settings1 second with per-round triggers0,8. Every run is terminal;
the native reference is the first two rounds of `native_replay_01`. Direct child
engine arguments and per-round effective arrival settings are preserved beside
outputs. These are correctness isolations, not paired performance measurements.

| Artifact suffix under `/mnt/vdb/chengzhi/layerkv_retained_20260907_` | Change | Result against native |
| --- | --- | --- |
| `adaptive_replay_01/lend.json` | Adaptive also uses triggers0,8 | Same long-request index58 token divergence, max logprob delta0.362092763 |
| `no_growth_01/fixed.json` | Fixed16 slots, zero extra slots, no expert loans | Identical failure; growth/recall not required |
| `no_evict_01/fixed.json` | Same fixed16 setup, reclaim budget0.000001 MiB | All256 tokens and logprobs exact |

The no-eviction run records zero `kvc_evict_count_total` and zero
`kvc_reload_required_count` in both rounds, not merely a configured disable
assumption. Expert demand loading remains enabled. All three runs have B2:63 in
round1 and B2:55/B1:16 in round2, so the observed batch histograms remain matched.
No-growth retains the same0.362092763 difference at the same location; this
rejects expert capacity growth/recall as a necessary trigger. The no-eviction
isolation places the remaining failure in the KV eviction/reload/reuse path or
its interaction with other runtime behavior; it does not yet identify a fix.

Next diagnostic: check returned overwrite locations against live cleanup owners
and inspect common/overwrite free-list alias consumption around the B2→B1
transition. Free locations are represented in ordinary lists, bitmaps and bit
chunks; duplicate publication/consumption is a candidate, not a proven cause.
Do not change allocator semantics solely from this hypothesis. The original
timed comparison remains invalid and its apparent throughput gains unaccepted.

## Ownership and write-address checks do not catch the model failure

Two further fixed16/no-growth reproductions use the same two rounds and
triggers0,8. `live_owner_01/fixed.json` checks every location returned by
`_pop_per_layer_overwrite_locs` against current request cleanup bitsets. No
conflict is found; long-request logprobs still first differ at index57 and token
IDs at58, with the exact previous differences. This rejects that specific live
overwrite-owner conflict as the observed mechanism, not all possible ownership
errors elsewhere.

`write_map_01/fixed.json` checks mapped decode KV writes against the corresponding
per-layer request table at `seq_len-1`. It only covers calls reaching the
nonidentity translation path with a layer override table. Every checked write
matches the table; the output discrepancy is unchanged, even with the probe's
extra synchronization. This is not proof that attention gathers the right
historical rows or that unmapped paths are correct. Both processes are terminal
and all temporary probes were removed afterward.

An independent CPU allocator repro publishes bit5 twice via
`_push_per_layer_overwrite_bits(3, 1 << 5)`, then calls
`_pop_per_layer_overwrite_locs(3, 1)` twice without an intervening free. Results
are `[5]`, `[5]`. Duplicate publication across chunks is therefore not idempotent
under allocation. It remains a separate correctness risk; the model live-owner
probe did not implicate it. Do not describe it as this model's proven root cause.

Next: compare selected per-layer activations against native immediately around
the surviving long request's transition to B1, to find the earliest attention or
metadata consumer where the numerical state differs. No production change or
accepted speedup results from these diagnostic checks.

## Layer snapshots localize the remaining failure to full-attention layer7

Explicit diagnostic flag `--activation-probe-seq-range 6053 6059` installs
keyword-aware forward hooks on Qwen3.6's40 decoder layers. It saves CPU copies
of hidden/residual outputs, sequence lengths, request pool indices and module
type. It is disabled by default, uses exclusive snapshot creation, and makes
parent comparison `valid=false`: synchronous snapshots are not performance data.
The existing three-argument hook API remains the default.

Completed artifacts under `/mnt/vdb/chengzhi/`:

- `layerkv_retained_20260907_activation_native_01/native.json` and
  `native.activations/` (560 snapshots).
- `layerkv_retained_20260907_activation_fixed_01/fixed.json` and
  `fixed.activations/` (560 snapshots).

Both use the documented two-round no-growth replay: BF16 Qwen3.6-35B-A3B,
requests2/max-running2, input128/late6000, output64, base arrival1second,
round-output-triggers0,8, baselinekv-heavy, fixed16/no extra slots. The literal
CLI `--expert-extra-slots 4` is ignored by the fixed kv-heavy arm as recorded in
engine arguments. Native is an output reference, not a matched-memory baseline.

The original symptom survives instrumentation: all other request outputs match;
round2 long logprobs first differ at index57, token IDs at58. Comparing the long
row of each captured hidden/residual output:

| Round2 long sequence length | Batch | Earliest differing layer | Max absolute hidden/residual difference there |
| --- | --- | --- | --- |
|6053,6054,6055|2|None; all40 layers exact|0 / 0|
|6056|1|7, Qwen3_5AttentionDecoderLayer|0.065765380859375 / 0.0859375|
|6057|1|7, Qwen3_5AttentionDecoderLayer|0.05126953125 / 0.0390625|
|6058|1|7, Qwen3_5AttentionDecoderLayer|0.099609375 / 0.0703125|
|6059|1|0; already after sampled-token divergence|Not root-cause evidence|

Layer0–6 remain bit-exact at6056–6058, including full-attention layer3.
Every layer7–39 output differs at6056. Thus the first observed divergence is
earlier than the first differing output logprob, at the first B1 step. This
narrows the next diagnostic to layer7's historical KV reads and attention
metadata at the transition; it does not yet distinguish a stale gather index
from corrupted/missing historical contents or a layer-local compute issue.
No numerical tolerance was relaxed and no production correctness fix is claimed.

Validation: full LayerKV suite376 passed,19 warnings. Targeted Ruff checks and
`git diff --check` pass. Both model children and the test process are terminal.

## Actual KV gathers identify live-address overwrite during reload

The next two completed BF16/no-growth runs retain the same replay inputs and
arrivals, but use `--activation-probe-seq-range 6055 6056
--activation-probe-kv-layers 7`. Artifacts are
`/mnt/vdb/chengzhi/layerkv_retained_20260907_kvgather_native_01/` and
`/mnt/vdb/chengzhi/layerkv_retained_20260907_kvgather_fixed_01/`.
The optional KV probe saves packed Triton indices/indptr and actual gathered K/V
from raw hybrid storage, avoiding public getters that trigger LayerKV preparation.

Both round1 snapshots and round2 B2 snapshot are exact against native. At round2
B1 length6056, exactly53 historical rows (positions6002–6054 inclusive) differ in
both K and V; maximum absolute differences11.828125 and3.80859375. The original
logprob/token divergence at57/58 survives. Those53 rows keep the same physical
indices across the transition (12287,12285,...); they are not a gather-index
change. There are no duplicate indices within the long request. The offloaded
prefill block at positions2048–4095 changes scratch addresses but retains exact
contents. Corrupted decode rows match earlier prefill values bit-for-bit, e.g.
position6002 now matches old prefill position4094,6003 matches4092,6054 matches3990.

The matching allocator defect is in `_reuse_evicted_per_layer_locs`: when the
original eviction block is no longer entirely overwrite-free, its fallback
checks only `_per_layer_arena_allocated_locs`. Decode overwrite allocation can
own locations solely through request cleanup bitsets, without that allocated
set. Reload can consequently reuse an old eviction address that now stores a
live decode token.

`test_reload_live_ownership.py` reproduces the real release→overwrite decode
allocation→cleanup ownership→original-address reload attempt. Both current and
legacy cleanup representations fail before the fix, returning `[100,101]`
instead of rejecting the range. The fix checks both ownership representations
before consuming any free addresses. A conflict returnsNone, allowing the
existing reload caller to allocate alternative locations. This does not disable
eviction or relax output checks, and leaves decode allocation unchanged.

Full suite after the fix:379 passed,19 warnings. Uninstrumented full-model
verification is tracked in `layerkv_retained_20260907_reload_ownerfix_01`;
its verdict must be checked separately before calling the model failure fixed.

Uninstrumented verification `reload_ownerfix_01/fixed.json` completed: all256
output tokens **and all output logprobs exactly match** the first two rounds of
the pre-existing `native_replay_01/native.json`. Both rounds report
`kvc_guard_pass=true` and `comparable=true`. Cumulative eviction counts20480/40960
and required reload token counts2048/6144 confirm that offload remains active.
Actual batch histograms remain B2:63, then B2:55/B1:16. Thus this is an original
model repro pass, not just a shallow allocator regression. No performance
acceptance is inferred from this no-growth/native correctness comparison.

Adaptive verification `reload_ownerfix_adaptive_01/lend.json` also completed,
uninstrumented, rounds3 with triggers0,8,13 and expert-heavy/adaptive retention.
All384 tokens and all output logprobs exactly match `native_replay_01`. Every
round reports KV/comparability/ownership/pointer guards true, physical pool
478150656 bytes and zero growth physical creates. Round2 actually grows16→20,
then recalls to16 for admission (cumulative grow1, admission recall1); final
loans are zero. Histograms are B2:63, B2:55/B1:16, B2:50/B1:26. Thus the fix
covers the dynamic expert-loan path as well as the minimized no-growth repro.

Both verification children are terminal. No temporary unconditional runtime
probes were introduced; the explicit activation/KV diagnostic remains off by
default. Next: fixed-arrival repeated matched-memory throughput measurements
with native schedule-specific correctness references. Neither this replay nor
old invalid gains establishes20% performance acceptance. Full-model FP16 remains
unverified; all full-model results in this section are BF16.
