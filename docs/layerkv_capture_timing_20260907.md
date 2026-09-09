# Capture-cost attribution, not a new throughput comparison

Added cumulative native-MoE capture_wall_ms,capture_sync_ms,capture_context_ms
and recapture_count. Timers execute only on the cold capture path, not replay.
No extra CUDA synchronization or GPU event is introduced. Sync/context timings
are subsets of the host-wall total, not additive kernel times. A CPU fake-CUDA
test passed, checking deterministic timing, replacement count and exactly the
existing one synchronization per capture. Ruff/diff checks passed.

One diagnostic adaptive child in
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_capturetime_01` used identical
selected-backing/horizon64 mixed arguments to `mixed_selected_01`. This rerun
collects timing evidence unavailable in old artifacts; fixed and native were
not rerun. Process exited0. All1152 token/logprob pairs match mixed_selected_01
exactly (including its middle-phase B1:32/B2:47 schedule). This is an exact
comparison to that artifact, not new schedule-matched native evidence for it.
KVC/expert/ownership guards pass, faults0, physical pool478150656bytes, slots
49→16→49. No performance-acceptance claim is attached to this diagnostic.

| Phase | Captures | Recaptures | Capture wall ms | Existing sync ms | Capture context ms | Whole phase ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 39 | 0 | 81.147 | 0.761 | 39.080 | 6319.935 |
| Long | 117 | 117 | 240.774 | 2.732 | 110.825 | 5434.458 |
| Short again | 39 | 39 | 79.007 | 0.904 | 36.444 | 5233.078 |

Final-short growth costs158.197ms. Its79ms capture cost is only1.51% of phase
time; intermediate captures total4.43% of that phase. A multi-shape cache still
needs first captures of B8/B1/B2; it would avoid repeated B1/B8 captures, not all
195 captures. The observed magnitude does not justify a graph-cache redesign
as the main route to20%, especially with additional graph-private memory.

Next attribution target: selected-layer prefill before regrowth. The current
controller grows only after decode observations; final-short prefill therefore
runs at16slots before regaining49. Existing phase totals cannot isolate its
chunking/prepare/core contribution, and graph timing does not explain the large
TTFT gap. Measure that specific boundary before implementing prefill-time growth;
any earlier borrowing must still protect already admitted KV and preserve the
no-publication donor lifecycle. No new growth mechanism is implemented here.
