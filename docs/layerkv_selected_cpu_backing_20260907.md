# Selected-layer immutable CPU backing

Motivation: mixed regrowth phase3 evicted650 cached CPU copies and performed622
eviction-time backing misses/D2H reconstructions, versus0 misses for fixed16.
See `layerkv_mixed_workload_20260907.md` for original counters and their meaning.
This is a scoped hypothesis about redundant immutable-weight backup, not proven
TTFT attribution or20% performance acceptance.

Added opt-in server `--layerkv-expert-cpu-backing-mode selected`; concurrent driver
uses `--expert-cpu-backing-mode selected`. Default remains none. Existing all
mode remains pageable/all-discovered-layers and is still disallowed in shared
VMM mode. Selected requires a shared layer and copies exactly its discovered
experts before compact installation. Missing/duplicate selection fails before
copying. Selected copies are pinned for the existing CUDA-batch transfer path.

Reuses the existing global backing lookup/lifetime semantics: slot eviction can
reuse the immutable CPU copy instead of reconstructing it via D2H. Cache aliases
may disappear without freeing permanent copies. No GPU pool sizing, KV admission
credit, expert routing or transfer kernel changes are made.

Host summary now separately reports immutable_backing_bytes/mode and counts
immutable storage once across global/state aliases. Mandatory/optional cache
ledger excludes these aliases; tracked_unique_storage_bytes includes their actual
storage. This is permanently retained host memory, not free optional-cache space
and not process RSS. For selected Qwen layer0,E256,H2048,I512,FP16, the full copy
is1536MiB pinned. Its incremental cost versus existing mandatory CPU backing
must be measured, not assumed to be an additional1536MiB in every phase.

Three focused tests passed: actual CUDA→pinned FP16/BF16 copies, selected-only
scope, immutable snapshot after source mutation, idempotence, global/state alias
deduplication and absent-layer rejection. Existing test suites and model runs
were not repeated. Changed new test/driver files pass Ruff; scanning whole older
mixins also reports existing unused imports and the preexisting injected
LayerKVRuntime name in a static helper. Those unrelated definitions were not
rewritten. `git diff --check` passed.

Next: validate effective server options and one adaptive mixed probe with selected
backing. Inspect D2H skips/copies, global lookup hits, CUDA-batch versus fallback,
host accounting and exact existing outputs. Shared GPU budget must remain fixed.
If promising, apply selected backing to fixed as well before making performance
claims. Do not enable all-layer backing or silently increase optional-cache limits.

## First selected-backing mixed model probe

Actual ServerArgs construction using the prior adaptive engine args plus selected
mode passed (shared layer0,optimized runtime). One adaptive child used the same
mixed horizon64 command plus `--expert-cpu-backing-mode selected`, fresh directory
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_mixed_selected_01`. It exited0; no fixed
or native rerun. Initial telemetry confirms256 experts preloaded,1536MiB pinned,
1107.99ms preload time before request measurement.

| Phase | Selected tok/s | Prior adaptive tok/s | Expert D2H copies | CUDA-batch H2D submissions |
| --- | ---: | ---: | ---: | ---: |
| Short (cold) | 82.919 | 68.757 | 0 | 100 |
| Long arrival | 23.561 | 23.227 | 0 | 140 |
| Short again | 97.838 | 97.012 | 0 | 206 |

Eviction-copy skips1896/1487/2813 and backing-cache misses0/0/0. Actual slots
49→16→49, cumulative grows1/1/2, one admission recall. All KVC/expert/ownership
guards pass, stable pointers, stale/alignment/physical failures0, shared physical
pool478150656bytes unchanged. Expert materializations1929/1487/2846; third-phase
count equals prior adaptive, so avoided D2H did not change incoming expert demand.

Tracked unique host storage1610612736bytes throughout, all immutable; optional
cache/pool usage0. Compared to prior adaptive, tracked host storage increases
41943040bytes (40MiB) in short phases and is equal in the long phase. Logical
ledger and optional-budget guards pass. These are tracked tensors, not RSS or
an independent host-memory peak measurement.

All1152 output token IDs match the prior adaptive run; short-phase1024 logprobs
are exact. Middle max logprob difference0.18318 and actual batch B1:32/B2:47
differs from the prior reference schedule: do not call full mixed output
validation complete or relax tolerance. Existing fixed/adaptive native evidence
does not automatically prove a newly observed schedule.

The intended redundant-D2H mechanism was eliminated, but final-short throughput
improves only about0.85% in this single launch. The first-phase increase includes
moving backing preparation before request timing; report the1.108s preload cost
separately and do not label it free throughput gain. This does not establish
D2H as the main mixed-workload bottleneck or achieve20%. Any eventual fixed
comparison must enable selected backing symmetrically and match memory costs.
