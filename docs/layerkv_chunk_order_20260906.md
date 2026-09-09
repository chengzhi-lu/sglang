# Prefill token-group profiling and resident-expert ordering

## Implementation and scope

`SharedExpertController.run_token_chunks()` keeps each token's complete top-k
reduction in one native MoE call, then scatters outputs back to their original
rows. This experiment retains the exact existing first-fit group membership,
token order within each group and expert capacity. Only group execution order
may change.

`--layerkv-shared-expert-chunk-order reuse` selects the next remaining group
with the fewest missing experts relative to **current** `logical_to_slot`.
Ties prefer larger resident overlap, then original group position. Residency is
read again after each group executes, so this does not assume ideal LRU or a
simulated eviction policy. The selection is deterministic, CPU-only and uses
expert bitsets already built by grouping. It adds no routing-data GPU reads.

Default order remains `input`; the optimization is an explicit experimental
option. Changing order changes the resident experts left for decode, which must
be included in end-to-end comparisons. This is not an optimal global ordering,
and fewer prefill misses do not guarantee fewer decode misses or lower latency.
The native MoE kernel, KV donor policy, memory budget, backing store and
admission logic are unchanged.

## Timing boundaries

`--layerkv-shared-expert-profile-chunks` enables profiling (off by default):

- `token_chunk_routing_wall_ms`: routing IDs transferred to a CPU list, including
  any existing wait this conversion entails.
- `token_chunk_group_wall_ms`: CPU first-fit group construction.
- `token_chunk_order_wall_ms`: reuse-order selection; zero in input mode.
- `token_chunk_{gather,prepare,moe,scatter}_wall_ms`: host elapsed intervals
  for index construction/gather, expert dispatch preparation, native MoE call
  and output scatter.
- Matching `*_stream_ms` fields: CUDA events around these four phases on the
  current stream. They include idle time while the host submits work and any
  stream waits. **They are not pure GPU kernel times or H2D-only durations.**
- `token_chunk_total_wall_ms`: complete chunked-call host interval, excluding
  collection of completed events from a previous batch.
- `token_chunk_materializations`: expert materialization counter increments
  specifically inside chunk preparation, rather than the whole request.
- `token_chunk_reordered_groups`: groups executed at a different ordinal.

CPU and stream intervals overlap; do not add them together. Preparation includes
range/unique-ID checks, residency updates, materialization, remapping and waits;
its wall time cannot be attributed solely to weight copies. Cold setup/JIT is
not separated from these phase timers. Profile-enabled runs may have additional
measurement overhead and must be compared with profile-enabled controls.

No per-group CUDA synchronization is added. Completed events are collected
with `query()` at stats reads and before the next chunked batch. Pending events
are retained, counted by `token_chunk_profile_pending`, and checked to be zero
by the benchmark's profiled-run verdict before interpreting per-request deltas.
Profiling-disabled runs do not create events; zero phase times then mean
profiling is disabled, not that the work costs nothing.

## Tests

The pre-change FP16/BF16 regression failed because a resident-expert group was
not selected first. After the change, the fixture needs four expert loads
instead of six, with identical outputs, complete top-k reductions, untouched
input activations and restored row order. The test exercises actual chunk
execution with an in-place synthetic MoE runner and changing resident mappings.
A CUDA regression verifies disabled profiling creates no events, and completed
events are collected only once. The LayerKV suite passed all 51 tests:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

## Matched replay

H100 NVL GPU 0; local Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`; BF16, TP=1, batch=1. Six sequential
requests per arm, 4096 synthetic input IDs and 64 greedy output tokens each.
Fixed16 then grow17, initial expert slots 16, shared expert layer 0, KV capacity
16384, context 8192, scratch 4095, KV block 2048, reclaim target 40 MiB.
Graphs/overlap/radix/chunked-prefill off, tokenizer on, per-token streaming on,
donor miss cache on, detailed chunk profiling on. Existing environment reused.

Artifacts:

- Input order: `/mnt/vdb/chengzhi/layerkv_chunk_input_20260906_01/`.
- Reuse order: `/mnt/vdb/chengzhi/layerkv_chunk_reuse_20260906_01/`.

The first request's prefill precedes compact expert installation, so it has
zero chunked-prefill calls. Request 2 is the first compact prefill. Different
rounds use different prompts and may see different cold shapes/setup; they are
not interchangeable steady-state samples. Missing scheduler prefill timestamps
remain null; client TTFT includes all layers and IPC, not just this expert layer.

Initial input-order measurement, fixed16 request 3: CPU grouping 19.119 ms,
expert preparation 793.123 ms, native MoE host interval 29.592 ms. The dominant
instrumented chunk phase was preparation, not group construction. Preparation
falls to 188.878 ms on request 6; the experiment does not assign that variation
to copies, JIT or another cause without finer measurement.

## Results

Both directories report `valid=true`, including completed profile events. Each
arm completed all six requests. Engine arguments for the same arm differ only
in chunk order (`input` -> `reuse`); fixed16/grow17 within each run differ only
in extra expert slots (0/1).

Fixed16 compact-prefill results, input -> reuse:

| Request | Prefill expert loads | Prepare wall (ms) | Client TTFT (s) |
| --- | ---: | ---: | ---: |
| 2 | 619 -> 454 | 2766.801 -> 1729.647 | 3.493 -> 2.444 |
| 3 | 617 -> 440 | 793.123 -> 894.793 | 1.719 -> 1.839 |
| 4 | 550 -> 389 | 396.750 -> 805.248 | 1.583 -> 1.983 |
| 5 | 578 -> 401 | 202.924 -> 153.522 | 1.369 -> 1.315 |
| 6 | 537 -> 371 | 188.878 -> 142.532 | 1.331 -> 1.268 |

Total prefill loads: **2901 -> 2055 (-29.2%)**. Reuse selection itself costs
1.2--2.1 ms per compact prefill. Group membership and MoE call counts remain
63/62/56/59/55 for requests 2--6. Request 1 uses no chunked-prefill path and is
excluded from the above table.

Grow17 compact-prefill results, input -> reuse:

| Request | Prefill expert loads | Prepare wall (ms) | Client TTFT (s) |
| --- | ---: | ---: | ---: |
| 2 | 619 -> 454 | 2700.902 -> 1612.631 | 3.440 -> 2.352 |
| 3 | 617 -> 440 | 834.709 -> 899.994 | 1.778 -> 1.853 |
| 4 | 550 -> 385 | 386.998 -> 602.949 | 1.483 -> 1.690 |
| 5 | 578 -> 401 | 253.272 -> 153.743 | 1.964 -> 1.209 |
| 6 | 537 -> 370 | 200.310 -> 142.625 | 1.685 -> 1.162 |

Total prefill loads: **2901 -> 2050 (-29.3%)**. Both arms show higher preparation
time and TTFT on requests 3/4 despite fewer loads. This one ordered, instrumented
comparison does not establish stable latency improvement or isolate cold
setup, synchronization and metadata costs inside preparation.

Decode is not invariant under this optimization. For example, fixed16 request
3's non-chunk materializations rise from 240 to 298 (request total minus measured
chunk materializations). Grow17 request 3 still borrows/returns three 2 MiB
pages, but borrowed-slot uses change from **43 to 18**. This is a changed cache
trajectory, not a lost physical loan; outputs still match. Large timing outliers
also occur in decode (e.g. fixed16 reuse request 4: 91.17 ms TPOT; input grow17
requests 4--6: 114.96/126.78/102.07 ms). Their cause is not attributed by these
prefill phase timers; do not infer a whole-run speedup from them.

All 384 paired fixed/grow output IDs and logprobs match exactly in each run.
Across orders, all 768 corresponding outputs/logprobs (both arms combined) also
match exactly, maximum logprob delta zero. KV/expert, pointer-stability and
ownership guards pass; every request finishes with zero host KV usage,
outstanding loans and blocked locations, 12289 free arena tokens, no pending
profile events and no admission-stall warnings. No growth-time physical pages
are created. Grow17 lends/recalls 6 MiB on requests 1 and 3 in both orders.

Decision: retain `input` as the default and expose `reuse` for controlled
experiments. The reduction in prefill loads is established for this trace;
stable end-to-end benefit is not. The next focused measurement should split
preparation into materialization/control, remap and waits, with cold/warm phases
separated, before changing defaults or expanding the matrix.

## Reproduce

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_chunk_reuse_new_run \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 \
  --chunk-order reuse --profile-chunks --execute
```

Use `--chunk-order input` in a fresh directory for the control. Remove
`--execute` for a dry plan; remove `--profile-chunks` for uninstrumented timing.
`PATH` exposes the reused environment's external tools such as Ninja; ordinary
experiment settings are CLI arguments. Engine arguments and plans are saved.
