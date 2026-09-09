# Native MoE replay seam: mechanism implemented, integration pending

Motivation: the bounded B8 decode profile in `layerkv_decode_profile_20260907.md`
shows a large host/submission component. Shared mode currently disables whole
and piecewise CUDA graphs. Do not enable either globally around mutable residency.

## Inspected interface

`FusedMoE.forward_impl()` performs dispatch, `run_moe_core`, combine and optional
parallel reduction. The hotness probe wraps `run_moe_core` and invokes its saved
original callable after collecting routing observations. Shared mode's expert
plan only installs the selected layer; other expert weights remain resident.
This makes the original native core a smaller potential replay seam than a
decoder layer or transformer block. KV attention, metadata, reload, routing,
hotness, dispatch/combine and parallel communication can stay outside replay.

The selected shared expert layer must be excluded; it changes weights/capacity
and requires host materialization. Any other installed/offloaded expert state
must also be excluded. The normal inference mode, standard dispatch, unquantized
FP16/BF16 and immutable resident weights are preconditions, not consequences of
successfully replaying a toy kernel.

## Implemented but not installed

`python/sglang/srt/layerkv/native_moe_graph.py` adds `NativeMoEGraph`, a small
callable around a supplied core and parameter provider. It has one exact-shape
entry, bounded batch size, eager warmups, input copies and pointer/shape/stride/
dtype/device fingerprints. Weight storage is retained while its graph can still
be in use. Shape/weight changes use eager warmup and synchronize before replacing
the old entry. Capture failures are explicit, not silently swallowed.

Native MoE may mutate its input inplace. Replay copies the resulting hidden input
back and preserves output aliasing; out-of-place output is cloned so later replay
cannot overwrite a previously returned value. Dynamic routing tensors are copied
each invocation. Graph workspace allocation is measured separately from the VMM
pool; a future performance comparison must account for this extra GPU memory in
both fixed and adaptive arms.

No server flag, runtime installation or whole-model graph behavior is changed yet.
This is not a performance result or proof of native Triton capture compatibility.

## Focused validation performed

From the worktree's `python/` directory:

```
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest ../test/registered/unit/layerkv/test_native_moe_graph.py -q
```

Two GPU mechanism cases passed (inplace and out-of-place). They exercise changing
routing data, exact outputs, input mutation, previous-output lifetime, shape
replacement, weight storage replacement and the batch-limit fallback. The test
core is a small matmul/routing calculation, not the model's MoE implementation.
No full-model launch, parameter sweep or full regression rerun was performed.

## Remaining implementation gates

1. One isolated actual Triton MoE core check at the existing FP16 B8 shapes,
   reusing packages/kernel caches. Check exact eager/replay results before wiring.
2. Explicit opt-in installation only for fully resident native cores in shared
   mode; hotness wrappers remain outside, selected/offloaded layers are excluded.
3. Expose replay/capture/fallback counts and additional graph workspace bytes.
   Guard unsupported quantization, distributed/overlap modes and graph nesting.
4. Then one bounded model validation using existing output references, not a new
   capacity matrix. Apply the common optimization symmetrically when eventually
   comparing static versus adaptive residency; do not credit a graph-only gain
   to the residency policy or call the20% objective complete prematurely.

## Actual Triton core gate completed

Added one focused `test_real_triton_b8_core_replay` using the actual
`fused_experts` implementation and standard dispatch/top-k/combine types:
256 experts, hidden2048, intermediate512, top-k8, FP16, B8. Synthetic seeded
weights occupy1.5GiB; no checkpoint/server is loaded. It uses the existing
environment and the same kernel shapes already exercised by the model runs.

```
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH /mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest ../test/registered/unit/layerkv/test_native_moe_graph.py -q -k real_triton --tb=short --disable-warnings
```

Run from the worktree's `python/` directory. PATH selects existing external
compiler/framework executables, not experiment configuration.

The initial fixture failed in the eager reference before capture: its scheduler
argument stub lacked `enable_deterministic_inference`. Inspection of the kernel
configuration readers identified that missing field; the fixture now sets it
false, matching the prior model runs, together with disabled fused sum/all-reduce.
No runtime/graph/kernel implementation or comparison tolerance was changed.

Corrected focused result: **1 passed,2 deselected** in12.85s. Four changing-routing
inputs match eager exactly (`rtol=0,atol=0`), are finite, preserve inplace aliasing,
and do not overwrite previous outputs. The assertions require1 capture and2
actual replays. This completes the isolated Triton compatibility gate, not the
runtime-installation, memory-budget or full-model performance gates. No broad
test suite or full-model benchmark was launched for this check.

## Opt-in runtime installation

Added `--layerkv-native-moe-graph-max-batch-size N` (default0, disabled).
Validation requires enabled LayerKV with a selected shared expert layer, TP/EP1,
disabled overlap, and disabled whole-model/piecewise CUDA graphs. Installation
only targets CUDA FP16/BF16 unquantized local Triton cores outside the selected
layer. Hotness sampling stays outside replay; prefill, extra call arguments and
subsequently offloaded layers use the original eager core. The saved original
core remains unmodified for the existing expert-install path.

Runtime summary includes installed layers, captures, replays, graph-internal
fallbacks and allocated workspace deltas. The workspace field explicitly is NOT
total physical memory accounting: graph-private allocator reservations and peak
GPU usage still need inspection before any fixed-budget performance acceptance.
No full-model correctness or speed claim follows from this installation.

Focused CPU installation tests cover option rejection, wrapper ordering/eager
bypass, and supported-core eligibility. Existing successful GPU replay tests
were not repeated. Next gate: driver opt-in propagation and one bounded model
validation with existing references, including graph counts and physical memory
telemetry; keep graph optimization symmetric between static/adaptive arms.

## Bounded model probe and driver accounting

The concurrent driver now accepts `--native-moe-graph-max-batch-size` and sends
the same value to fixed/adaptive arms; native reference stays eager. Graph runs
are diagnostic-only, including individual round timing eligibility. Summary
rejects performance acceptance when graph mode appears either in CLI arguments
or saved raw snapshots, even if shared-VMM physical pools match. Seven new
focused driver checks passed (three CLI-budget guard cases, three artifact-based
guard cases and one symmetric-propagation check); existing tests were not rerun.

Enabled runtime summaries now include Torch allocated/reserved and lifetime peak
allocated/reserved bytes, plus device-wide used/total bytes. Device-wide readings
include other processes; they are snapshots, not sampled peak physical usage.
Graph workspace deltas alone still do not establish a fixed physical budget.

One adaptive-only probe was launched after a dry plan and idle GPU check:

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH /mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_nativegraph_01 \
  --dtype float16 --requests 8 --max-running-requests 8 --rounds 3 \
  --input-tokens 128 --output-tokens 64 --scratch-tokens 2048 \
  --expert-extra-slots 48 --expert-policy adaptive \
  --retain-experts-across-requests --native-moe-graph-max-batch-size 8 \
  --execute --arm lend
```

The output directory was created fresh before this child invocation. Reuse the
first three rounds of `layerkv_fp16_20260907_shortb8_scratch2k_01/lend.json` for
output/logprob comparison. This does not provide an independent native reference,
a contemporaneous fixed arm, or a dynamic-residency throughput acceptance result.

### First probe exposed an installation filter bug

`b8_nativegraph_01` exited successfully with1536 exact token/logprob pairs and
all KVC/expert/ownership guards passing, but installed_layers was empty and both
capture/replay counters were0. Therefore it exercised eager, not graph replay;
its timings are not graph evidence. The physical pool remained478150656bytes
and the selected layer retained49slots.

Root cause: `module.use_triton_kernels` denotes the distinct `triton_kernel`
backend, not the `triton` backend used here. A CPU regression using the real
`MoeRunnerBackend.TRITON` enum failed before the fix. Eligibility now checks
`quant_method.runner.runner_backend == MoeRunnerBackend.TRITON`, still excluding
the other backend, distributed modes, quantization and the selected layer.
That regression plus eight affected eligibility cases passed after the fix.

The driver now fails immediately if graph opt-in installs no resident cores.
One corrective retry uses the identical bounded child command above, changing
only output directory to `layerkv_fp16_20260907_b8_nativegraph_02`. This is a
necessary installation-fix verification, not a repeat of a successful graph run.

### Corrective model probe completed

`b8_nativegraph_02` exited0. All1536 tokens and FP16 logprobs match the first
three historical scratch2k adaptive rounds exactly, with finite logprobs and
eight64-token responses per round. Each round passed KVC/expert/ownership
guards with zero stale entries, alignment violations and physical failures.
Layer0 retained49slots; shared physical pool stayed478150656bytes.

Layers1–39 installed replay; layer0 remained eager. Captures stayed39 after
round1, cumulative replays were2379/4836/7293, and warmup fallbacks stayed78.
Every round observed B8 for63 decode forwards. This closes the previously
missing actual runtime replay/correctness gate for this bounded shape only.

| Round | Graph tok/s | Historical same-round eager tok/s |
| --- | ---: | ---: |
| 1 (cold) | 68.906 | 62.701 |
| 2 | 128.866 | 106.359 |
| 3 | 127.111 | 104.848 |

Round3 is about+21.23% versus the historical same-round eager observation.
There is only one post-two-warmup round here, no independent repetitions and no
contemporaneous graph-enabled fixed arm. These are diagnostic observations,
NOT achievement of the20% dynamic-residency goal. The earlier failed-install
probe also provides a recent eager observation (round3:106.381tok/s), against
which this graph round is about+19.49%; launch variation matters.

Graph allocated workspace deltas sum1478656bytes. At final snapshot Torch
allocated69329836032bytes, reserved69654806528bytes, lifetime peak allocated
71119306752bytes and reserved71223476224bytes; device used71166459904bytes.
Initial device used72133246976bytes includes the not-yet-compacted selected
expert backing, so initial-to-final subtraction cannot isolate graph cost.
No matched eager device snapshot or sampled physical peak was collected.
Keep physical-budget acceptance unverified; do not equate1.41MiB allocated
workspace with total graph-private reserved memory.

Next meaningful comparison is graph-enabled fixed versus graph-enabled adaptive
with the same memory telemetry/budget and explicit warm timing boundary. Do not
rerun the successful shape compatibility tests to obtain that comparison.

## Graph-enabled fixed comparison

Added only the missing fixed child, reusing the completed adaptive artifact:
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_nativegraph_fixed_01/fixed.json`
versus `b8_nativegraph_02/lend.json`. The invocation is the same corrective
adaptive command above with output directory changed to the fixed directory and
`--arm fixed`. The driver translates the effective expert policy to fixed and
extra slots to0. A read-only engine-argument comparison before launch confirmed
these were the only two engine differences. Both use graph batch limit8.
The fixed child exited0; no compatibility test or adaptive model run repeated.

All1536 tokens/logprobs are exact between these two arms. Every guard passes,
stale/alignment/physical-failure counters are0, B8 has63 forwards each round,
both arms have39captures and7293 cumulative replays, and pool bytes remain
478150656. Initial and all three final memory snapshots match field-for-field,
including Torch allocation/reservation/lifetime peaks and device-used bytes.
This strengthens memory comparability for this pair but is not a sampled peak
physical-budget enforcement proof or broad workload acceptance.

| Round | Fixed16 tok/s | Adaptive49 tok/s | Gain |
| --- | ---: | ---: | ---: |
| 1 (cold) | 65.390 | 68.906 | 5.38% |
| 2 | 114.820 | 128.866 | 12.23% |
| 3 (after two warmups) | 111.355 | 127.111 | 14.15% |

Round3 mean TPOT58.857→54.936ms (-6.66%) and mean TTFT0.88967→0.56672s.
Thus14.15% batch-makespan throughput improvement is not14.15% decode-latency
improvement: both prefill/TTFT and decode contribute. Expert materializations
4086→2004, materialize CPU-observed time523.52→220.23ms, and main-stream wait
295.46→186.50ms in that round. Those timers can overlap; do not add them.

This pair still compares a KV-heavy fixed16 to adaptive49, not the best fixed
short-context split or a mixed-workload optimum. Only one post-two-warmup round,
no independent repetitions;20% dynamic-residency acceptance remains incomplete.

An actionable remaining control cost is already visible without another profile:
adaptive round3 makes8 donor scans, all misses, costing110.66ms. Ordinary-free
KV donation disables the existing negative cache because it lacks reliable
invalidation for ordinary-free ownership changes. Optimizing this requires a
correct availability-change signal (or an equally safe capacity proof), not
blindly turning the old cache on. Its measured standalone ceiling is only about
3% of this round, so this alone should not be advertised as enough for20%.
