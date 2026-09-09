# B8 steady decode profile: one capture, no new performance comparison

Artifact: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_profile_01`.
Only the adaptive child was run. Effective engine arguments, full log, responses,
runtime stats and compressed stage traces are saved. Same Qwen3.6 FP16/H100 GPU0
B8 input128/output64 workload, scratch2048, base16/max64, achieved49 slots.
Two warmup rounds followed by one profiled round. No fixed-arm rerun.

The concurrent driver now accepts explicit `--profile-round` and `--profile-steps`.
It uses existing engine stage profiling, CPU/GPU activities, no stacks or shape
recording, and marks all results diagnostic-only. Profiling makes performance
acceptance false even when outputs and physical guards match. Step counts must
leave a subsequent decode step for automatic flush. The driver tests passed49
cases; after adding the acceptance guard cases, only the8 profile-related cases
were rerun and passed. No repeated full LayerKV/GPU regression suite was run.

## Capture validation

- Selected round3,8 forwards per stage; scheduler produced separate EXTEND and
  DECODE files. EXTEND has1 CPU step, DECODE has exactly8 CPU `user_annotation`
  events named `step[DECODE bs=8]`.
- Do not count similarly named `gpu_user_annotation` events as additional steps.
- All1536 generated tokens and logprobs exactly match the first3 rounds of the
  existing `shortb8_scratch2k_01/lend.json` reference.
- No runtime policy, KV ownership or numerical tolerance was changed.
- Profiling and trace export increase wall time. Report no throughput gain from
  this run; use prior uninstrumented measurements for performance.

## Timeline observations

DECODE trace:
`lend.trace/1788773116.1302083-TP-0-DECODE.trace.json.gz`.
Window is first CPU decode annotation start through last CPU decode annotation
end. Clip GPU event intervals to that window and take their union (reuse
`union_us` from `scripts/layerkv_wait_trace_analyze.py`); do not add nested CPU
events or overlapping GPU categories.

| Quantity | Observed |
|---|---:|
| CPU decode annotations | 8 at B8 |
| First-start to last-end span | 694.890ms |
| Individual CPU step durations | 88.774,77.997,91.787,78.256,79.492,81.566,86.920,87.132ms |
| Observed GPU activity union in window | 96.621ms (13.9% of span) |
| GPU kernel union in window | 59.248ms |
| GPU memcpy union in window | 37.424ms |
| CPU CUDA runtime/driver call union in window | 76.454ms |
| GPU kernel event count in decode trace | 8945 |
| CPU operator event count | 44318 |

Kernel event totals include642 `fused_moe_kernel` calls,24.627ms summed device
duration. There are4026 `cuLaunchKernelEx`,2732 `cudaLaunchKernel` and1866
`cudaLaunchKernelExC` host events. These APIs may nest; do not sum them as disjoint
CPU cost.139 stream synchronization events sum to15.050ms host duration.

Named LayerKV CPU ranges (inclusive, overlapping, not additive): prepare93.810ms,
materialize49.161ms, H2D submit35.303ms, cache trim6.726ms, ready wait1.011ms.
The trace includes about1.44GB pinned H2D and0.77GB pinned D2H. Chunk-only counters
cannot describe total transfers once larger capacity avoids chunking.

The profile therefore redirects the next optimization toward host execution and
kernel submission, rather than assuming more scratch donation or PCIe bandwidth
alone can reach20%. The unobserved-GPU portion is NOT a recoverable-speedup bound:
it includes normal host work, dependencies, profiler overhead and inter-step gaps.

## Next action, not yet implemented

Inspect whether existing piecewise/CUDA-graph infrastructure can cover stable,
fully resident computation while leaving the selected offloaded expert layer,
KV reload/metadata ownership and admission outside replay. Shared-mode benchmarks
currently disable whole and piecewise graphs in both arms. Restoring safe replay
for unaffected computation may remove a large common overhead that hides the
residency tradeoff; it must be applied symmetrically to static and adaptive arms.
Do not enable whole-model graph replay around mutable expert capacity or bypass
LayerKV hooks. First establish the exact replay seam and ownership invariants,
then one focused test/prototype, not another capacity sweep.
