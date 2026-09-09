# Cross-request expert retention and admission comparison

`--layerkv-shared-expert-retain-across-requests` preserves expert loans at
prefill and request-finish boundaries. Request-owned KV cleanup remains active.
The default is disabled, preserving existing behavior. Retention requires an
enabled shared expert layer and free-KV donors.

`--layerkv-shared-expert-admission-policy recall` remains the default: an actual
admission shortage can remap borrowed physical backing to KV, then publish only
the common allocator capacity actually restored. `retain` is an explicit fixed
expert-heavy diagnostic baseline, requires fixed policy and cross-request
retention, and returns zero admission credit without recalling loans.

The concurrent driver exposes `--baseline-split expert-heavy`. Both arms have
identical initial and maximum expert capacity and preserve loans across request
boundaries. Fixed retains loans under admission pressure; adaptive can recall.
The old `kv-heavy` default still sets fixed extra capacity to zero.

This is not a preconditioned fixed-split acceptance benchmark: the fixed arm
grows one slot per decode step initially. Output-token-triggered late arrivals
are a mechanism diagnostic, not a matched wall-clock arrival process. Report
actual observed slots and batches rather than assuming the target was reached.

Retained-mode result guards require explicit retention/admission telemetry,
consistent nonzero loans and blocked KV, stable pointers, ownership correctness,
zero growth physical creates and the original KV/output/physical-budget checks.
Legacy runs still require zero outstanding loans. Intentional persistent loans
must not be silently treated as leaked pages or advertised as free KV capacity.

## Validation

Real CUDA tests cover request cleanup with a live location outside donor pages,
retained loan exclusion from KV credit, default automatic recall, prefill boundary
behavior and explicit admission restoration. CPU driver tests cover baseline
argument propagation and rejection of inconsistent loan/policy telemetry.
Full LayerKV suite: 359 passed, 19 warnings. The subsequent configuration
propagation assertion passes with all 15 residency-budget tests.

Targeted Ruff passes for the driver, tests, configuration, shared controller and
server arguments. Broader checks report pre-existing unused imports and missing
annotation names in runtime mixins; these were checked against HEAD and left
untouched. `git diff --check` passes. No full repository pre-commit run.

## Bounded long-context diagnostic

Artifacts: `/mnt/vdb/chengzhi/layerkv_retained_20260907_long_01`.
Qwen3.6-35B-A3B BF16, H100 NVL GPU0, TP1; two requests, initial input128,
late input6000 after16 output tokens, output64 each, one cold round, base16/max20,
KV16384/scratch8192/block2048. Exact arguments and child commands are in
`plan.json`. BF16 model evidence does not resolve the earlier full-model FP16
GDN dtype limitation. This run is not 20% throughput acceptance.

| Metric | Fixed retained | Adaptive |
| --- | ---: | ---: |
| Output tok/s (cold, includes prefill) | 10.3994 | 13.7285 |
| Actual decode histogram | B1:126 | B1:32, B2:47 |
| Long request TTFT seconds | 3.6353 | 0.8278 |
| Short request TPOT ms | 90.06 | 102.44 |
| Long request TPOT ms | 64.77 | 64.15 |
| Final expert slots | 20 | 16 |
| Admission recalls / restored common tokens | 0 / 0 | 1 / 2048 |
| Final physical pool bytes | 478150656 | 478150656 |

Both arms reached 20 slots. Fixed retains 24 MiB of loans; adaptive returns all
loans, with one recall measured at 12.63 ms. Adaptive has six measured cost-gate
rejections after calibration growth. Ownership, pointer, KV and physical-budget
guards pass. All 128 token IDs match, but seven logprobs in the short request
differ by more than 0.01 (zero-based positions17–22 and51), maximum0.1242578.
Long-request logprobs match within threshold. **valid=false**; the apparent
32.01% gain is not accepted. Batch enlargement is observed, but numerical
validation and matched warm/repeated performance remain open. A larger batch
also worsens the short request's TPOT in this diagnostic.

An independent `--arm native` follow-up with the same requests/output trigger
is saved as `native.json`. Native disables all LayerKV options and is an output
reference, not a matched-memory throughput baseline. Adaptive matches native
**exactly for all 128 tokens and logprobs**. Fixed matches native token IDs but
has the same short-request maximum logprob difference0.1242578; the long request
is exact. This isolates the discrepancy to fixed-serial versus native-concurrent
execution, but does not prove batching is the sole cause. A native serial
reference is still needed. Do not relax the original cross-arm tolerance or
rewrite `summary.json` as valid based on this follow-up.

The final guard additionally requires the fixed retained arm's final slot count
to reach the configured initial+extra target, so failure to grow cannot silently
masquerade as an expert-heavy baseline. This is checked against this artifact's
actual20 slots; measured steady-state retention across multiple workload phases
still requires a preconditioned experiment.

## Native serial isolation and fixed-time arrival support

`/mnt/vdb/chengzhi/layerkv_retained_20260907_native_serial_01/native.json`
replays the original requests with native SGLang and `max-running-requests=1`.
Both requests'64 tokens and logprobs match the fixed retained arm exactly.
Together with the native concurrent reference above, both LayerKV arms match
their corresponding native concurrency regimes; native serial versus concurrent
reproduces the same maximum0.1242578 discrepancy. This is evidence of a
concurrency-dependent native numerical difference, not a LayerKV-specific
corruption. The precise kernel/reduction source is not localized. Original
cross-arm `valid=false` is retained; no tolerance was relaxed.

The driver now supports `--late-after-seconds`, mutually exclusive with the
output-token trigger. Timed submission does not wait for initial output, uses a
round-relative timer, propagates early initial errors, cancels pending tasks on
failure and records `late_submission_lag_s`. Initial completion before the late
deadline is allowed. This is a scheduled arrival, not a guarantee of zero event
loop delay; actual submission timestamps must be inspected for comparability.

Two event-loop tests prove late submission even when the initial request cannot
produce any output until the late request starts, and failure propagation without
submitting the second wave. Full LayerKV suite:361 passed,19 warnings. Targeted
Black/Ruff pass. No runtime model change was needed for the native numerical
isolation; the diagnostic skill kept the original correctness threshold intact.

The bounded timed experiment uses the same model/memory/request settings as
above, changing only arrival to1 second and rounds to3. Artifact directory:
`/mnt/vdb/chengzhi/layerkv_retained_20260907_timed_01`; `plan.json` records exact
arguments and commands. Round1 remains cold and includes initial growth;
rounds2–3 observe carried state, not an independently preconditioned or
reversed-order acceptance study.

### Timed run stopped: retained-baseline liveness failure

The fixed child completed round1 at14.2991 tok/s (B2:63, final16 slots/no loans)
and round2 at14.1926 tok/s (B1:126, final20 slots/24 MiB loans). Round1 never
established the required expert-heavy split, so it cannot be its baseline.
Round2 ended with6144 reported common free KV tokens. In round3 the short request
finished, but the6000-token request remained queued with running=0 and no pending
KV eviction. Repeated no-prefill-progress logs from07:12:18 through07:13:14 UTC
confirmed lack of progress. The exact fixed child process group was terminated;
the parent exited1 and wrote `failure.json`. Adaptive was not launched by this
failed parent run. Completed artifacts and logs are preserved.

No speedup or correctness conclusion is drawn from this incomplete pair. The
next required diagnostic is whether retained loans interact with admission
reservation/allocator cleanup across multiple requests:6144 end-of-round common
tokens is insufficient evidence that the following6000-token admission can make
progress. Do not call a permanently stalled split a performance baseline, nor
silently enable recall on the fixed side and keep calling it fixed.

Actual timed submission deviations in the two completed rounds were+1.844 ms
and-0.0255 ms. The timer is subject to event-loop resolution; metadata records
the signed deviation rather than clipping it to zero. Tests also cover successful
initial completion before the timer and invalid/ambiguous CLI trigger arguments.

Final full validation after the additional boundary tests:366 passed,19 warnings.
