# LayerKV adaptive expert chunk order (2026-09-07)

## Scope

This change keeps the default expert token-chunk order as `input` and adds an
explicit `adaptive` option. It does not define a dataset-specific expert set.
The selected order is decided once per forward from the current request shape
and current shared-expert capacity:

- choose `reuse` when the actual token batch exceeds the resident slot
  capacity or when a multi-request batch's routed row demand (`batch_size *
  top_k`) reaches half of the resident slot capacity;
- choose an internal bounded `window-reuse` order for a multi-request
  long-context batch when the resident capacity is at most two top-k route
  widths (`capacity <= 2 * top_k`); this selects among at most 32 pending
  groups rather than scanning the full forward;
- choose `reuse` for a small batch with a long average prefix (currently
  `avg_prefix > 2048` and `batch <= capacity // 2`) when the low-capacity
  window branch above does not apply;
- otherwise keep `input` order.

The decision uses current runtime observations (`batch_size`, slot capacity,
the current layer's `top_k`, and average prefix length). It does not persist
expert identities or use a dataset-specific working set.

The command-line entry points are:

```text
scripts/layerkv_concurrent_perf.py --expert-chunk-order {input,reuse,adaptive}
scripts/layerkv_shared_perf.py --chunk-order {input,reuse,adaptive}
```

The local launcher resolves the venv and Hugging Face cache paths:

```bash
python scripts/layerkv_local.py paths
python scripts/layerkv_local.py prefetch-test
```

## Validation

Focused LayerKV tests passed after adding mock-compatible defaults:

```text
51 passed
```

The complete LayerKV unit suite passed with the venv's external build tools
available on `PATH`:

```text
526 passed, 19 warnings
```

The matched FP16 Qwen3.6-35B-A3B overflow-admission pair used the same
workload and compared `adaptive` fixed residency against `adaptive` lending:

| mode | long-request actual batch | output tok/s |
| --- | ---: | ---: |
| fixed | 2.667 | 20.2077 |
| lending | 4 | 20.8760 |

This is a `+3.31%` long-request throughput improvement while admitting four
requests instead of the fixed split's average 2.667. Request output IDs and
logprobs matched exactly, and KVC/admission guards remained valid. This result
is below the project target of `+20%`; it is evidence that adaptive chunk
ordering reduces transfer churn, not evidence that the target has been met.

In a separate, non-matched directional run, explicit `reuse` ordering reduced
long-run expert materializations from `15229` to `11893` and H2D busy time from
`1012.7 ms` to `675.5 ms`. Those counters are useful for mechanism diagnosis,
but should not be used as a controlled speedup claim because the run was not a
matched pair.

## B8 short-context policy correction

The first adaptive implementation compared request batch size directly with
the 49-slot capacity. Consequently, the target B8/top-k=4 workload selected
`input` on every forward (`token_chunk_adaptive_reuse_count=0`), even though
its routed row demand is 32. The policy now treats a multi-request routed
working set reaching half capacity as wide enough to amortize reuse ordering;
B8 selects `reuse`, while B2 remains on `input` for this shape.

Artifact: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_adaptive_route_demand_01`.
It uses the same effective engine arguments as
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_adaptive_order_gate_01` and changes
only this policy code. Outputs and logprobs match exactly across all 512-token
rounds. Round-3 lend-only throughput is 81.441 versus 50.419 tok/s
(+61.53%). This is a directional same-arm mechanism result; both runs use the
diagnostic native-MoE graph mode and it is not the fixed/lend 20% acceptance
result.

The corrected run still issues speculative route prefetch (210 IDs, 66 useful,
144 wasted in round 3). That traffic is now the next admission-control target;
the result does not justify increasing the prefetch window.

## Low-capacity long-context chunk-order gate

The long-context/small-batch lender workload exposed a separate CPU-side
cost: adaptive reuse ordering performed `17,879` group reorders and reached
`5.455 tok/s`, while the same lender workload with explicit `input` ordering
performed no reorders and reached `6.448 tok/s` (`+18.2%` directional). The
adaptive gate was first narrowed to the post-admission nine-slot state and
reached `6.365 tok/s` (`+16.7%` versus the earlier adaptive artifact).

The current gate extends that rule to the pre-admission boundary at
`capacity <= 2 * top_k`, while retaining `reuse` for larger capacities and
single-request long-context work. The low-capacity branch now uses a bounded
32-group lookahead (`window-reuse`) rather than the full O(groups^2) reuse
scan. This is a route-shape policy, not a fixed expert set. The explicit-input
and first-gate artifacts are diagnostic single-arm runs; they do not replace a
matched fixed/lend acceptance result.

Artifacts:

- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_layer20_long_input_only_01`
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_layer20_long_window_01`
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_layer20_long_window_adaptive_gate_02`

## Bounded route-reuse probe

The bounded lookahead was validated once with the same FP16 Qwen3.6-35B-A3B,
3 requests, 4900 input tokens, 64 output tokens, and E=9 lending settings as
the `lend_18_blockaligned` reference. The adaptive lender selected
`window-reuse` for 62 of 63 token-chunk batches, with a 32-group lookahead;
the first batch remained ordinary `reuse`.

Compared with the historical input-order lender, the candidate reduced expert
materializations from `6948` to `6806`, H2D batches from `2937` to `2936`, H2D
bytes from `43,731,910,656` to `42,838,523,904` (about `2.04%`), and ready-miss
stall from `83.2 ms` to `73.1 ms`. It preserved exact output IDs/logprobs,
physical-pool matching, and all guards. The fresh pair measured `9.406/10.929`
tok/s for fixed/lend (`+16.19%`); because the fixed arm is a single run and the
historical reference is not a same-process pair, this is not a 20% acceptance
claim. The candidate is retained as a transfer-volume reduction; the next
material target is overlapping or batching the remaining on-demand H2D work.

## Matched short-context B8 lending result

The first controlled B8 run used the same FP16 Qwen3.6-35B-A3B model,
`requests=8`, `input_tokens=128`, `output_tokens=64`, `max_running_requests=8`,
and no native-MoE graph. The fixed arm has 16 resident expert slots and no
extra slots; the lending arm starts at 16 and may borrow free KV pages for up
to 48 extra slots. Both arms reached actual decode batch 8. The lender grew to
49 peak slots (`grow_count=1`, 68 borrowed-slot uses), while fixed stayed at
16. Physical pools, guards, output token IDs, and logprobs matched exactly.

The benchmark now supports an explicit `--arm-order` so the child-process
startup/JIT order can be reversed. Warm aggregate throughput is computed over
rounds 2--3 from total output tokens divided by total client makespan:

| launch order | fixed warm tok/s | lend warm tok/s | lending gain |
| --- | ---: | ---: | ---: |
| fixed, lend | 50.862 | 85.971 | +69.03% |
| lend, fixed | 50.314 | 84.599 | +68.14% |

The pooled warm result is `+68.59%`. Round 1 is excluded because it is the
startup round: lending is `-8.80%` and `-5.12%` in the two launch orders while
the first process pays cold setup. This is the first matched short-context,
large-batch result above the provisional `+20%` target. It is still scoped to
one GPU, one selected expert layer, and this B8 shape; it does not establish a
general-model or mixed-workload gain.

Artifacts:

- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_adaptive_threshold05_kvheavy_matched_01`
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_adaptive_threshold05_kvheavy_reverse_01`

## Next measurement

The remaining research boundary is the opposite policy regime: long context
with a small request batch. The next useful comparison should keep the same
physical pool and output guards, use KVC-heavy pressure, and verify that the
planner retains KVC while recalling expert loans instead of repeatedly
materializing experts. Do not expand the benchmark matrix until that KVC-first
path has a matched timing artifact.
