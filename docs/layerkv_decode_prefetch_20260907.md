# LayerKV decode prefetch and CPU-known preparation

## Scope

This change has two bounded pieces:

- The shared expert chunk path may reuse the CPU routing snapshot already
  consumed by `_expert_chunked_core_required()` during decode. It still uses the
  GPU remap table, readiness wait, and post-load guard; other callers remain
  decode-ineligible for `cpu-known` preparation.
- Ordinary same-layer expert prefetch now caps a previous-route prediction by
  the overlap of the last two decode routes. The existing context/batch gate
  remains in force, and explicit cross-layer lookahead keeps its layer-budget
  limit.

The implementation is in `expert_hooks.py`, `shared_expert.py`, and
`scheduler_runtime.py`. Unit coverage includes the decode snapshot path and
route-confidence budget.

## Validation

The focused unit set passed with the existing venv:

```text
58 passed, 2 warnings
```

The model check used Qwen3.6-35B-A3B FP16 from the local HF snapshot, H100,
batch 8, input 128, output 64, 3 rounds, 2K scratch, selected-layer CPU
backing, and native MoE graph limit 8. The launcher recorded the resolved venv,
Python source, hub, and model paths in the plan artifact.

Both arms produced exact tokens/logprobs, passed guards, and matched the
physical pool. The summary remains diagnostic (`graph_budget_unverified=true`),
not a 20% acceptance result.

| Arm | Round | tok/s | prefetch issued | useful | wasted |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 3 | 53.345 | 0 | 0 | 0 |
| lend | 3 | 48.513 | 207 | 68 | 139 |

Compared with the immediately preceding same-configuration artifact, the new
budget reduced lend prefetch traffic in round 1 from 238 to 168 IDs and wasted
IDs from 141 to 96. In round 3 it reduced issued IDs from 226 to 207 and wasted
IDs from 151 to 139. It did not produce a throughput gain in this shape: lend
was 9.06% below fixed in round 3. The fixed/lend arms also use different
selected-layer expert capacities (16 versus 49), so this pair is mechanism
evidence, not a standalone residency-speedup claim.

Artifacts:

- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_prefetch_confidence_pair_01`
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_cpu_known_decode_graph_pair_01`
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_exact_chunk_prefetch_lend_02`

## Exact chunk overlap follow-up

The exact next-group path was then enabled for the `input` chunk order. It
protects the current group's slots and submits only the next group's IDs that
fit in the remaining capacity. In the 2-round lend-only check it submitted 58
exact IDs in 22 groups, with 128 ready-before-use hits and all guards passing.
The ready-miss stall time was 149.68 ms in round 1 and 165.58 ms in round 2,
versus 158.92 ms and 172.25 ms in the immediately preceding lend-only check.
The measured tok/s was 41.30 and 47.83, so this is an overlap/correctness
result, not a demonstrated throughput improvement; the run also uses the
diagnostic native-MoE graph mode.

Previous-route speculation should remain bounded because its useful fraction
is still low. With only one selected expert layer, the cross-layer counter
remains zero; a multi-layer experiment is needed before claiming cross-layer
expert lookahead. The next performance decision is whether the saved ready
stall survives a matched warm fixed/lend workload after accounting for the
different expert-kernel shapes, not whether to issue more speculative copies.

## Cross-layer trigger reachability fix

The previous zero-counter observation was specific to `shared_expert_layer=0`:
the cross-layer helper was only called from the active per-layer KVC-control
branch. Short-context runs have no offloaded KVC entries, so the KVC wrapper
returned before reaching the helper. The wrapper now invokes the same bounded
helper on the inactive path as well; it still requires decode mode, an existing
copy stream, a prepared expert plan, route/request compatibility, and the
configured context/batch gate. KVC preparation and access semantics are
unchanged.

The focused wrapper regression passes. A matched Qwen3.6-35B-A3B FP16 B32 run
with `shared_expert_layer=20`, 8192-token scratch, 16/24 fixed/lend capacity,
and one-layer lookahead shows the expected effect:

| Arm | Warm tok/s | Cross-layer issues | Ready-before-use | Warm ready-miss stall |
| --- | ---: | ---: | ---: | ---: |
| fixed | 163.451 | 29 | 424 | 583.10 ms |
| lend | 177.704 | 30 | 568 | 555.58 ms |

Outputs and all physical/KVC/expert/ownership guards are exact. The matched
warm lending gain is 8.72%; the earlier pre-fix layer20 pair was only 0.30%.
Because these are separate launches, this is strong mechanism evidence but not
a repeated statistical estimate. The artifact is
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_crosslayer_layer20_after_01`.

The next combined probe enables native replay for resident non-selected layers
in this same layer20 setup. Its graph workspace remains diagnostic until total
physical-memory accounting is closed; it must not be counted as a validated
20% result.
