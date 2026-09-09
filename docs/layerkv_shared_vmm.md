# Physical KV pages funding expert residency

`--layerkv-shared-expert-layer L` enables a bounded CUDA VMM path on
`perf/layerkv-allocation-fastpath`. It is off by default (`L=-1`).
`--layerkv-shared-expert-all-layers` selects the same physical page budget for
every discovered MoE layer; it is the intended LayerKV strategy for joint KVC
and expert residency. The all-layer manager keeps one controller per layer and
routes loans through the common `SharedVMM` arena.
It moves the **same physical allocation handles** from full-attention KV
storage into expert weight tensors. It does not merely decrement a KV counter
and allocate new CUDA memory for experts.

## Ownership and lifetime

1. A backend-neutral `buffer_factory` lets the hybrid MHA pool allocate its K/V
   buffers through `SharedVMM`; GDN state and default pool allocation stay native.
2. Each selected MoE layer gets CPU-backed compact expert slots. Initial backing
   is explicitly allocated; virtual space is reserved for all logical experts.
   `--layerkv-shared-expert-initial-slots` sets the initial residency per layer
   (default 16). All layers share the arena's physical page budget, while
   layer-local controllers preserve ownership boundaries during loans.
3. After decode, the controller selects only already-published free pages.
   KV backup completion and metadata publication belong to the normal KV
   lifecycle, not donor discovery (see the concurrent regression in
   [adaptive residency](layerkv_adaptive_residency_20260907.md)). By default the
   tokens must be backed and current. `--layerkv-shared-expert-free-kv-donors`
   also permits wholly unused reserved common-free pages without CPU backup.
   Padding, partial tail pages and already reused locations are excluded.
4. Donor locations are removed from KV allocator reuse. Under device-wide
   synchronization, the same handles are unmapped from KV and mapped into
   expert weight tails. All parameters of one extra expert must be funded
   together; failed transfers roll back. No `cuMemCreate` occurs during growth.
   Weight views grow without moving their base pointers. Existing expert
   materialization copies real weights from CPU into the borrowed slot.
5. Before a later prefill or request completion, borrowed expert weights are
   backed up/invalidated, handles return to their original KV addresses, and
   KV locations become available again. Returning pages does not restore their
   contents: KV recovery uses the CPU backing, never overwritten old contents.

For the single-layer mode, the maximum tail growth is
`--layerkv-shared-expert-extra-slots` (default 1), one fully funded slot per
decode step. In all-layer mode this value is the **global** extra-slot budget;
the manager assigns it in hotness order instead of multiplying it by the layer
count. Set it to 0 for the fixed-capacity control arm, which still uses VMM and
the same KV offload policy. Too few eligible pages leaves capacity unchanged.
Later prefill batches whose routed experts exceed capacity execute
in token groups whose combined expert set fits the budget. Each token's entire
top-k reduction stays inside one native MoE call, avoiding BF16 rounding from
splitting its experts across multiple reductions. Shared-mode ordinary CUDA
growth is explicitly rejected; initial capacity must cover one token's top-k.
This closes the existing fallback that otherwise expanded a later prefill to
135 slots independently of the KV budget. Recovery-task construction also now
imports the missing expert-demand type exposed by cross-request execution.
Token groups keep activation inputs separate, because the Triton runner may
overwrite its input with its output. The existing generic expert-chunk path
also now protects each chunk's input against this aliasing.

Tensor aliases retain virtual allocation ownership. Mapping changes synchronize
all device work; this is intentionally a correctness-first implementation.
CUDA VMM allocation/mapping semantics are described in the
[CUDA driver documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html).

## What the metrics mean

For opt-in per-function CPU attribution inside chunk preparation, see
[prepare profiling](layerkv_prepare_profile_20260906.md). The benchmark flags
are `--profile-chunks --profile-prepare`; normal execution keeps both disabled.

For the optional CUDA batch expert transfer backend and owning CPU backing
reuse, see [expert transfer results](layerkv_expert_transfer_20260906.md).
Benchmark flags are `--expert-transfer-backend cuda-batch` and
`--expert-batch-backing-layout individual`. Defaults retain the torch path;
initial allocation savings and later-request regressions are reported separately.

Demand D2H can optionally use GPU stream ordering instead of an early host
barrier with `--expert-demand-d2h-wait stream` (CUDA batch only). See
[unprofiled demand-wait comparison](layerkv_demand_wait_20260906.md); its small
incremental timing change does not justify changing defaults.

`shared_vmm` in the validated LayerKV summary reports:

| Field | Meaning |
| --- | --- |
| `physical_bytes`, `physical_create_count` | Live handles and cumulative initial allocations in the shared allocator, not total GPU usage |
| `kv_to_expert_pages`, `expert_to_kv_pages` | Actual page ownership transfers and returns |
| `loan_bytes`, `peak_loan_bytes` | Current and peak physical KV bytes held by experts |
| `growth_physical_create_count` | New physical pages allocated while funding growth; must be zero |
| `lend_wait_ms`, `recall_wait_ms` | Host wall time in the VMM transfer/return's existing device synchronization |
| `lend_remap_ms`, `recall_remap_ms` | Host wall time in mapping changes after that synchronization; no extra timing fences |
| `current_expert_slots`, `peak_expert_slots` | Actual compact weight tensor capacity |
| `borrowed_slot_use_count` | MoE calls routing to a borrowed slot, not merely reserving one |
| `token_chunk_batches`, `token_chunk_calls` | Batches and native calls executed within the fixed expert budget |
| `blocked_kv_tokens` | Locations excluded from KV allocator reuse while pages are lent |
| `ownership_guard_pass`, `expert_pointers_stable` | One owner per live handle; original expert base addresses unchanged |

Ordinary `physical_kvc_reclaim_peak_mb` still measures reusable KV capacity;
it is **not** the amount transferred to experts. CUDA VMM backing also is not
accounted by PyTorch's caching allocator; use the shared ownership metrics.
The baseline expert cache reduction is separate from incremental KV-funded
growth: this test starts at 16 of 256 experts, not 256 growing to 257.

## Reproduce using the existing environment

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q

cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_hybrid_validation.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_shared_vmm_new_run \
  --gpu 0 --dtype bfloat16 --reclaim-mb 40 \
  --input-tokens 4096 --output-tokens 12 --batch-size 1 --rounds 2 \
  --context-length 8192 --max-total-tokens 16384 \
  --scratch-tokens 4095 --block-tokens 2048 \
  --shared-expert-layer 0 --expert-initial-slots 16 --expert-extra-slots 1 \
  --timeout-s 600
```

`PATH` is for the external JIT toolchain's Ninja. All experiment configuration
uses CLI arguments. No package installation or custom extension build is needed
with this environment's existing `cuda.bindings.driver`. The output directory
must be new; commands, effective server arguments, responses and stats are saved.
The driver uses `expert-first`: in this branch that means preserving experts
and reclaiming KV. `kv-first` preserves KV in joint mode and cannot fund this test.

## Scope

This opt-in path requires CUDA, hybrid FP16/BF16 MHA storage, page size 1,
TP=PP=1, Triton attention, `kvc-expert`, `expert-first`, `per-layer-arena`,
disabled radix/overlap/chunked-prefill/graphs, and no memory-saver/speculation.
It uses Triton MoE without A2A, positive fixed reclaim pressure, and no separate
dynamic-pressure or expert-budget/backing override.
Expert parameter rows must align to the driver's allocation granularity.
On the tested H100 NVL that granularity is 2 MiB; a Qwen expert needs 4 MiB
for W13 and 2 MiB for W2. One transferred expert therefore needs three pages.

The full Qwen run uses **BF16**, because the local FP16 model execution has an
independent GDN convolution dtype mismatch. CUDA transfer/GEMM/restore tests
cover both FP16 and BF16. This is a single-layer synchronous correctness path,
and the all-layer path now validates multi-layer installation and routing, but
it is not yet a validated multi-layer adaptive residency policy or throughput
improvement. The initial all-layer mode uses equal per-layer slot defaults;
the global extra-slot budget is now assigned by hotness while pressure gates
protect KV headroom. A fuller transfer-cost/KV-benefit optimizer is still a
later policy layer. Prefix sharing, retraction, distributed serving and
asynchronous page transfers are not established by these experiments.

## Pressure-first validation

Do not interpret a short-context run with unused KV capacity as a SharedVMM
benefit. Use a workload whose live context plus scheduler headroom exceeds the
per-layer arena capacity, and inspect `shared_vmm.last_global_plan` together
with `kvc_*_shortage_tokens`, `kvc_evict_count_total`, `kv_to_expert_pages`,
`expert_to_kv_pages` and `loan_bytes`. Under context or admission pressure the
all-layer manager recalls outstanding expert loans and blocks new growth; when
pressure is absent it spends the global extra-slot budget in hotness order.
The validation driver exposes `--shared-expert-free-kv-donors` for the
short-context arm, where completely unused KV pages may be lent without first
requiring a KVC eviction.

## Verified single-request result (2026-09-06)

`/mnt/vdb/chengzhi/layerkv_shared_vmm_20260906_02/summary.json` passed on H100 NVL
with the above command except `--rounds 1`:

- Three physical pages (6 MiB) moved KV -> expert -> KV; capacity 16 -> 17 -> 16.
- The borrowed slot participated in five MoE calls. Growth created zero new
  physical pages; handle ownership and expert pointer guards passed.
- All 12 generated token IDs and output logprobs matched the independent baseline
  exactly. KV reclaim peak was 40 MiB, distinct from 6 MiB transferred to experts.
- KV/expert guards passed, stale/alignment counts were zero, and final CPU KV
  backing usage, outstanding loans and blocked KV locations were zero.

Earlier numbered failures are preserved diagnostics, not passing runs.

## Verified cross-request result (2026-09-06)

`/mnt/vdb/chengzhi/layerkv_shared_vmm_20260906_06/summary.json` passed with the
two-round command above on the final implementation:

- Both independent processes exited successfully. All 24 output token IDs and
  output logprobs matched exactly (`max_output_logprob_delta=0`).
- Actual transfer remained three 2 MiB pages, with three returned pages, peak
  residency 17 slots and final residency 16. The borrowed slot ran five times;
  no physical pages were created during growth, and all ownership guards passed.
- The second prefill used 63 token-group MoE calls within the 16-slot budget.
  There was one funded growth event in this run, not one per request: a later
  eviction does not by itself guarantee another complete eligible page loan.
- Total KV evictions/reloads were 40,960/4,096 layer-tokens. There were 738 expert
  materializations, including fixed-capacity prefill; these are not 738 new slots.
- Final host KV usage, outstanding loans and blocked locations were zero;
  both expert/KV guards and the arena accounting checks passed.
- All 37 LayerKV unit tests passed, including FP16/BF16 real physical handle
  identity/GEMM/restore tests, mapping rollback, allocator exclusion, fixed
  capacity checks, scheduler recovery and in-place/chunk reduction regressions.
  Import smoke, changed-file compilation, new-file Black/isort/Ruff checks and
  `git diff --check` passed. A full-repository lint run is not claimed: existing
  unrelated F821 issues remain in `planner.py` and `expert_backing.py`.

Runs 03--05 are intentionally retained failure evidence. They exposed ordinary
expert growth bypass, the missing recovery-demand import, in-place activation
aliasing and BF16 expert-chunk reduction drift. No tolerance was relaxed to make
run 06 pass; the output-logprob threshold remains 0.01.

## Fixed-capacity performance control

`scripts/layerkv_shared_perf.py` dry-plans by default; `--execute` runs two
isolated processes with identical settings except extra slots (0 versus 1).
Both keep KV pressure and compact initial residency, unlike the full-resident
baseline used by the correctness driver. It saves effective engine arguments,
per-request cumulative-stat differences, stream arrival times, outputs and guards.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_shared_perf_new_run \
  --gpu 0 --dtype bfloat16 --rounds 2 \
  --input-tokens 4096 --output-tokens 64 --skip-tokenizer-init --execute
```

TTFT and TPOT use client stream arrivals; TPOT is `(last-first)/(output_tokens-1)`
and requires the first event to contain exactly one token. Scheduler prefill
duration uses `prefill_finished_time-forward_entry_time`; it is not pure kernel
time. The first request (including lazy setup/JIT) is reported separately, and
later requests are not automatically called steady state. Examine per-request
page loans and actual borrowed-slot uses before attributing a timing change to
extra residency. Mapping timers exclude donor selection, CPU backing and expert
metadata updates; they are not total controller overhead.

The first bounded comparison and its admission limitation are recorded in
[the performance report](layerkv_shared_perf_20260906.md). The two-round default
does not establish sustained serving. The subsequent
[admission repair and six-request replay](layerkv_admission_20260906.md) covers
arena reuse after native capacity is exhausted, including renewed expert loans.
Tail-window analysis defaults to output
tokens 17--64 (`--decode-tail-start 16`) and can be recomputed from saved stream
events with `--summarize-only`; execution arguments must match the recorded plan.

### Donor miss caching

The shared expert controller now caches insufficient-donor results until KV
availability or the expert capacity requirement changes. It still finalizes
pending offloads before checking the cache and revalidates actual donor ownership
before every loan. It does not retain positive donor locations or handles.
Use `--disable-donor-cache` in the performance driver (server equivalent:
`--layerkv-shared-expert-disable-donor-cache`) for an uncached ablation.
`donor_scan_count`, `donor_scan_ms`, `donor_miss_count`, `donor_cache_hit_count`
and `donor_finalize_ms` are cumulative shared-VMM counters; use per-request
deltas for comparisons. See the [donor cache report](layerkv_donor_cache_20260906.md)
for the measured baseline, invalidation rules and matched replay.

### Prefill token-group reuse order

The performance driver accepts `--chunk-order input|reuse` (default `input`).
`reuse` orders the existing groups by current resident expert overlap without
splitting a token's top-k or changing output row placement. Server equivalent:
`--layerkv-shared-expert-chunk-order reuse`. This remains opt-in: fewer prefill
loads need not imply faster TTFT or a better cache state for decode.

`--profile-chunks` (server: `--layerkv-shared-expert-profile-chunks`) separates
CPU routing/grouping, gather, expert preparation, native MoE and scatter phases.
CUDA stream intervals include host submission gaps and waits, not just kernels.
See [the chunk-order report](layerkv_chunk_order_20260906.md) for timer boundaries,
correctness checks and the controlled comparison.

### CPU-known prefill demand

The driver accepts `--prepare-path generic|cpu-known` (default `generic`), with
server equivalent `--layerkv-shared-expert-prepare-path`. The opt-in path reuses
the existing CPU group expert set for demand discovery, while preserving
materialization order, hotness behavior, ready waits and GPU remap validation.
See [the CPU-known prepare report](layerkv_cpu_known_prepare_20260906.md) for
eligibility/fallback boundaries and matched unprofiled comparisons.
