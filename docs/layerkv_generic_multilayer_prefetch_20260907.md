# Generic multi-layer expert residency and prefetch

## Purpose

The single-layer SharedVMM path is useful for physical KV-page lending, but it
cannot demonstrate cross-layer expert lookahead because it installs only one
`shared_expert_layer`. The existing generic LayerKV planner can install compact
expert slots on multiple discovered MoE layers when
`layerkv_shared_expert_layer=-1` and `layerkv_mode=kvc-expert`.

This path does not specify a dataset-specific expert ID set. Runtime hotness
chooses initial resident IDs, while the planner uses the current reclaim target,
layer shape, observed route cost, actual batch and context metadata. The model
driver now exposes this explicitly as `--expert-layer-mode generic`; the default
`shared` mode is unchanged.

## Validation

The local CPU validation script was repaired to use the current ForwardBatch
metadata contract and is available through the path launcher:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_local.py expert-validation -- \
  --output-dir /tmp/layerkv_generic_cpu_probe \
  --reclaim-limit-mb 0.0002
```

Result: 7/7 policy cases valid, including the generic two-layer planner,
physical slot rebinding, materialization, top-k rewrite and unsupported-quant
guard.

The Qwen3.6-35B-A3B FP16 model check used the resolved local venv/HF paths and
did not provide any expert IDs:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_local.py hybrid-validation -- \
  --output-dir /mnt/vdb/chengzhi/layerkv_generic_fp16_multilayer_20260907_01 \
  --gpu 0 --dtype float16 --reclaim-mb 64 \
  --input-tokens 128 --output-tokens 16 --batch-size 4 --rounds 1 \
  --context-length 8192 --max-total-tokens 16384 \
  --scratch-tokens 2048 --block-tokens 2048 \
  --shared-expert-layer -1 --expert-layer-mode generic \
  --expert-policy ratio-50-50 --expert-cpu-backing-mode none \
  --expert-install-layers-per-step 16 --expert-install-budget-mb 256 \
  --expert-prefetch-lookahead-layers 2 --timeout-s 600
```

The run produced exact tokens/logprobs and passed KVC/expert guards. The
generic planner discovered 40 MoE layers and installed six of them, reclaiming
36 MiB physically against a 32 MiB expert target. KVC did not physically evict
in this short-context check, so cross-layer prefetch counters remained zero;
this is correctness/installation evidence only. A separate long-context,
KV-pressure workload is required to measure overlap between later KVC reloads
and expert H2D copies.

Artifact: `/mnt/vdb/chengzhi/layerkv_generic_fp16_multilayer_20260907_01`.

## Long-context policy gate

A separate Qwen3.6 FP16 check used `input_tokens=3072`, batch 2,
`max_total_tokens=8192`, `scratch_tokens=2048`, and the same generic planner.
It physically exercised KVC pressure: 16,320 KVC evictions, 6,528 reloads and
31.875 MiB peak physical KVC reclaim, with exact output/logprobs and zero KVC
guard errors. The expert planner still installed six layers and reclaimed
36 MiB.

The expert cross-layer path was intentionally skipped 23 times with
`kv-priority-long-context-small-batch`; no speculative expert H2D was issued.
This is the intended policy outcome for this workload, not a missing candidate.
The artifact was generated before the driver relaxed its overly strict
requirement that every generic run must also materialize an evicted expert;
physical expert reclaim and multi-layer installation were already present.

Artifact: `/mnt/vdb/chengzhi/layerkv_generic_fp16_longctx_prefetch_20260907_01`.

To isolate the physical next-layer KVC path from virtual scratch, a second
check used `input_tokens=4096`, batch 2, `max_total_tokens=8192`,
`scratch_tokens=0`, and `block_tokens=16`:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_local.py hybrid-validation -- \
  --output-dir /mnt/vdb/chengzhi/layerkv_generic_fp16_physical_prefetch_20260907_01 \
  --gpu 0 --dtype float16 --reclaim-mb 64 \
  --input-tokens 4096 --output-tokens 8 --batch-size 2 --rounds 1 \
  --context-length 8192 --max-total-tokens 8192 \
  --scratch-tokens 0 --block-tokens 16 \
  --shared-expert-layer -1 --expert-layer-mode generic \
  --expert-policy ratio-50-50 --expert-cpu-backing-mode none \
  --expert-install-layers-per-step 16 --expert-install-budget-mb 256 \
  --expert-prefetch-lookahead-layers 2 --timeout-s 600
```

This run passed the exact-output/logprob and residency guards. The physical
per-layer KVC path issued 30 next-layer prefetches and reached 35
ready-before-use events with zero KVC failures; virtual KVC counters stayed at
zero. The generic expert path still installed six of 40 discovered layers and
reclaimed 36 MiB. Its cross-layer expert prefetch remained disabled by the
long-context/small-batch policy gate, so no expert H2D was issued speculatively
while KVC was under pressure.

Artifact: `/mnt/vdb/chengzhi/layerkv_generic_fp16_physical_prefetch_20260907_01`.
