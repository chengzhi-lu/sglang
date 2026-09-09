# Hybrid full-attention KVC offload

The `perf/layerkv-allocation-fastpath` implementation supports the MHA storage
inside `HybridLinearKVPool`. A storage view preserves model layer IDs (for
Qwen3.6-35B-A3B: 3, 7, ..., 39), while host backing maps these IDs to compact
buffer offsets. Hooks stay on the outer hybrid pool. Metadata rewrites target
the full-attention backend, not the linear-attention backend.

The existing per-layer arena performs real device-to-host backup and
host-to-device recovery/materialization. Evicted slots may be overwritten and
reused; recovery is not dependent on their old contents. K/V buffer pointers
remain stable. GDN recurrent/conv state remains resident in its native pool;
LayerKV request completion also releases the request's GDN slots.

Native extension slots transferred to LayerKV have one allocator owner. Cleanup
tracks all full-attention layers, including layers not selected for eviction.
New extensions reset cloned per-layer request mappings. CPU backing retained
across reloads remains tracked until request completion.
The prefill hook preserves newly registered live allocations. Triton index
rewrites use the backend's packed KV lengths, which are prefix-only for normal
extend attention, rather than writing the full sequence into a smaller buffer.
Late pruning of released-generation tombstones removes metadata only; it must
not release the same GPU/host slots a second time.
Completion also invalidates scratch/materialization caches for affected layers:
reusing a host slot number in a new request does not preserve its old KV contents.

## Scope and limits

- The controlled validation uses TP=1, Triton attention, `per-layer-arena`,
  async-deadline scheduling, and disabled prefix caching, overlap scheduling,
  and CUDA graphs. Prefix sharing, retraction, PP/TP>1 and other attention
  backends are not established by this validation.
- MLA and FP4 storage are not supported by this adapter.
- Reclaim MB means reusable KV slots **inside the existing pool**, not CUDA
  allocations returned to PyTorch or memory made available to expert weights.
  The new opt-in [shared VMM path](layerkv_shared_vmm.md) additionally transfers
  physical KV pages to expert weights; ordinary per-layer-arena behavior is
  unchanged. Its actual transferred bytes are reported separately.
- CUDA copy tests cover FP16 and BF16. End-to-end Qwen validation currently uses
  BF16: the local FP16 model run encounters a GDN convolution dtype mismatch
  independently of LayerKV. This change does not resolve that model issue.
- This is a correctness test, not a throughput or residency-tradeoff benchmark.

## Reproduce

Reuse the existing environment; no reinstall of torch or kernel packages:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q

cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_hybrid_validation.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_hybrid_new_run \
  --gpu 0 --dtype bfloat16 --reclaim-mb 0.5
```

`PATH` is only for the external JIT toolchain to locate the environment's Ninja.
All experiment options are explicit arguments. The output directory must be new.

The driver launches independent baseline/offload processes. Each performs two
rounds of two requests (128/160 input tokens, 20 output tokens, greedy sampling).
It records launch commands, effective server arguments, responses/logprobs,
server logs and a final validated LayerKV summary. Passing requires identical
80 output token IDs, finite logprob differences within 0.01, nonzero evict/reload
and physical reclaim, a passing KVC guard, and zero stale/alignment/failure counts.
It also requires final host usage of zero and arena free counts within capacity.

## Verified result (2026-09-06)

On one H100 NVL with the snapshot and BF16 settings above:

- `test/registered/unit/layerkv/`: 26 tests passed, including real CUDA FP16/BF16
  overwrite/reload checks; `scripts/layerkv_smoke.py` also passed.
- `/mnt/vdb/chengzhi/layerkv_hybrid_20260906_07/summary.json`: `valid=true`.
- All 80 generated token IDs matched; maximum output-logprob difference was 0.
- 320 layer-token evictions (all async), 64 layer-token reloads, 20 scratch
  materializations; measured peak reusable KVC was 0.3125 MiB for a 0.5 MiB target.
  Reload tokens and scratch materialization calls are different counters.
- KVC guard passed; stale entries, alignment violations and final host used
  tokens were zero. Final arena capacity/min-free/common-free were all 1600.

Earlier numbered run directories are preserved diagnostics, not passing results.
They exposed GDN request-slot leakage, allocator ownership/repeated cleanup,
prefill mapping/index bounds, retained CPU backing, and stale scratch-cache reuse.
No end-to-end FP16, throughput improvement, or KVC-to-expert memory transfer is
claimed by this result.
