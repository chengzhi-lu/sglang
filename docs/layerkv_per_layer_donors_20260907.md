# Per-layer free donors and incremental KV-capacity selection

The opt-in `--layerkv-shared-expert-free-kv-donors` path now discovers donors
from each layer's own free ledger. A page does not need matching free addresses
in every other layer. Ordinary free locations and published overwrite bitsets
are intersected with reserved ownership and exclude allocated, protected and
pending-cleanup locations. Partial pages and padding remain ineligible. No new
eviction completion is published by donor discovery.

Strict per-layer claims still remove addresses before VMM unmaps a K or V page.
Claimed token locations remain unavailable for both K and V in that layer until
recall. This intentionally does not introduce channel-specific allocator credit.
Recall remaps backing before returning free locations. Global scheduler credit
remains the intersection of reusable addresses, not a sum over layers.

The selector greedily minimizes **incremental lost common KV tokens** for each
required physical page. It prefers pages whose locations already cost no common
credit, including K/V pairs at the same locations. Stable discovery order breaks
ties. This is a bounded greedy selection, not a global optimal-packing claim.
`donor_select_ms` records selection overhead separately from `donor_scan_ms`.

## Regression evidence

- FP16/BF16 real VMM tests first failed when a second layer could not lend after
  common locations were consumed. The extended test now grows two extra expert
  slots using six physical pages across four KV layers at the same locations.
  Common capacity is charged once and restored once, with no physical creates.
- A larger single-layer pool first lost 6144 common tokens for three pages. The
  K/V pairing regression now loses only 4096, restoring all credit on recall.
- Existing active/protected/unreserved/cleanup exclusion and rollback tests pass.
- Full LayerKV suite: **329 passed, 19 warnings** before adding the selection-only
  timing counter. Targeted Ruff and `git diff --check` pass.

## Bounded model experiment

Dry plan inspected before launch:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  ../scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_adaptive_20260907_b8_slots24_01 \
  --requests 8 --max-running-requests 8 --rounds 2 \
  --input-tokens 128 --output-tokens 32 \
  --expert-extra-slots 8 --expert-policy adaptive --execute
```

BF16 Qwen3.6-35B-A3B/H100 NVL/TP1; both arms start with 16 backed slots and
identical KV allocation. The adaptive arm may reach 24 by transferring physical
pages, not allocating new backing. This tests short-context capacity scaling;
it does not by itself establish greater batch capacity under long-context
admission pressure or complete the research goal.

### Result and next bottleneck

The pair completes with `valid=true`, all 512 output tokens/logprobs exact across
arms and matching physical pools (478150656 bytes). Both rounds run 31 actual B8
decode forwards. Round 1 grows no expert slots in either arm; round 2 adaptive
reaches 24 slots via eight growth episodes with zero growth physical creates.

| Round | Fixed tok/s | Adaptive tok/s | Interpretation |
| --- | ---: | ---: | --- |
| 1, cold | 33.4284 | 42.5217 | +27.20%, but no loans or materialization reduction; not evidence of residency benefit |
| 2 | 89.1627 | 71.4329 | -19.88%, despite successfully growing to 24 slots |

Round-2 TPOT is approximately 69.40 ms fixed and 93.50 ms adaptive. Materializations
fall only from 2107 to 2071 (36 fewer), and measured H2D busy time from 272.33 to
267.83 ms (4.50 ms saved). Eight donor scans take 169.73 ms and selection another
70.70 ms. This supports batching growth to the selected target instead of rescanning
and introducing another kernel shape for each intermediate slot, and requiring
benefit to pay for actual control/transfer costs. It does not establish that either
change alone will reach 20%. Cold/JIT effects and fixed-first ordering still require
controlled repeats. No independent native B8 comparison was run in this turn.
