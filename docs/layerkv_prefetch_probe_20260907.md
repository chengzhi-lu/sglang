# One-group expert H2D lookahead feasibility

## Scope

Read-only feasibility measurement after batched remap publication. No prefetch
is executed and no slot, residency, LRU, backing cache, group order, KV budget,
copy dependency or final GPU guard is changed. Reuse the existing opt-in
`--trace-waits-request N` driver / `--layerkv-shared-expert-trace-waits` server
diagnostic rather than adding an ordinary environment-variable switch.

When tracing is enabled, `SharedExpertController` records the latest prefill's
group metadata under `shared_vmm.prefetch_probe`. The snapshot occurs after
current-group preparation and before its MoE call, using CPU metadata already
available to grouping/materialization. It makes no CUDA calls and retains no
weight/storage references. Each new prefill replaces the prior snapshots; a
batch ID permits freshness checks. Default execution collects no snapshots.
Only input-order grouping is supported by this probe; reuse-order traces remain
usable for other analysis, but this analyzer rejects them as unsupported.

For current group C and next group N:

1. Protect all slots used by C for its entire MoE call. Do not assume an expert
   is finished earlier inside a fused kernel.
2. Also protect already-resident experts required by N. Evicting them to load
   another member of N would not provide net missing-expert coverage.
3. Remaining slots give a structural capacity bound. For a no-extra-D2H bound,
   require either a free slot or published CPU backing for its victim, and
   published CPU backing for the incoming expert.
4. The bound is the smaller of available destinations and missing sources.
   Terminal groups have no next-group opportunity; their missing-demand fields
   are absent, not measured zero demand for a nonexistent next group.

These are **optimistic structural bounds, not DMA/lifetime admission**. The
probe does not inspect/retire pending transfer owners, run a slot reservation,
or establish cross-stream readiness. Positive bounds require further safety
work before implementation. A zero capacity bound does rule out this specific
whole-group lookahead without extra slots, regardless of lifetime bookkeeping.
Backing presence follows the runtime's published local-backing contract; it is
not a new pinned-memory or owner validation shortcut.

## Timing and validation

`scripts/layerkv_prefetch_probe_analyze.py` joins the snapshots with actual
`layerkv/chunk/moe` kernel activities by their CPU launch correlation, not by
whether device execution falls inside a CPU range. It reuses the wait analyzer
to validate batch H2D visibility, counts, bytes and prefill guard coverage. It
also requires exactly one trace per arm, fresh batch metadata, full input-token
coverage, matching group counts, and equality between the probe's next missing
bytes and the next group's observed H2D bytes. Missing/ambiguous kernel records
are rejected, never silently converted to zero.

For each current group it reports GPU kernel interval union and first-to-last
kernel span separately. A positive structural candidate gates a deliberately
loose overlap ceiling: `min(current MoE device span, next group's H2D activity)`.
This ignores candidate byte fraction, submission latency, contention and all
ownership dependencies, and includes launch gaps in the span. It is not an
achievable speedup prediction. No GPU events/waits are added by the snapshot
probe. Kineto itself perturbs CPU submission; traced E2E is not a benchmark.

The current analyzer targets the controlled request whose prefill groups all
have H2D misses, using the existing wait analyzer's fail-closed scope. It is not
a general all-hit, multi-worker, multi-layer or arbitrary-dtype throughput tool.

## Reproduction

H100 NVL GPU 0, Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16, TP=1, eager, batch=1. Retain
the validated model/GDN setup; FP16 unit coverage is not a full-model FP16 run.
One fixed16/grow17 pair, each with six different 4096-input/64-output greedy
requests; trace only request 6. The first five requests restore the matched
residency history. No concurrent GPU tests or other model runs.

From `/home/chengzhi/github/sglang-perf-layerkv`:

```bash
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_prefetch_probe_fresh \
  --gpu 0 --dtype bfloat16 --rounds 6 --input-tokens 4096 --output-tokens 64 \
  --chunk-order input --expert-transfer-backend cuda-batch \
  --expert-batch-backing-layout individual --expert-demand-d2h-wait stream \
  --prepare-path cpu-known --expert-backing-release-validation admission \
  --expert-host-extra-budget-mb 256 --expert-backing-cache-mb 128 \
  --expert-backing-cache-accounting scan --expert-remap-update batch \
  --trace-waits-request 6 --execute

/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python scripts/layerkv_prefetch_probe_analyze.py \
  --run-dir /mnt/vdb/chengzhi/layerkv_prefetch_probe_20260907_batch \
  --output /mnt/vdb/chengzhi/layerkv_prefetch_probe_fresh_analysis.json
```

Omit `--execute` for the dry plan and use fresh result filenames/directories.
`PATH` only exposes external framework executables from the reused virtualenv;
ordinary configuration is explicit CLI metadata. Unchanged defaults: context
8192, KV tokens 16384, scratch 4095, block 2048, reclaim target 40 MiB, shared
expert layer 0, initial slots 16, extra slots 0/1, CPU-copy cache/pool caps each
128 MiB. No packages or CUDA extensions are installed/compiled for this probe.

## Regression coverage

The full LayerKV suite passes **223 tests** (19 warnings), including ten new
capacity/integration cases and nine analyzer cases. Tests cover full occupancy,
protection of next-resident experts, missing source/victim backing, free slots,
terminal groups, invalid residency maps, real FP16/BF16 chunk execution with
identical traced/untraced outputs, mappings, LRU and transfers, delayed GPU
execution relative to CPU ranges, and missing/ambiguous trace data. Run from
the worktree's `python/` directory:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Targeted Black/isort/F821 and `git diff --check` pass. Full-repository pre-commit
is not run over the large existing dirty worktree. No commit or unrelated
cleanup is performed.

## Measured result and decision

Artifacts are in `/mnt/vdb/chengzhi/layerkv_prefetch_probe_20260907_batch/`:
`plan.json`, effective engine args, logs, raw per-request snapshots, both
`*.trace/*.trace.json.gz` files, `summary.json`, and the final `analysis.json`.
Both runtime summaries and the fail-closed trace analyzer pass. This is one
diagnostic family with two arms, not an unprofiled performance comparison.

The metadata/trace join for request 6 gives the same structural result in both
arms. **All 55 prefill groups have capacity 16, including grow17.** The arm name
is not evidence of an extra prefill slot: loans are recalled before prefill,
and the snapshots confirm the actual capacity. The full run still exercises
grow17's six 2 MiB KV page lends/returns and ends with no outstanding loan.

| Request 6 measurement | fixed16 | grow17 |
| --- | ---: | ---: |
| Groups / next-group transitions | 55 / 54 | 55 / 54 |
| Groups using all 16 slots | 50 | 50 |
| Transitions with a positive backed capacity bound | 3 | 3 |
| Next missing experts (excluding first group's loads) | 521 | 521 |
| Maximum candidate experts / bytes | 4 / 24 MiB | 4 / 24 MiB |
| Missing-expert coverage upper bound | 0.768% | 0.768% |
| All prefill expert H2D device activity | 58.755 ms | 58.757 ms |
| All prefill MoE kernel interval union | 3.700 ms | 3.704 ms |
| All prefill MoE first-to-last kernel spans | 17.905 ms | 18.046 ms |
| Candidate-gated loose overlap ceiling | 1.080 ms | 1.099 ms |

Only groups 2, 53 and 54 have a positive candidate bound. Group 55 is terminal,
so its unused slots cannot help a next group. One other non-full group has no
net candidate after preserving next-resident experts. Candidate details below
use fixed16; grow17 has the same group/token/expert counts and nearly identical
device timings:

| Current group | Tokens | Active experts | Next missing | Candidate bound | Current MoE kernel union | Current device span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 15 | 11 | 1 | 0.0445 ms | 0.3721 ms |
| 53 | 41 | 14 | 9 | 2 | 0.0606 ms | 0.2683 ms |
| 54 | 10 | 15 | 4 | 1 | 0.0590 ms | 0.4818 ms |

The three candidates' actual kernel work totals only about 0.164 ms per arm.
The approximately 1.1 ms window ceiling is intentionally much looser: it
includes CPU submission gaps between kernels and ignores that only 4 of the
521 missing experts have candidate slots. It is neither pure compute time nor
a predicted E2E saving. DMA ownership validation could further reduce this
opportunity. No claim is made about overlapping beyond these current-MoE
windows, changing the grouping policy, or granting additional physical slots.

CPU snapshots also cover requests 2–5 without GPU traces; request 1 has no
chunk sample, not zero prefill work. Each arm's structural history is:

| Request | Groups | Positive transitions | Candidate experts | Next missing experts |
| --- | ---: | ---: | ---: | ---: |
| 2 | 63 | 7 | 8 | 604 |
| 3 | 62 | 4 | 5 | 601 |
| 4 | 56 | 5 | 9 | 536 |
| 5 | 59 | 3 | 6 | 562 |
| 6 | 55 | 3 | 4 | 521 |

Across these 295 groups / 290 transitions, only 22 transitions have candidates,
covering at most 32/2824 missing experts (**1.13%**). No device timing is inferred
for requests 2–5 from request 6's trace.

This updated trace also confirms the remap batching mechanism at model scale:
55 pre-H2D metadata copies/synchronizations for 55 groups (17184 packed metadata
bytes), versus 1074 scalar copies/synchronizations in the earlier scalar trace.
The current `h2d_submit` CPU phase still totals 33.689/33.653 ms in fixed/grow.
Those inclusive profiled CPU times are not pure DMA time or recoverable savings.

**Decision:** do not implement one-group lookahead under the current grouping
and slot budget. Its structural coverage and useful compute windows are too
small to prioritize the additional scheduling/ownership complexity. Next,
measure descriptor construction and repeated static validation on the H2D
submission path. Any cached validation must be invalidated when backing/storage
or slot layout changes and must retain live-owner, device and overlap checks.
That optimization is not implemented by this probe.

Against `/mnt/vdb/chengzhi/layerkv_remap_20260907_batch_01/`, all 12 requests
have exact output IDs and logprobs, identical expert materializations, H2D/D2H,
remap publication counts, token-group calls and materializations, KV lend/return
counts, new physical allocation counts and host-budget summaries. Driver
arguments differ only in output directory and trace request; engine arguments
only in the tracing flag. All KVC/expert/ownership guards pass, expert pointers
are stable, stale/alignment counts are zero, and final pending transfers,
blocked KV tokens and loans are zero. Host backing remains 1440 MiB mandatory
+ 96 MiB cached, with no idle pool or batch-owner storage. Trace-enabled E2E
times are not used to claim a speedup.
