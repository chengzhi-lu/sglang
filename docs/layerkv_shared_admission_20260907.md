# Expert-page recall before KV admission rejection

This continues the fixed-memory adaptive residency objective in
[the implementation log](layerkv_adaptive_residency_20260907.md). It does not
establish the 20% throughput target or increased full-model decode batch.

## Implemented interface

`LayerKVRuntime.recall_shared_expert_for_admission(shortage_tokens=...)` is an
explicit mutation at admission time, separate from read-only credit queries.
It is active only with `--layerkv-shared-expert-free-kv-donors`. It recalls existing
expert loans synchronously and returns the measured increase in common usable KV
addresses, not an estimate based on MB. No new physical backing is created.
The existing shared-mode requirement that overlap is disabled remains in force.

`PrefillAdder` invokes this interface before memory rejection in:

- Normal requests' total-token check, both before and after prefix locking.
- Ignore-EOS requests' initial input-token capacity check.
- Ignore-EOS requests' conservative future-decode reserve check.

Every decision re-reads real remaining capacity after recall. Returned counts
are not added optimistically to the budget. Fractional reservation shortages are
rounded up. Missing LayerKV interfaces are no-ops; ownership-transfer errors
propagate instead of admitting requests after a potentially partial failure.
Other scheduler constraints (SWA, request limits, input budgets, etc.) remain.

An actual shortage also records pressure even when no expert pages are currently
borrowed. While the scheduler reports a nonempty waiting queue, free-page expert
growth is paused; it resumes once the queue drains. This prevents immediately
borrowing back capacity just returned for queued demand. It is demand-driven
pressure control, not a complete context/batch working-set optimizer.

Runtime stats add `shared_expert_admission_recall_count`,
`shared_expert_admission_recovered_tokens`, and
`shared_expert_admission_recall_ms`. Shared-controller stats add
`admission_shortage_tokens` and `admission_pressure_skip_count`.

## Evidence and limits

`test_shared_admission.py` exercises the real PrefillAdder methods with small
allocator adapters: sufficient capacity, input/future-token shortage, ordinary
requests before/after locking, missing interface, fractional shortage, false
reclaim estimates and propagation of mapping errors.

CUDA tests in `test_free_kv_donors.py` connect the actual PrefillAdder capacity
query and recall path to a real SharedVMM pool. Lending three 2-MiB pages removes
4096 common KV tokens; admission restores the same pages and accepts a previously
blocked request. Physical bytes/create counts and ownership are checked. Another
test keeps the queue nonempty to prevent reborrowing, then drains it and verifies
that physical expert growth can resume. This validates an admission decision and
the page lifecycle, not an end-to-end model-throughput result.

Full LayerKV unit suite: **308 passed**, 19 existing/environment warnings:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Targeted formatting and `git diff --check` pass. The scheduler file has existing
E402 diagnostics due to imports after module initialization in HEAD; targeted
Ruff with E402 excluded passes. No unrelated import-layout rewrite was made.

The next model regression uses the unchanged 4-request, 2-round, 2048-in/16-out
fixed/lend pair from the earlier log. Results are stored separately under
`/mnt/vdb/chengzhi/layerkv_concurrent_admission_20260907_b4_01`.
Its same-length waves are primarily a regression/pressure-pause check: the old
controller still recalls loans at every request completion, so a mixed-arrival
trace is required to demonstrate admission-time recall in the full model.

The pair completed with `valid=true`, identical tokens/logprobs and matched
physical pools. Cold-round fixed/lend throughput is 10.4350/10.4242 tok/s.
Second-round throughput is 19.3260/16.9172 tok/s: lending remains 12.46% slower,
compared with the previous pair's 21.22% regression. This is not a 20% gain.
Actual batch remains B3/B1 in round 1, B2 in round 2 (mean B=2).

Lending skips 15 pressure-blocked growth attempts in each round. Second-round
donor scans fall from 30 to 15 and take 473.76 ms; the first B2 wave has TPOT
about 69.1 ms, while the queue-empty B2 wave still scans and has TPOT about
101.3 ms. There are **zero admission-time recalls** in this full-model trace;
the successful loan is still repaid by request completion. Do not claim this
run demonstrates additional admitted requests.

The saved fixed arm's skip counter also counted 15 no-op steps per round where
its growth target was already satisfied. A subsequent counter-only correction
checks the target first, with a focused regression test; it does not change
allocation or model execution. Raw artifacts retain the original counter values.

Remaining: native-reference concurrent correctness, per-layer free-page capacity
scaling, expert working-set demand control, safe cross-request residency, mixed
short/long arrivals, and a repeated fixed-memory throughput comparison proving
actual batch growth and the 20% target without concealing latency costs.
