# Retained expert admission: repeated allocator-residue diagnosis

Continuation of `layerkv_retained_experts_20260907.md`. No runtime fix or
performance acceptance is claimed here.

Three bounded reproductions use the same timed workload: Qwen3.6-35B-A3B BF16,
GPU0, requests2, input128/6000, late arrival1 second, output64, rounds3,
fixed retained expert slots16+4 and unchanged physical KV budget. Each parent
writes its exact command/engine arguments in `plan.json` and stops at a bounded
timeout (100/85/80 seconds respectively). The fixed child stalls in round3;
the parent exits1 without launching the adaptive arm. Artifacts are preserved:

- `/mnt/vdb/chengzhi/layerkv_retained_20260907_probe_01`
- `/mnt/vdb/chengzhi/layerkv_retained_20260907_probe_02`
- `/mnt/vdb/chengzhi/layerkv_retained_20260907_probe_03`

## Observed state

Read-only probes at the existing five-second no-prefill warning query exact
common allocator capacity rather than relying on the previous summary's cached
common-free statistic. All three reproduce the following idle state:

| State | Value |
| --- | --- |
| Waiting / running requests | 1 / 0 |
| Pending eviction | 0 |
| Waiting input / maximum output | 6000 / 64 |
| Arena reserved locations | 8192 |
| Exact common reusable locations | 4157 |
| Protected locations, each layer | 0 |
| Request cleanup-state keys | empty |
| Allocated locations, layer3 | 2048 |
| Allocated locations, layers7,11,15,19,23,27,31 | 4035 each |
| Allocated locations, layers35,39 | 1987 each |

Relative to the retained loan pattern, layers after3 retain1987 additional
allocated locations. The idle credit is therefore insufficient for admission;
this is not merely a stale batch-is-full flag. The previous round's reported6144
common-free count did not prove current capacity after subsequent transitions.
During the third short request, exact credit is4005 and its cleanup index exists;
after it finishes, the index disappears and credit rises only to4157.

## What the probes rule out, and what remains

The scheduler already retries empty running batches and clears batch-is-full.
The idle failure has no protected locations or pending eviction to wait for.
Additional probes on `_alloc_per_layer_locs`, `_reuse_evicted_per_layer_locs`
and `_import_per_layer_arena_locs` for requests larger than128 locations do not
show a1987-location transaction. They do show ordinary2048-location layer3
reloads through `_run_deadline_scheduler` at steps2 and128. This does NOT rule
out scalar imports, common allocation, multiple small allocations, incomplete
release, or cleanup indexing: the probe size threshold explicitly limits scope.

Next: capture address ownership deltas at the second long-request cleanup and
third short-request allocation/cleanup, distinguishing loan-owned addresses from
request-owned ones. Build a regression at that actual ownership seam before
changing release behavior. Do not clear all allocated sets when idle: those sets
also protect physical pages currently backing experts.

All DEBUG-retained probes were removed after their processes terminated. No
model process remains from these runs. This turn narrows the liveness failure;
the20% goal, multi-round validity and final architecture remain unfinished.

## Located ownership-release defect

The request-finish probe in
`/mnt/vdb/chengzhi/layerkv_retained_20260907_ownership_01` locates the transition
at the second long request's cleanup, not the next short request. Before cleanup
(step189), layers after3 still have all6000 arena-allocated prefill locations,
whereas their request cleanup indices contain only4015 locations (6000-2048+63).
After cleanup they retain1987 non-loan addresses. Layer3 had reloaded its2048
evicted locations and correctly tracked their cleanup, so it retains only loans.

Both synchronous eviction and asynchronous completion remove evicted addresses
from request cleanup and publish overwrite bits, but omitted removing those same
addresses from `_per_layer_arena_allocated_locs`. Native prefill did not populate
that set, explaining why the first round worked. Arena-allocated later prefills
expose the defect. Decode reuses61 of the2048 evicted addresses; the remaining
1987 stay falsely allocated after the request is gone.

Fix: immediately before overwrite publication, after backup completion, remove
exactly the published addresses from that layer's allocated set. Synchronous and
asynchronous paths both implement this. No idle-wide clearing, extra admission
credit or forced expert recall is introduced.

`test_eviction_releases_arena_ownership_before_publishing_reuse` exercises real
CUDA host backup with pre-existing arena ownership and an unrelated allocated
sentinel. Both sync and async cases failed with `{2,5,7}` instead of `{7}` before
the fix; both now pass. Async additionally proves ownership persists until
finalization. Published locations become common reusable capacity, and request
cleanup leaves the unrelated owner intact. Full suite:368 passed,19 warnings.
Temporary ownership probes are removed. Original-model rerun is recorded under
`/mnt/vdb/chengzhi/layerkv_retained_20260907_releasefix_01` with the same workload
and a150-second per-arm safety timeout.

## Original three-round model verification after the fix

Both arms now complete all three rounds, including the previously stalled third
long request. Fixed retains20 slots/24 MiB loans in rounds2–3; it does not recall
experts to escape the failure. Physical backing is478150656 bytes in both arms
throughout. Recorded ownership/KV guards pass in rounds2–3. All temporary probes
were removed before this run; GPU processes are terminal afterward.

| Round | Fixed tok/s | Adaptive tok/s | Apparent gain | Actual fixed / adaptive decode |
| --- | ---: | ---: | ---: | --- |
| 1 cold | 14.3617 | 16.2992 | 13.49% | B2:63 / B2:63 |
| 2 | 14.1791 | 21.7738 | 53.56% | B1:126 / B1:16,B2:55 |
| 3 | 14.7590 | 22.2826 | 50.98% | B1:126 / B1:26,B2:50 |

**Overall valid=false; these gains are not acceptance.** Round1 never establishes
the configured fixed expert-heavy capacity (final16 slots), so its baseline guard
fails despite exact outputs/logprobs. In round2 the long request first differs at
zero-based output position58; in round3 the short request first differs at30.
The other request's token IDs match in each round. Maximum cross-arm logprob
differences are0.36209 and1.73742. This goes beyond the earlier run's logprob-only
difference and needs fresh native/schedule-matched output validation, not reuse
of that earlier correctness conclusion.

Timed-arrival deviations are fixed/adaptive+0.507/+0.902 ms in round2 and
+0.904/+1.223 ms in round3. Long-request TTFT improves3.971→0.728 s and
3.773→0.746 s; short-request TPOT worsens64.16→79.58 ms and62.28→74.30 ms.
The actual-batch mechanism is observed, while the output and20% acceptance gates
remain open. These are rounds in one fixed-first process pair, not independent
repeats or reversed-order confirmation. Full-model precision remains BF16.
