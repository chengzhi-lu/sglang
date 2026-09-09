# Decode benefit, batch semantics and handoff evidence

## Clarified research objective

The user clarified that the target is fixed-total-GPU-memory throughput through
adaptive KVC/expert residency and real request concurrency, not simply increasing
the token batch of the existing B=1 MoE call:

- Short context / large actual batch: favor resident experts to avoid repeated
  expert exchanges across a broad routed demand set.
- Long context / small actual batch: favor resident KVC, offload some experts,
  and use the freed physical memory for additional KV capacity and admission.
- Re-evaluate as the admitted batch changes; these are workload hypotheses,
  not an unconditional context-length-only switch.

Accordingly, the B=1 measurements below are mechanism evidence only and do not
validate the central throughput objective. The immediate priority is to audit
the expert-reclaim → physical KV capacity → scheduler admission chain, then
compare both residency preferences on bounded short/long-context concurrent
loads. A freed CUDA allocation alone does not enlarge a preallocated KV pool.
The shared-VMM experiments validate KV-to-expert lending and repayment; they
have not demonstrated expert-to-additional-KV admission capacity at larger B.

Current joint DP code scores KVC/expert reclaim splits for a supplied reclaim
target. It observes workload batch/context information, but that is not by
itself proof of optimizing reachable batch size or output tokens/s. The native
scheduler retains request admission ownership. The desired evaluation should
hold physical GPU memory and offered workload fixed, observe actual admitted
decode batch size, and report throughput with TPOT/TTFT limits. Do not hold B=1
fixed while claiming to test increased admission capacity.

## What the current implementation actually changes

Inspection and read-only reanalysis of all four `layerkv_chunk_recheck_20260907_*`
families; no new GPU runs, runtime changes or policy/default changes. The full
model is Qwen3.6-35B-A3B BF16. Local snapshot config has 256 routed experts and
`num_experts_per_tok=8`.

`scripts/layerkv_shared_perf.py:child` fixes `batch_size=1`, and all eight saved
engine configurations confirm `max_running_requests=1`. Generation requests are
sequential. There are no speculative/multiple-token decode inputs in this setup.
Thus each normal decode forward processes one token, with at most eight distinct
routed experts. Both 16 and 17 physical expert slots already cover this demand.

`_expert_chunked_core_required` enters the shared token-group path only when the
current routed expert union exceeds slot capacity. A one-token, top-8 decode
cannot cross the 16-slot threshold. `reuse` only changes existing groups' order;
it does not change their membership or merge groups. It therefore does not
directly enlarge decode's token batch in these runs. Its decode influence is
through residency/scheduler history left by prefill and previous requests.

Three quantities must remain separate:

| Quantity | What changes it | What it can improve |
| --- | --- | --- |
| Resident expert capacity | KV pages lent to expert weights, e.g. 16 → 17 slots | Fewer weight reloads; potentially fewer token groups at larger actual batch |
| Tokens per MoE invocation | Actual multi-token demand and its grouping | Less fragmentation; more tokens per call, conditional on routing |
| Concurrent request batch | Admission and actual scheduler batch formation | More decode tokens processed together; potentially more reuse per expert |

More resident experts alone do not increase tokens per expert GEMM. At actual
decode batch 2, top-8 routing has at most 16 distinct experts, so the same 16-slot
capacity already avoids token grouping. Starting at batch 3, a routed union of
17 experts could fit capacity 17 but not 16. This is a structural example, not
an observed routing distribution or measured speedup. Large batches with unions
above 17 can remain split. Same-request future autoregressive tokens cannot be
batched freely without changing the decoding algorithm.

If the intended meaning is that physical KV offload can admit more concurrent
requests, that is a separate plausible benefit which this B=1 benchmark has not
tested. KV memory handed to resident expert weights cannot simultaneously be
counted as additional on-GPU KV capacity. Admission, scratch/reload capacity,
GDN state and transfer bandwidth also constrain concurrency.

## Existing decode measurements

Use the latest unprofiled matched runs; aggregate requests 2–6 only, retaining
request 1 separately in the original comparison. Client decode is the elapsed
interval from first to last streamed output token, not isolated GPU compute.
Positive values below mean less decode time.

| Effect held separate | Repeat 1 | Repeat 2 |
| --- | ---: | ---: |
| input → reuse at fixed16 | +1.170% | +0.729% |
| input → reuse at grow17 | +2.063% | -0.321% |
| fixed16 → grow17 with input order | +0.208% | +0.414% |
| fixed16 → grow17 with reuse order | +1.110% | -0.639% |

Ordering and lending capacity are different comparisons; do not attribute one
effect to the other. Fixed16/grow17 launch order itself was not reversed, so the
capacity comparisons have additional ordering limitations. None establishes a
general batch-size-driven decode gain. In each repetition, all-six expert loads
are 5326 → 5321 for input fixed→grow, and 4520 → 4514 for reuse fixed→grow: the
extra slot changes only five/six net materializations in this particular trace.
These are total loads, not pure decode demand misses. Borrowed-slot-use counts
count dispatches touching a borrowed slot, not avoided transfers.

## Handoff: what can and cannot be concluded

Request 3 still has additional non-chunk loads with reuse: fixed16 240 → 298,
grow17 236 → 288, reproduced in both repetitions. Non-chunk is total minus
compact-prefill loads and includes scheduler prefetch. This establishes a
different residency trajectory, not which exact experts were displaced or when.

The existing stream samples cover every output token 1–64. For repeat-2 grow17,
requests 2–6 aggregate decode windows are:

| Output-token interval | input (s) | reuse (s) | Additional reuse time (ms) |
| --- | ---: | ---: | ---: |
| 1 → 8 (7 intervals) | 2.489030 | 2.499437 | 10.407 |
| 8 → 16 (8 intervals) | 2.536948 | 2.552102 | 15.154 |
| 16 → 64 (48 intervals) | 15.233809 | 15.273362 | 39.552 |

The extra 65.1 ms is not confined to the first few output intervals; compare
window lengths before interpreting absolute totals. Conversely, request-3
fixed16 has 58 extra non-chunk loads but its first 1→8 interval is slightly
faster under reuse in both repetitions (493.804 → 487.984 ms; 487.599 → 486.240
ms). Therefore a simple causal claim that prefill-tail eviction explains the
observed decode slowdown is not established by these measurements.

The competing explanations remain: altered early resident-set hits, later
scheduler prefetch/recall behavior, and timing/system variation at otherwise
repeatable work counts. Current JSON stores aggregate counters, not end-prefill
logical-to-slot maps or per-step routing/materialization records. The runtime
has a recent-copy descriptor buffer capped at 256 entries, but that buffer was
not serialized into these artifacts. It is not a recoverable historical trace.
Missing per-expert evidence must not be treated as zero activity or reconstructed
from aggregate counts.

## Next bounded validation

Do not change defaults or implement a tail-aware policy from the current timing
alone. First collect a diagnostic handoff snapshot plus early-decode records:
logical-to-slot map, current routed IDs, hit/miss IDs, demand versus prefetch
copies, slot capacity and lending/recall boundaries. CPU-owned metadata should
be preferred; any additional GPU reads must be isolated from performance timing.
Compare input/reuse using the same request history, especially request 3.

For the batch hypothesis, use a separate concurrent-decode experiment with
explicit request batch limits and confirm actual per-forward token counts,
unique expert counts, token-group sizes/call counts, tokens per expert, H2D
bytes, KV reload volume, TPOT and total output tokens/s. Check admission and
scratch/GDN capacity before launching; do not simply increase a CLI batch flag
while retaining a workload that cannot admit it. Keep total physical memory
and output correctness matched. Start with one small concurrent pair and a
shape/capacity check, not a broad matrix. No such run was launched by this audit.
