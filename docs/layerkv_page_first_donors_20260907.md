# Page-first ordinary-free donor discovery

Motivation: graph-enabled adaptive B8 round3 still performed8 unsuccessful
donor scans costing110.66ms. See `layerkv_native_moe_graph_20260907.md` for the
matched replay comparison and14.15% diagnostic throughput gain versus fixed16.

Ordinary-free availability lacks the complete mutation-version contract needed
for safe negative caching. This change does not enable that cache. Every scan
still reads current allocator sets, pending bits, cleanup bits and mapped pages.
Admission credits, page claims, VMM mapping, positive donor ordering and backing
publication remain unchanged.

The scan now considers whole mapped pages first. Ordinary-free lists become
sets without expanding all pending bits. Fully pending pages need no free-list
enumeration; fully ordinary pages use subset checks. Mixed pages first reject
an absent lowest missing location, then check all remaining missing locations.
Only complete free candidates undergo reserved/live/protected checks. Cleanup
overlap rejects the page. Page0 and partial tails remain excluded.

An initial whole-reserved/free bitmap implementation was rejected on CPU cost;
it was not model-benchmarked. Eagerly expanding every mixed-page missing bitmap
was also expensive, motivating the cheap first-missing-location rejection.

Final CPU diagnostic on the existing ten-layer fragmented fixture,20 calls per
method in one process: independent set reference5.122ms/call, page-first0.753ms.
The reference reproduces the donor set using direct set membership; it is not
an exact historical-runtime timing baseline. The earlier asynchronously launched
measurement labeled `baseline_fragmented_ms_per_call` is not used because module
import may have overlapped the edit. No end-to-end speedup follows from these
CPU measurements.

Validation from the worktree's `python/` directory, existing environment:

```bash
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/test_donor_scan.py \
  ../test/registered/unit/layerkv/test_free_kv_donors.py -q \
  -k 'page_first or fragmented_free or common_free or backed_donors or distinct_layers_lend or donor_selection_reuses' \
  --disable-warnings
```

18 passed. After adding the lowest-missing-location rejection, reran only
`test_donor_scan.py -k page_first`:8 passed. These differential cases include
in-place reserved/free/pending/allocated/protected/cleanup changes and mapped-page
removal; they compare exact ordered donor lists before and after mutation.
CUDA coverage before that final rejection includes FP16/BF16, batched/scalar
mapping, distinct-layer reuse and nonduplicated common admission credit.
Ruff and `git diff --check` passed. No full suite, profile or full model rerun.

Next: one adaptive child with the same B8/replay settings and old exact outputs,
checking donor_scan_ms, unchanged slot49, guards, memory and warm TPOT. Do not
rerun the existing fixed/reference arms just to test this local scan change.
The measured110.66ms cost alone has only about3% end-to-end headroom; the20%
residency research target still needs stronger workload/baseline evidence.

## Full-model targeted validation completed

Artifact: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_b8_pagefirst_01/lend.json`.
One adaptive child, identical engine arguments to `b8_nativegraph_02`, using
the existing FP16 model/environment, B8,input128,output64,rounds3,scratch2048,
base16,max64,retention enabled and native replay limit8. The exact command is
the adaptive command in `layerkv_native_moe_graph_20260907.md` with output directory
changed to the artifact directory above. Effective engine arguments and logs
are saved next to the result. Process exited0; no fixed/reference rerun.

All1536 output tokens and logprobs match the existing adaptive reference exactly
and are finite. All KVC/expert/ownership guards pass; stale/alignment/physical
failures remain0. All rounds retain49slots, pool478150656bytes, actual B8 for63
decode forwards, and replay counters2379/4836/7293. Each round's full memory
snapshot matches the old adaptive snapshot field-for-field. Materializations
remain1929/1955/2004, confirming unchanged residency/transfer demand.

| Round | Old scan ms | New scan ms | Old adaptive tok/s | New tok/s | Existing fixed tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (cold) | 113.790 | 40.312 | 68.906 | 69.002 | 65.390 |
| 2 | 105.035 | 5.855 | 128.866 | 131.161 | 114.820 |
| 3 | 110.657 | 3.326 | 127.111 | 130.400 | 111.355 |

Each round has8 scans; round1 has7 misses (initial growth), later rounds8 misses.
Round3 mean TPOT53.238ms, TTFT0.57209s. New throughput is+2.59% versus the old
adaptive round and+17.10% versus existing graph-enabled fixed16. Only one round
after two warmups is available, no independent repetitions; do not claim20%
acceptance or superiority to the best fixed split.

The targeted scan bottleneck is now3.3ms/round, so further scan tuning has little
headroom. Next research priority is a bounded same-engine short→long→short
workload transition, preserving residency across rounds and checking real queued
admission/recall and subsequent regrowth, with common optimizations applied to
fixed/adaptive. Current driver varies arrival triggers per round but not request
counts/context lengths; extend explicit workload configuration before launching
another experiment family. This is a mechanism gate before main real-trace
evaluation, not a replacement for it.
