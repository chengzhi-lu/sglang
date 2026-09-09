# Batched growth to the adaptive expert target

Adaptive mode now sizes one growth transaction from the target, available donor
pages and post-loan KV headroom. Greedy page selection stops at the headroom
limit; only a whole number of expert rows is retained. Parameter page ranges are
then built for that complete row count. A single VMM loan transaction maps all
selected pages before publishing the new tensor shapes/capacity and free slots.
Fixed mode retains its previous one-slot-per-step behavior.

This avoids repeated scans and intermediate expert shapes when moving, for
example, from 16 to 24 slots. `grow_count` counts growth transactions, while peak
capacity records the reached slot count. Existing stats record scan/selection
time separately. No extra physical backing is allocated by growth.

When capacity and target are both at base and no policy decision is due, the
controller skips the common-free allocator query. There is no outstanding loan
to recall on this path; scheduler admission still checks actual capacity, and
expert demand observation continues. Active loans still undergo headroom checks.

## Regression tests

- Real FP16/BF16 multi-layer VMM tests first failed when a single adaptive call
  stopped one slot short. They now publish two complete extra rows in one scan
  and one transaction, without duplicated common KV credit.
- A late failure at page 5 of w13's batch rolls back all prior mappings and
  claims; tensor capacity/free slots remain unchanged, common capacity is fully
  restored, and physical creates/ownership guards match the pre-call state.
- Four donor pages with three required per expert grow one full row rather than
  publishing a partial second row.
- A loan-free warmup test first failed on an unnecessary allocator query, then
  passes with the fast path.
- Full LayerKV suite: **334 passed, 19 warnings**. Targeted Ruff/Black and
  `git diff --check` pass. No full repository pre-commit run.

## Matched model rerun

Dry plan inspected, then the same B8 experiment as
[per-layer donors](layerkv_per_layer_donors_20260907.md), changing only output
directory to `/mnt/vdb/chengzhi/layerkv_adaptive_20260907_batchgrow_b8_01`.
BF16 Qwen3.6-35B-A3B/H100 NVL/TP1, eight simultaneous requests, 128 input and
32 output tokens, two rounds, 16 base slots/8 extra, adaptive interval 8 and
headroom 16. This is fixed-first, not a reversed-order repeated acceptance study.

The driver currently compares fixed base16 against adaptive base16..24 at the
same physical budget. That is useful for short-context expert benefit, but does
not provide the fixed-expert-heavy split needed to demonstrate *greater* long-
context admission capacity: adaptive cannot return more than its loans above
the same base16. A retained expert-heavy baseline and controlled arrivals remain
necessary for that research requirement. This limitation must not be disguised
as proof of increased actual batch.

### Result

`valid=true`: all 512 output token IDs and logprobs match exactly; all recorded
KV/expert/ownership guards pass and both physical pools are 478150656 bytes.
Both rounds retain actual B8 for 31 decode forwards. Adaptive round 1 does not
grow; round 2 reaches 24 slots in **one scan and one growth transaction**, with
zero growth physical creates. Scan/selection totals are 27.17/11.46 ms, versus
169.73/70.70 ms for eight incremental transactions in the prior experiment.

| Round | Fixed tok/s | Adaptive tok/s | Relative throughput |
| --- | ---: | ---: | ---: |
| 1, cold | 43.7150 | 43.4469 | -0.61% |
| 2 | 90.9411 | 84.7621 | -6.79% |

Round-2 TPOT is approximately 68.68 ms fixed and 75.30 ms adaptive. Materialization
counts are 2107/2044 and measured H2D time 271.20/261.67 ms. Recorded total LayerKV
Python overhead is 62.32/229.67 ms (includes more than donor work); the roughly
9.52 ms transfer saving does not justify that added control cost. The previous
adaptive throughput was 71.43 tok/s, but this is not a randomized attribution
study: batch growth and loan-free warmup skipping changed together and caches
were warm. No 20% acceptance or independent native B8 verification is claimed.

Next: avoid decisions that cannot repay control/transfer costs over useful
residency, and add a fixed expert-heavy retained split for long-context admission
comparison. Returning loans from a base16..24 policy cannot exceed fixed base16's
KV capacity, so repeating that baseline alone cannot prove the batch-growth goal.
