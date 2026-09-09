# FP16 fixed-arrival residency comparison

## Scope and launch

Artifact directory: `/mnt/vdb/chengzhi/layerkv_fp16_20260907_timed5_01`.
Read-only dry plan was inspected before execution; `plan.json`, both effective
engine argument files, full logs, per-round responses/stats and strict
`summary.json` are preserved. Both child processes and parent are terminal.
The parent exits1 because strict cross-arm correctness is not satisfied.

Qwen3.6-35B-A3B FP16 snapshot995ad96eacd98c81ed38be0c5b274b04031597b0,
H100 NVL GPU0/TP1, requests2/max-running2, short input128, late input6000,
late arrival1second independent of output progress, output64 each, rounds5.
Expert-heavy fixed20 versus adaptive16..20, retained across requests. Same
KV16384/scratch8192/block2048/reclaim40MiB, layer0 expert base16/extra4,
CUDA batch transfers, CPU-known preparation and batch remap. No output triggers
or activation probes are enabled in this performance pair.

Before launch, rounds1–2 were designated startup/warmup and rounds3–5 the
candidate measurement window. This is one launch per arm, fixed then adaptive;
three rounds are not three independent launch repetitions. All10 round-end
physical pools are478150656 bytes. Warm round guards pass, fixed retains24MiB
of loans, adaptive ends loan-free. Adaptive grows/recalls once in round2; later
growth is rejected by its cost gate. Fixed is B1 throughout rounds2–5; adaptive
has B2:55/B1:16 in round2, B2:50/B1:26 in rounds3–5. Offered limits did not change.

## Initial measurements, pending native correctness isolation

|Round|Fixed output tok/s|Adaptive output tok/s|Gain|Cross-arm token IDs|Max logprob delta|
|---|---:|---:|---:|---|---:|
|1 startup|14.3921|16.3730|13.76%|Exact|0|
|2 warmup|14.1935|21.7641|53.34%|Short request differs at index61|0.223423|
|3|14.8632|22.2602|49.77%|Exact|0.007385|
|4|15.0769|22.2441|47.54%|Exact|0.060615|
|5|14.9988|22.2289|48.21%|Exact|0.004097|

Rounds3–5 aggregate uses total384 output tokens divided by summed makespan:
14.97910595→22.24438878 tok/s, +48.50278%. This is **client end-to-end output
throughput**, not GPU kernel time or a decode-only kernel speedup.
Long request TTFT3.621–3.750s→0.722–0.758s. Short request TPOT60.706–61.973ms
→74.443–74.762ms; long TPOT61.278–61.895ms→63.343–64.070ms. Increased batch
improves serving throughput while per-request token latency regresses.
Round3–5 arrival lag is0.074–1.140ms fixed and0.166–1.018ms adaptive.

Strict summary remains `valid=false`, `physical_pool_matched=true`. Round1
fixed has not yet reached20 slots; round2 token mismatch and round4 logprob
delta exceed the unchanged0.01 threshold. Do not silently discard round4 or
count this as20% acceptance. Native serial and schedule-replayed parallel
references must independently establish whether the differences are inherent
batch effects. Those native runs are output references, not memory-matched
performance baselines. Independent/reverse-order repetitions and a KV-heavy
fixed alternative remain necessary to assess robustness and baseline sensitivity.

## Native isolation completed

Both native references completed with exit0:

- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_serial5_01/native.json`:
  max-running1, same5 input rounds and fixed1second arrival. Fixed policy
  rounds2–5: all512 token IDs and logprobs exactly match. Cold fixed round1
  runs B2 and has a short-request logprob difference0.140524 against this B1
  reference; its tokens match. Round1 is not in the measurement window.
- `/mnt/vdb/chengzhi/layerkv_fp16_20260907_parallel5_01/native.json`:
  max-running2, same5 input rounds, base1second arrival with diagnostic
  round-output-triggers0,8,13,13,13. All640 adaptive token IDs and logprobs
  exactly match, including round2 short-token differences and round4 logprobs
  seen in the cross-policy comparison.

Native parallel and adaptive streams both report long-request token counts
63,55,50,50,50 immediately before short-request completion in rounds1–5.
This supports the targeted batch-transition replay; a histogram or stream
timing is not itself a full ordered GPU execution trace. Exact complete outputs
and logprobs establish the relevant correctness result independently.

Therefore the cross-policy differences in this experiment are reproduced by
native batch execution, not evidence of a remaining LayerKV output error. The
original strict `summary.json` is deliberately unchanged (`valid=false`); its
criterion is cross-arm equality, not per-arm native equivalence. The separate
native evidence validates this particular mechanism comparison without raising
the logprob tolerance or discarding the offending round.

Result: a single-launch, native-validated candidate48.5% end-to-end throughput
gain over expert-heavy fixed residency, with short TPOT roughly22% worse. This
does not finish the research objective: no independent/reverse-order launch
replication yet, no best-fixed-split sensitivity check, no short-context/large-B
benefit established, and no representative mixed/real trace validation. The
current experiment remains single expert-layer0. All processes from this
comparison and both native references are terminal; GPU0 is released.

## KV-heavy sensitivity check: no dynamic advantage in this long-context case

`/mnt/vdb/chengzhi/layerkv_fp16_20260907_kvheavy5_01/fixed.json` completed,
same FP16 model,5 input rounds, physical478150656-byte pool and1second arrivals.
The only baseline change is kv-heavy: fixed16 expert slots, zero extra slots,
no loans. Round3–5 rates22.36824,22.28389,22.48163 tok/s aggregate to22.37762,
versus adaptive22.24439: **adaptive is0.5954% slower**, within the scale where
independent repetitions are needed to distinguish noise. Both use B2:50/B1:26
throughout the measurement window, and all384 measured tokens and logprobs
exactly match the parallel native reference. KV/comparability guards pass.

KV-heavy warmup round2 has B2:52/B1:22, unlike the existing native trigger8
reference. Its short request differs there (max logprob0.195781), so that
warmup comparison is not schedule-matched and is not declared validated.
Round1 and rounds3–5 are exact. No baseline results or failing summaries were
overwritten.

This materially changes the next action: the48.5% result demonstrates avoiding
expert-heavy admission blockage, **not superiority over a sensible static KV
split**. Prioritize demonstrating the other side of the tradeoff (short-context,
large-batch expert reuse) before multiplying replications of the weaker baseline.

The driver now exposes `--retain-experts-across-requests` independently of
`--baseline-split`. KV-heavy fixed stays at16/no loans, while adaptive can keep
borrowed pages across request boundaries and still recall on admission pressure.
Previously kv-heavy selection implicitly forced boundary recall in both arms,
preventing a fair retained-adaptive versus KV-heavy comparison. Defaults and
expert-heavy behavior remain unchanged. Summary guards validate declared
retention, correct admission policy, fixed capacity and physical credit/loan
consistency; output tolerances are unchanged.44 driver tests pass.
