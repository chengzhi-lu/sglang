# FP16 short-context B8 retained-expert pilot

Artifact directory:
`/mnt/vdb/chengzhi/layerkv_fp16_20260907_shortb8_retained_01`.
Dry plan inspected before execution; effective CLI/engine settings, both full
logs, responses, stats and summary are preserved. Same Qwen3.6 FP16/H100 GPU0
setup as the fixed-arrival studies, but8 simultaneous requests, input128,
output64,5 rounds, fixed16 versus adaptive16..24, baselinekv-heavy and explicit
`--retain-experts-across-requests`. Other KV/pool/transfer settings unchanged.
The experiment tests expert reuse at matched actualB8, not admission gains.
Rounds1–2 excluded from candidate warm measurement, as in the earlier pilot.

Both processes and parent completed (exit0), strict summary `valid=true` and
physical pools matched. All5 rounds have exact cross-arm tokens and logprobs;
actual batchB8 for63 decode steps, no KVC evictions. This is cross-arm
correctness, not an independent native B8 reference. Full LayerKV regression
after the driver flag change:391 passed,19 warnings.

|Round|Fixed tok/s|Adaptive tok/s|Fixed materializations|Adaptive materializations|Adaptive end slots|
|---|---:|---:|---:|---:|---:|
|1 startup|51.0662|62.5677|2885|2885|16|
|2 warmup|92.7021|99.1679|3568|3568|16|
|3|96.6110|97.3067|4077|4077|16|
|4|97.2893|94.7092|3616|3573|16|
|5|96.3234|95.3886|3716|3716|16|

Materializations above are per-round differences of shared-VMM
`token_chunk_materializations`, not cumulative totals or inferred zeros.
Warm rounds3–5 aggregate96.73955→95.78894 tok/s, **−0.98265%**. Cold gains do
not indicate a residency benefit: both remain16 with identical materialization
counts. Dynamic grows once in round4 (measured transition65.864ms), but has
already returned to16 by the round-end snapshot. Its budget reason is
`no-miss-benefit` with zero shadow saved misses at each round end. No20% gain.

Next falsifiable hypothesis: estimator input loses token-group reuse.
`_expert_chunked_core_required` feeds `ResidencyBudget.observe` only the unique
expert IDs for the whole forward. Actual `run_token_chunks` groups token rows
by expert-union capacity, and larger capacity changes those groups and repeated
materializations. Unique-set LRU cannot represent these within-forward group
differences. This is a candidate explanation, not yet a proven fix: capture or
replay the already CPU-known routing rows, compare predicted misses at16/24
against actual grouped execution, and retain strict numerical/ownership gates.
Do not add another GPU routing readback merely to gather the same data.
