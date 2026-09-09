# Staggered short/long requests: admission recall diagnostic

This tests the missing full-model path in
[shared admission](layerkv_shared_admission_20260907.md). It is not yet a valid
performance result: the first pair triggers admission recall but diverges in output.

## Driver

`scripts/layerkv_concurrent_perf.py` adds explicit arguments:

- `--late-after-output-tokens N`: submit a later wave after request 0 streams N
  output tokens. Zero retains the previous simultaneous request mode.
- `--initial-requests N`: number of requests in the initial wave (default 1).
- `--late-input-tokens N`: input length for later requests.

The two waves use the Engine's existing event loop and async streaming interface.
Per-wave response indices are mapped back to global input positions. Raw events,
per-request submission times, prompt lengths and final responses are retained.
TTFT is measured from each request's own submission, not the start of the entire
round. Batch makespan remains separate. Failure of the initial wave releases the
arrival waiter; pending tasks are cancelled and joined on error. CPU tests cover
index mapping, trigger ordering, missing trigger/early failure and TTFT boundaries.

The trigger is tied to output progress, **not fixed wall-clock arrivals**. This
is a mechanism diagnostic, not the 20% fixed-offered-load acceptance experiment.
Actual overlap is checked and required; a delayed request submitted after the
initial wave ends must not be presented as a concurrency result.

An internal `--arm native` executes the same requests with LayerKV disabled as
an independent output reference. It records `reference_only=true` and null
LayerKV stats; it does not invent decode batch counts or supply a memory-matched
performance baseline.

## Initial run

Artifact directory: `/mnt/vdb/chengzhi/layerkv_staggered_20260907_01`.
The dry plan was inspected before execution. H100 NVL GPU0, Qwen3.6-35B-A3B BF16,
TP1, same transfer/cache/preparation settings as the earlier concurrent driver.

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  ../scripts/layerkv_concurrent_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_staggered_20260907_01 \
  --requests 2 --max-running-requests 2 --rounds 1 \
  --input-tokens 128 --output-tokens 32 \
  --late-input-tokens 6000 --late-after-output-tokens 4 --execute
```

PATH selects existing external framework executables; all experiment settings
are command-line arguments, and all effective engine arguments are saved.

Both arms execute 27 decode forwards at B2 and eight at B1 (mean B=1.7714).
The later long request enters prefill with one request already running. Lending
triggers **one admission-time recall, recovering 6144 common KV tokens**. Across
the run it lends and returns six physical pages in two growth episodes, with
zero outstanding loans or growth physical creates; pool bytes and guards match.

Nevertheless `valid=false`: request 0's 14th output logprob and 15th token ID
diverge; request 1's 10th output logprob and 11th token ID diverge. Those correspond
to the same joint decode step after the later prefill. Maximum logprob difference
over the complete responses is 2.347874. Fixed/lend throughput of 9.7329/9.1000
tok/s is therefore **not an accepted performance comparison**.

The reduced reproduction keeps the same inputs/trigger but uses
`--output-tokens 16`, under `layerkv_staggered_20260907_min_01`. The first priority
is output correctness with an independent native reference, then page-capacity
scaling and the full short/long adaptive policy. Admission recall alone is not
completion of the research objective.

## Follow-up isolation (not a fix)

The reduced workload's native reference matches fixed16 tokens and logprobs
exactly. Temporarily suppressing only regrowth after the admission recall also
matches native (`layerkv_staggered_20260907_no_regrow_01`), while retaining the
first loan and admission recall. That guard is not retained as the final policy.

The selected-page trace shows the second loan uses common-free page 7 from the
K/V allocations of layer 3 and K allocation of layer 7. An independent copy of
the original expert weights was compared against every actual routed slot on
all 19 decode steps (`layerkv_staggered_20260907_weights_01`): weight checks pass,
but output divergence remains. This rules out weight-content corruption in that
instrumented run, not all routing, compute or KV-state faults.

A standalone BF16 Triton MoE comparison on H100 NVL (seed 7, batch 2, hidden
2048, intermediate 512, top-k 8) relocates active expert 15 to slot 16 in an
otherwise identical 17-slot allocation. Outputs are bitwise equal, max error
0.0. This synthetic case does not reproduce the full-model bug and cannot
establish full-model correctness. It uses the existing venv and its ninja via
PATH, without installing packages. Temporary original-weight copies and
per-decode D2H audit instrumentation have been removed from runtime code.

The outstanding investigation is the recall/regrowth transition, including KV
mapping and full-model expert dispatch. No performance acceptance follows from
these diagnostics.

### Same-input full-model expert-core reference

`/mnt/vdb/chengzhi/layerkv_staggered_20260907_core_01` runs the reduced lending
child with a temporary independent GPU clone of the selected layer's original
full expert weights. Each decode saves activations before the inplace core,
executes the actual compact core, and executes native Triton fused experts with
the saved activations, original logical top-k, and original full weights.
The reference copies the actual runner configuration, changing only the expert
counts to the full count. It does not feed its output back into generation.

All 19 decode forwards match bitwise (maximum absolute error 0.0), including the
16-to-17-slot transitions and the step where final output diverges. Comparison
of saved responses against `min_01/native.json` still finds zero-based first
logprob differences at 13/9 and first token differences at 14/10 for requests
0/1. Thus the symptom survives the probe while the selected layer's decode core
is correct on the actual inputs observed. This does not independently validate
prefill chunking or prove which KV operation is faulty.

The next probe should locate the first divergent KV read/write boundary around
recall/regrowth. This instrumented run uses extra reference GPU memory and
synchronization, so its timing is not a memory-matched performance result.
Temporary `[DEBUG-stagger-core]` source and full-weight clones were removed
after the run; the output artifacts remain for comparison.

### First full-attention mismatch localized

`/mnt/vdb/chengzhi/layerkv_staggered_20260907_attn_01` repeats both arms with
temporary SHA256-prefix fingerprints of Q/K/V immediately before each full
attention call and its output immediately afterward. Both logs contain 210
records: ten full-attention layers, two prefill forwards and 19 decode forwards.
Extract the tagged substring anywhere in each log line: concurrent logging may
prepend another message, so a startswith-only filter incorrectly drops records.
The recorded position count is `positions.numel()` (three-axis mRoPE), not batch.

The first differing record is zero-based 143: **layer 15, decode step 13, B2**.
All preceding recorded boundaries match. At that boundary the Q/K/V fingerprints
match (`e6f6348d97e2b936`, `5cd7910de42df55b`, `5b53a277552f7af5`) but attention
output differs: fixed `95eaf479be7fd406`, lend `bc03cfbcc9e744c9`.
The pair still fails output validation with maximum logprob delta
0.8494990020990372, while physical pools and aggregate guards match.

This localizes the first observed difference to attention's KV/index/execution
state, not an already-divergent Q/K/V input. It does not yet distinguish incorrect
reload contents from stale metadata or another attention-state issue. In
particular, layer 15 is not one of the second loan's donor layers (3 and 7), so
inspect the shared reload/scratch selection and indices rather than assuming a
direct write into a donated page. Temporary model fingerprint code is removed.

The uninstrumented LayerKV suite passed 313 tests (19 warnings) after core-probe
cleanup, before installing the attention fingerprint probe. The attention pair
is correctness instrumentation, not accepted throughput evidence.

### Root cause: valid decode address permutations deleted during writes

`/mnt/vdb/chengzhi/layerkv_staggered_20260907_kvread_01/pair` repeats the reduced
pair, capturing actual indexed K/V, Q, indptr and split counts at layer 15 steps
12/13 in the parent directory's `read_{0,1}_{12,13}.pt` files. The probe evaluates
the usual buffer getters once, then captures the post-prepare kernel operands.
The same final-output failure survives the probe.

At step 12, all logical-order K/V contents, Q and split counts match despite
different physical addresses. At step 13, only packed rows 140 and 6149 differ:
the newly appended tokens of the two requests. The lending arm's K/V rows are
exactly exchanged, not corrupted historical KV. Its canonical output locations
are `[12416, 12417]`, while the corresponding layer-specific attention indices
are `[12417, 12416]`. Fixed locations/indices are `[10363, 10364]`.

`_translate_per_layer_locs()` incorrectly inferred staleness whenever both ends
of a nonidentity mapping appeared in the current write batch. A valid two-cycle
therefore got deleted, writes went to canonical locations, and attention retained
the correct permuted indices. Larger cycles fail the same way. Allocation already
records per-layer mappings and handles identity reuse; translation must not
infer ownership changes from overlapping address sets.

`test_kv_write_mapping.py` exercises the real wrapped KV write with two-cycle,
three-cycle and disjoint mappings. Before the fix: **2 failed, 1 passed**, with
the same exchanged-row symptom. The fix removes only the overlap-based deletion
from translation. After the fix: the complete LayerKV suite is **316 passed,
19 warnings**. Temporary backend capture instrumentation is removed; captures
remain. The uninstrumented original 32-output-token model rerun is recorded
separately below once its result is available.

### Uninstrumented original workload after the mapping fix

`/mnt/vdb/chengzhi/layerkv_staggered_20260907_mappingfix_01` uses the original
command above with that output directory: two requests, 128/6000 input tokens,
32 outputs each, later request after four streamed tokens, same BF16 model and
physical budget. **valid=true**, all 64 output tokens/logprobs exact across arms,
maximum logprob delta 0.0. Fixed outputs also match the pre-fix fixed artifact.
This keeps two expert growth episodes, one admission recall restoring 6144 KV
tokens, peak 17 slots, no growth physical creates, and zero final loan bytes.
Both arms finish with 478150656 shared physical bytes and 228 physical creates.

Actual batches remain identical: 27 B2 forwards and eight B1 forwards. Fixed
throughput is 9.9337 tok/s, lending 9.1752 tok/s (-7.64%). This is still an
output-triggered diagnostic, not a matched-wall-clock-arrival performance study
or a 20% result. The fix establishes the recall/regrowth correctness path without
disabling regrowth. Budget selection and actual concurrency gains remain open.

An independent `--arm native` child with the same 32-output workload is saved in
the same directory. Both fixed and lending match native for each request's 32
token IDs and all logprobs (maximum absolute error 0.0). Native disables LayerKV
and is an output oracle only, not the same-memory performance baseline.

Validation: Black/isort on the new test, `git diff --check`, and targeted Ruff
with the allocator's existing unused-import F401 findings excluded pass. Full
repository pre-commit was not run. No model/backend debug edits remain.
