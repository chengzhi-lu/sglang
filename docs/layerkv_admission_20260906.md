# LayerKV common-address prefill admission repair

## Failure and cause

The original four-request performance check stalled on request three with
`waiting=1 running=0 kv_available=0`, although arena telemetry reported 12289
free tokens. See `layerkv_shared_perf_20260906_01` under
`/mnt/vdb/chengzhi/` for the unchanged original evidence.

There were two separate bookkeeping problems:

1. Finished native KV requests transfer ownership to the per-layer arena and
   publish reusable locations through lazy overwrite lists/bitsets/bitmaps.
   Prefill admission and common-address allocation only considered ordinary
   free lists. Thus native availability reached zero while usable arena
   addresses were invisible to admission.
2. After common arena allocation, indexed request cleanup published locations
   for overwrite reuse without removing their arena `allocated` ownership.
   Counting reusable locations while excluding allocated addresses therefore
   still lost capacity on subsequent requests.

The first partial-fix replay (`layerkv_admission_20260906_01`) completed three
requests, then stalled on request four. Its process group was stopped and all
artifacts retained. It did not include the second fix and is not a passing run.

## Fix and invariants

- Prefill admission now uses the exact intersection of reusable addresses
  across all full-attention KV layers, combining ordinary and lazy free
  representations. It excludes allocated, protected and non-arena addresses,
  including locations blocked by expert loans. Admission does not mutate
  ownership or materialize pages.
- Common prefill allocation promotes those same reusable addresses out of the
  lazy representations into ordinary common free lists. It does not replace KV
  pointers or create physical backing for already available addresses.
- If arena addresses alone are insufficient, native capacity only needs to
  cover the missing amount, not the entire request.
- Indexed finished-request cleanup clears allocated ownership for exactly that
  request's tracked locations before publishing them for overwrite reuse.

Do not replace this intersection with the minimum per-layer free count: two
layers can each have free slots but no common physical address. Decode's
per-layer admission semantics remain unchanged. Generic scheduler policy and
expert donor selection were not modified.

## Regression coverage

`test/registered/unit/layerkv/test_hybrid_kvc.py` adds three regressions:

1. Real `PrefillAdder` capacity properties after exhausting the native allocator,
   followed by actual common prefill allocation and cleanup four times. Native
   availability remains zero; arena admission capacity returns to its initial
   value after every release.
2. Disjoint layer free addresses grant zero common prefill credit and cannot
   satisfy common allocation, despite nonzero per-layer free counts.
3. Mixed free representations, repeated read-only admission checks, exclusion
   of allocated/protected slots, and an allocation requiring both arena and
   native capacity.

The first regression was observed failing before both corresponding fixes:
initial credit was 0 instead of 16; after fixing admission, post-release credit
was 8 instead of 16. All 41 LayerKV unit tests subsequently passed:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv/python
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python -m pytest \
  ../test/registered/unit/layerkv/ -q
```

Import smoke and `git diff --check` passed. The regression file passes Black,
isort and Ruff checks. Full-file Ruff reports the same 29 existing findings in
the three runtime files as HEAD (unused imports and forward type names); these
unrelated findings were not rewritten. No full-repository pre-commit run or
package installation was performed.

## Bounded model replay

The final-code replay uses H100 NVL GPU 0, local Qwen3.6-35B-A3B, TP=1, batch=1,
BF16, six sequential requests per arm, and 4096 synthetic input IDs / 64 greedy
output tokens per request. Fixed16 and grow17 differ only in permitted extra
expert slots (0 versus 1). KV capacity is 16384, context 8192, scratch 4095,
block size 2048, reclaim target 40 MiB, expert layer 0, initial slots 16.
Graphs, overlap, radix and chunked prefill remain disabled. The tokenizer stays
enabled to match the original admission failure; unavailable scheduler prefill
timestamps remain null. This does not validate FP16 or concurrent serving.

Run from the active worktree, with a fresh output directory:

```bash
cd /home/chengzhi/github/sglang-perf-layerkv
PATH=/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin:$PATH \
/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python \
  scripts/layerkv_shared_perf.py \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --output-dir /mnt/vdb/chengzhi/layerkv_admission_20260906_02 \
  --gpu 0 --dtype bfloat16 --rounds 6 \
  --input-tokens 4096 --output-tokens 64 --execute
```

`PATH` only exposes the existing environment's external tools (including Ninja);
experiment settings use CLI arguments. Omit `--execute` for the dry plan.
Saved `plan.json` and `*.engine_args.json` contain effective arguments.

Replay results (`layerkv_admission_20260906_02/summary.json`): **valid=true**.
Both arms completed all six requests, with no `LayerKV no-prefill progress`
warnings. Saved engine arguments differ only in
`layerkv_shared_expert_extra_slots` (0 versus 1).

| Request | Fixed16 total (s) | Grow17 total (s) | Grow17 pages lent / returned | Borrowed-slot uses |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9.236 | 9.370 | 3 / 3 | 23 |
| 2 | 7.809 | 8.143 | 0 / 0 | 0 |
| 3 | 6.059 | 6.621 | 3 / 3 | 43 |
| 4 | 5.720 | 5.966 | 0 / 0 | 0 |
| 5 | 5.485 | 5.768 | 0 / 0 | 0 |
| 6 | 5.509 | 5.886 | 0 / 0 | 0 |

All 384 paired output token IDs and logprobs match exactly (maximum logprob
delta 0). Every request passes KV/expert, ownership and pointer-stability guards;
stale entries, alignment violations, physical failures, final host KV usage,
outstanding loans, blocked KV locations and growth-time physical page creations
are zero. Free arena tokens return to 12289 after every request in both arms.
Each actual loan transfers three 2 MiB pages, expands 16 -> 17 slots, and returns
to 16 slots at request completion. Initial expert backing allocation remains
distinct from loan-funded growth.

Times are client end-to-end seconds, including prefill, decode and IPC; the
first request includes lazy setup. Prompts change across rounds, arms run in
fixed order, and most growth-enabled requests receive no loan. This is a bounded
admission/correctness validation, not evidence of an expert-residency speedup,
steady-state performance, or long-running/concurrent serving stability.
