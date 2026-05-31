# Codex Session Handoff

Date: 2026-05-31

Repository: `/sgl-workspace/sglang`

Branch: `layerkv/v0.5.12-integration`

Upstream: `chengzhi-lu/layerkv/v0.5.12-integration`

## Current Focus

LayerKV integration work is the active context. The staged changes cover
LayerKV CoResid runtime behavior, scheduler integration, KVC/expert validation
scripts, benchmark config, and local Codex agent skill files.

## Git State Before Push

The branch was ahead of upstream by one existing commit:

- `41331baa1 Optimize LayerKV CoResid dynamic pressure`

Additional staged changes were committed in the same session before pushing.

## Useful Validation Commands

Run targeted LayerKV checks before relying on performance or correctness
claims:

```bash
python3 scripts/layerkv_smoke.py
python3 scripts/layerkv_kvc_validation.py
python3 scripts/layerkv_expert_validation.py
python3 scripts/layerkv_pd_coresid_trace_replay.py
```

For wider repository checks:

```bash
pytest test/registered/unit/ -v
pre-commit run --all-files
```

## Resume Notes

- Start by checking `git status --short --branch`.
- Compare against the pushed upstream branch before adding new work.
- Keep LayerKV-specific validation first; avoid unrelated SGLang refactors.
- Important files in this handoff include:
  - `python/sglang/srt/layerkv/runtime.py`
  - `python/sglang/srt/managers/scheduler.py`
  - `python/sglang/srt/managers/schedule_batch.py`
  - `scripts/layerkv_*`
