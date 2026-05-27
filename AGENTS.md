# Repository Guidelines

## Project Focus

This branch centers on LayerKV integration in SGLang: policy planning, KVC/expert reclaim, arena-backed residency, reload/materialization scheduling, and runtime stats validation. Treat generic SGLang changes as supporting work unless required by LayerKV.

## Project Structure & Module Organization

LayerKV runtime code lives in `python/sglang/srt/layerkv/`, with server integration exposed through arguments such as `--enable-layerkv`, `--layerkv-mode`, `--layerkv-policy`, and KVC scheduler/debug flags. Validation and experiment drivers are in `scripts/layerkv_*.py`; shared helpers are in `scripts/layerkv_eval_common.py`. The evaluation plan is in `docs/layerkv_evaluation_plan.md`. Core runtime code remains under `python/sglang/srt/`, kernel/package work is in `sgl-kernel/`, and CI tests live in `test/registered/`.

## Build, Test, and Development Commands

- `pip install -e python[dev]`: install SGLang with development and test dependencies.
- `python3 scripts/layerkv_smoke.py`: import-level smoke test for `LayerKVConfig` and `LayerKVRuntime`.
- `python3 scripts/layerkv_server_smoke.py --scenario joint_dp`: end-to-end server launch and `/generate` validation.
- `python3 scripts/layerkv_kvc_validation.py`: validate physical KVC eviction/reload semantics and stats guards.
- `python3 scripts/layerkv_policy_eval.py`: compare policies such as `kv-first`, `expert-first`, ratios, and `layer-aware-joint-dp`.
- `pytest test/registered/unit/ -v`: run fast unit tests for non-server logic.
- `pre-commit run --all-files`: run formatting and repository checks before PRs.

## Coding Style & Naming Conventions

Use Python 3.10+ and follow existing SGLang patterns. Python is formatted with Black and isort; Ruff checks selected import and undefined-name issues. Keep LayerKV names explicit: prefer `layerkv_*`, `kvc_*`, `expert_*`, `planner_*`, and `*_reclaim_mb` for metrics and arguments. Runtime hot paths must avoid unnecessary CPU-GPU synchronization, repeated per-layer checks, and destructive KVC pointer replacement.

## Testing Guidelines

Every LayerKV change should include a correctness or validation path. For KVC work, check `kvc_guard_pass`, stale entries, alignment violations, evict/reload counts, and physical reclaim MB. For expert or joint planning work, verify planned versus physical reclaim, materialization counts, comparability, and policy fractions. Prefer deterministic smoke tests first, then real-trace or controlled-pressure scripts for performance claims.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects like `Fix ...`, `Optimize ...`, and `Reduce ...`. For LayerKV PRs, describe the affected path: planner, KVC arena, expert backing, scheduler, server args, or evaluation script. Include exact commands run, key stats or CSV outputs, and hardware/model assumptions. Avoid unrelated upstream refactors, and run `pre-commit run --all-files` plus relevant LayerKV validation scripts before review.
