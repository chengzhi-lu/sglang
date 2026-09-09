#!/usr/bin/env python3
"""Local LayerKV launcher with stable repository, venv, and HF-cache paths.

Examples:

  python scripts/layerkv_local.py
  python scripts/layerkv_local.py prefetch-test
  python scripts/layerkv_local.py concurrent-perf --output-dir /mnt/vdb/chengzhi/run
  python scripts/layerkv_local.py --model-revision REV concurrent-perf --output-dir /mnt/vdb/chengzhi/run
  python scripts/layerkv_local.py --config scripts/layerkv_local_config.json paths

The launcher uses the configured venv Python for child scripts and injects the
resolved local model path for workflows that accept ``--model-path``.  It also
exports the explicitly selected source and Hugging Face cache paths to child
processes so nested SGLang commands use the same installation and cache.
All experiment settings remain arguments to the child script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE_ROOT = REPO_ROOT / "python"
DEFAULT_VENV_PYTHON = Path("/mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python")
DEFAULT_HF_HUB_ROOT = Path("/mnt/vdb/hf_home/hub")
DEFAULT_MODEL_CACHE_NAME = "models--Qwen--Qwen3.6-35B-A3B"
DEFAULT_PATH_CONFIG = REPO_ROOT / "scripts" / "layerkv_local_config.json"

PATH_CONFIG_KEYS = {
    "venv_python",
    "python_library_root",
    "hub_root",
    "model_cache_name",
    "model_revision",
    "model_path",
}

TARGETS = {
    "unit-tests": None,
    "gpu-group-bench": REPO_ROOT / "scripts" / "layerkv_gpu_group_bench.py",
    "prefetch-test": REPO_ROOT / "scripts" / "layerkv_prefetch_test.py",
    "concurrent-perf": REPO_ROOT / "scripts" / "layerkv_concurrent_perf.py",
    "hybrid-validation": REPO_ROOT / "scripts" / "layerkv_hybrid_validation.py",
    "shared-perf": REPO_ROOT / "scripts" / "layerkv_shared_perf.py",
    "server-smoke": REPO_ROOT / "scripts" / "layerkv_server_smoke.py",
    "policy-eval": REPO_ROOT / "scripts" / "layerkv_policy_eval.py",
    "real-validation": REPO_ROOT / "scripts" / "layerkv_real_validation.py",
    "expert-validation": REPO_ROOT / "scripts" / "layerkv_expert_validation.py",
    "reference-compare": REPO_ROOT / "scripts" / "layerkv_reference_compare.py",
}

# These scripts either require a real model or have a model-backed mode.  The
# KVC semantic validator is deliberately excluded: it defaults to a tiny
# dummy-weight model and should not accidentally start the 35B model.
MODEL_PATH_TARGETS = {
    "concurrent-perf",
    "hybrid-validation",
    "shared-perf",
    "server-smoke",
    "policy-eval",
    "real-validation",
}


def _load_path_config(config_path: Path, explicit: bool) -> dict:
    """Load persistent path defaults without making experiment settings implicit."""
    if not config_path.is_file():
        if explicit:
            raise FileNotFoundError(f"path config does not exist: {config_path}")
        return {}
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read path config {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"path config must contain a JSON object: {config_path}")
    unknown = sorted(set(payload) - PATH_CONFIG_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unsupported path config keys in {config_path}: {names}")
    return payload


def _path_config_value(
    args: argparse.Namespace, config: dict, name: str, builtin_default
):
    value = getattr(args, name)
    return builtin_default if value is None and name not in config else (
        config.get(name) if value is None else value
    )


def _apply_path_config(args: argparse.Namespace, config: dict) -> None:
    args.venv_python = _path_config_value(
        args, config, "venv_python", str(DEFAULT_VENV_PYTHON)
    )
    args.python_library_root = _path_config_value(
        args, config, "python_library_root", str(PYTHON_SOURCE_ROOT)
    )
    args.hub_root = _path_config_value(
        args, config, "hub_root", str(DEFAULT_HF_HUB_ROOT)
    )
    args.model_cache_name = _path_config_value(
        args, config, "model_cache_name", DEFAULT_MODEL_CACHE_NAME
    )
    args.model_revision = _path_config_value(args, config, "model_revision", None)
    args.model_path = _path_config_value(args, config, "model_path", None)


def _snapshot_from_cache(
    hub_root: Path, cache_name: str, revision: Optional[str]
) -> Path:
    cache_root = hub_root / cache_name
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        raise FileNotFoundError(f"model snapshots directory does not exist: {snapshots_root}")
    if revision:
        snapshot = snapshots_root / revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"model snapshot does not exist: {snapshot}")
        return snapshot

    main_ref = cache_root / "refs" / "main"
    if main_ref.is_file():
        snapshot = (snapshots_root / main_ref.read_text().strip()).resolve()
        if snapshot.is_dir():
            return snapshot

    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    if len(snapshots) == 1:
        return snapshots[0]
    if not snapshots:
        raise FileNotFoundError(f"no model snapshots found under: {snapshots_root}")
    names = ", ".join(path.name for path in snapshots)
    raise RuntimeError(
        "multiple model snapshots found; pass --model-revision explicitly: " + names
    )


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        path = Path(args.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"model path does not exist: {path}")
        return path.resolve()
    return _snapshot_from_cache(
        Path(args.hub_root).expanduser(), args.model_cache_name, args.model_revision
    )


def _has_model_path(child_args: List[str]) -> bool:
    return any(
        arg == "--model-path" or arg.startswith("--model-path=")
        for arg in child_args
    )


def _normalize_unit_test_args(
    child_args: List[str], child_cwd: Path
) -> List[str]:
    """Make repository-relative pytest paths independent of child cwd."""
    normalized = []
    for arg in child_args:
        if arg.startswith("-") or Path(arg).is_absolute():
            normalized.append(arg)
            continue
        path_text, separator, node_id = arg.partition("::")
        candidates = (
            child_cwd / path_text,
            REPO_ROOT / path_text,
            Path.cwd() / path_text,
        )
        path = next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            None,
        )
        if path is None:
            normalized.append(arg)
        else:
            normalized.append(
                str(path) + (separator + node_id if separator else "")
            )
    return normalized


def _has_unit_test_path(child_args: List[str], child_cwd: Path) -> bool:
    for arg in child_args:
        if arg.startswith("-") or Path(arg).is_absolute():
            continue
        path_text = arg.partition("::")[0]
        if any(
            candidate.exists()
            for candidate in (
                child_cwd / path_text,
                REPO_ROOT / path_text,
                Path.cwd() / path_text,
            )
        ):
            return True
    return False


def _print_paths(args: argparse.Namespace) -> int:
    model_path = None
    model_error = None
    try:
        model_path = str(resolve_model_path(args))
    except (FileNotFoundError, RuntimeError) as error:
        model_error = str(error)
    payload = {
        "repo_root": str(REPO_ROOT),
        "path_config": str(args.path_config),
        "python_source_root": str(Path(args.python_library_root).expanduser().resolve()),
        "python_library_root": str(Path(args.python_library_root).expanduser().resolve()),
        "venv_python": str(Path(args.venv_python).expanduser()),
        "hf_hub_root": str(Path(args.hub_root).expanduser()),
        "hf_home": str(Path(args.hub_root).expanduser().parent),
        "model_cache_name": args.model_cache_name,
        "model_path": model_path,
        "model_error": model_error,
    }
    print(json.dumps(payload, indent=2))
    return 0 if model_error is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="path_config",
        default=str(DEFAULT_PATH_CONFIG),
        help=(
            "JSON file containing persistent path defaults; command-line path "
            "options override it"
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=["paths", *TARGETS],
        default="paths",
        help="child workflow to run; default prints resolved paths",
    )
    parser.add_argument(
        "child_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the selected child script",
    )
    parser.add_argument(
        "--venv-python",
        default=None,
        help="Python executable used for child scripts",
    )
    parser.add_argument(
        "--python-library-root",
        "--python-source-root",
        dest="python_library_root",
        default=None,
        help="SGLang Python library/source root made importable for child processes",
    )
    parser.add_argument(
        "--hub-root",
        default=None,
        help="HuggingFace hub cache root",
    )
    parser.add_argument(
        "--model-cache-name",
        default=None,
        help="directory name under --hub-root",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="snapshot revision; required when the cache has multiple snapshots",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="explicit model directory; overrides --hub-root and cache resolution",
    )
    args = parser.parse_args()

    path_config = Path(args.path_config).expanduser()
    if not path_config.is_absolute():
        path_config = REPO_ROOT / path_config
    args.path_config = path_config.resolve()
    try:
        config = _load_path_config(args.path_config, explicit=args.path_config != DEFAULT_PATH_CONFIG)
        _apply_path_config(args, config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.target == "paths":
        if args.child_args:
            parser.error("paths does not accept child arguments")
        return _print_paths(args)

    if args.target == "unit-tests":
        child = None
    else:
        child = TARGETS[args.target]
    if child is not None and not child.is_file():
        print(f"missing child script: {child}", file=sys.stderr)
        return 2
    venv_python = Path(args.venv_python).expanduser()
    if not venv_python.is_file():
        print(f"venv Python does not exist: {venv_python}", file=sys.stderr)
        return 2
    python_library_root = Path(args.python_library_root).expanduser().resolve()
    if not python_library_root.is_dir():
        print(
            f"Python library/source root does not exist: {python_library_root}",
            file=sys.stderr,
        )
        return 2
    hub_root = Path(args.hub_root).expanduser().resolve()

    child_args = list(args.child_args)
    if child_args and child_args[0] == "--":
        child_args.pop(0)
    if args.target == "unit-tests" and not _has_unit_test_path(
        child_args, python_library_root
    ):
        child_args = ["../test/registered/unit/layerkv", *child_args]
    if args.target == "unit-tests":
        child_args = _normalize_unit_test_args(child_args, python_library_root)
    if args.target in MODEL_PATH_TARGETS and not _has_model_path(child_args):
        child_args = ["--model-path", str(resolve_model_path(args)), *child_args]

    command = (
        [str(venv_python), "-m", "pytest", *child_args]
        if args.target == "unit-tests"
        else [str(venv_python), str(child), *child_args]
    )
    child_cwd = python_library_root if args.target == "unit-tests" else REPO_ROOT
    print(
        json.dumps(
            {
                "repo_root": str(REPO_ROOT),
                "python_source_root": str(python_library_root),
                "python_library_root": str(python_library_root),
                "venv_python": str(venv_python),
                "hf_hub_root": str(hub_root),
                "hf_home": str(hub_root.parent),
                "child_cwd": str(child_cwd),
                "command": command,
            },
            indent=2,
        ),
        flush=True,
    )

    # PATH is only extended for external build tools such as ninja.  Runtime
    # experiment settings are forwarded as explicit child arguments.
    child_env = os.environ.copy()
    venv_bin = str(venv_python.parent)
    child_env["PATH"] = venv_bin + os.pathsep + child_env.get("PATH", "")
    source_root = str(python_library_root)
    child_env["PYTHONPATH"] = (
        source_root
        if not child_env.get("PYTHONPATH")
        else source_root + os.pathsep + child_env["PYTHONPATH"]
    )
    # These are Hugging Face framework cache locations, not experiment knobs.
    # They are derived from the explicit --hub-root argument so nested child
    # processes do not silently select a different cache.
    child_env["HF_HOME"] = str(hub_root.parent)
    child_env["HUGGINGFACE_HUB_CACHE"] = str(hub_root)
    return subprocess.run(command, cwd=child_cwd, env=child_env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
