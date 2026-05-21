#!/usr/bin/env python3
"""Shared helpers for LayerKV SGLang evaluation scripts."""

from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List


DEFAULT_MODEL_PATH = (
    "/data/wenyan/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B/"
    "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)


def parse_layerkv_stats(text: str) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    marker = "LayerKV stats after "
    for line in text.splitlines():
        if marker not in line:
            continue
        try:
            payload = line.split(": ", 1)[1]
            parsed = ast.literal_eval(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            stats.append(parsed)
    return stats


def parse_benchmark_latencies(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    match = re.search(r"Benchmark .*?Prefill\. latency:\s*([0-9.]+) s", text, re.S)
    if match:
        out["benchmark_prefill_latency_s"] = float(match.group(1))
    match = re.search(
        r"Benchmark .*?Decode 0\. Batch size: \d+, latency:\s*([0-9.]+) s",
        text,
        re.S,
    )
    if match:
        out["benchmark_decode0_latency_s"] = float(match.group(1))
    return out


def base_command(args: Any, result_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        args.model_path,
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--result-filename",
        str(result_path),
        "--log-level",
        args.log_level,
    ]


def layerkv_flags(
    *,
    mode: str,
    policy: str,
    target_reclaim_mb: float,
    kvc_block_tokens: int,
    scheduler: str = "async-deadline",
    debug_stats: bool = True,
) -> List[str]:
    flags = [
        "--enable-layerkv",
        "--layerkv-mode",
        mode,
        "--layerkv-policy",
        policy,
        "--layerkv-target-reclaim-mb",
        str(target_reclaim_mb),
        "--layerkv-kvc-block-tokens",
        str(kvc_block_tokens),
        "--layerkv-kvc-scheduler",
        scheduler,
    ]
    if debug_stats:
        flags.append("--layerkv-debug-stats")
    return flags


def run_bench_command(
    *,
    cmd: List[str],
    output_dir: Path,
    run_name: str,
    args: Any,
) -> Dict[str, Any]:
    stdout_path = output_dir / f"{run_name}.stdout.log"
    stderr_path = output_dir / f"{run_name}.stderr.log"
    for path in (stdout_path, stderr_path):
        path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TMPDIR"] = args.tmpdir
    repo_python = str(Path.cwd() / "python")
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else repo_python + os.pathsep + env["PYTHONPATH"]
    )

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    text = proc.stdout + "\n" + proc.stderr
    stats = parse_layerkv_stats(text)
    return {
        "returncode": proc.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stats": stats,
        "final_stats": stats[-1] if stats else {},
        "latencies": parse_benchmark_latencies(text),
        "stats_line_count": len(stats),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
