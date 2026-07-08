#!/usr/bin/env python3
"""Run Qwen-Bailian trace-replayer against a real SGLang PD backend."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import request

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def free_port(preferred: int, used: set[int]) -> int:
    if preferred and preferred not in used and is_free(preferred):
        used.add(preferred)
        return preferred
    for port in range(30000, 50000):
        if port not in used and is_free(port):
            used.add(port)
            return port
    raise RuntimeError("no free port")


def is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_health(url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited before ready: {url}")
        try:
            with request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"not ready after {timeout_s}s: {url}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    os.killpg(proc.pid, signal.SIGKILL)


def popen(cmd: list[str], log: Path, env: dict[str, str]) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("w")
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def base_env(gpu: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCX")
    env.setdefault("SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}")
    env.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,tcp")
    env.setdefault("SGLANG_MAMBA_CONV_DTYPE", "float16")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def server_cmd(args: argparse.Namespace, mode: str, port: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.model_path,
        "--served-model-name",
        args.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disaggregation-mode",
        mode,
        "--disaggregation-transfer-backend",
        "nixl",
        "--dtype",
        "float16",
        "--mamba-ssm-dtype",
        "float16",
        "--trust-remote-code",
        "--enable-metrics",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
        "--log-level",
        args.log_level,
    ]
    if mode == "decode":
        cmd += [
            "--decode-prealloc-reclaim-policy",
            args.decode_prealloc_reclaim_policy,
            "--expert-residency-budget-ratio",
            str(args.expert_residency_budget_ratio),
        ]
        if args.decode_prealloc_reclaim_dry_run:
            cmd.append("--decode-prealloc-reclaim-dry-run")
        if not args.decode_prealloc_kv_reclaim_safe_only:
            cmd.append("--no-decode-prealloc-kv-reclaim-safe-only")
    if args.mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.max_running_requests is not None:
        cmd += ["--max-running-requests", str(args.max_running_requests)]
    if args.max_total_tokens is not None:
        cmd += ["--max-total-tokens", str(args.max_total_tokens)]
    return cmd


def router_cmd(
    args: argparse.Namespace, prefill_port: int, decode_port: int
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        f"http://127.0.0.1:{prefill_port}",
        "--decode",
        f"http://127.0.0.1:{decode_port}",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--prometheus-port",
        str(args.router_metrics_port),
        "--model-path",
        args.model_path,
    ]


def replayer_cmd(args: argparse.Namespace, rate: float, out_jsonl: Path) -> list[str]:
    return [
        args.replayer_bin,
        "--tokenizer",
        str(Path(args.model_path) / "tokenizer.json"),
        "--tokenizer-config",
        str(Path(args.model_path) / "tokenizer_config.json"),
        "--endpoint",
        f"http://127.0.0.1:{args.router_port}/v1/chat/completions",
        "--api",
        "openai",
        "--dataset",
        "bailian",
        "--dataset-path",
        args.trace_path,
        "--scale-factor",
        str(rate),
        "--time-in-secs",
        str(args.time_in_secs),
        "--num-producer",
        str(args.num_producer),
        "--channel-capacity",
        str(args.channel_capacity),
        "--output-path",
        str(out_jsonl),
        "--summary-path",
        str(out_jsonl.with_suffix(".summary.json")),
        "--model-name",
        args.model_name,
        "--stream",
    ]


def scrape_prom(url: str) -> dict[str, float]:
    try:
        text = request.urlopen(url, timeout=1).read().decode("utf-8", "replace")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if not name.startswith("sglang:"):
            continue
        if name.endswith("_bucket"):
            match = re.search(r'[,{}]le="([^"]+)"', line)
            if match:
                name = f"{name}_le_{metric_label_suffix(match.group(1))}"
        try:
            value = float(line.rsplit(" ", 1)[1])
        except Exception:
            continue
        out[name] = max(out.get(name, value), value)
    return out


def metric_label_suffix(value: str) -> str:
    return value.replace("+Inf", "inf").replace(".", "p").replace("-", "m")


def gpu_row() -> dict[str, float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        rows = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception:
        return {"gpu_memory_used_mb": 0.0, "gpu_utilization_pct": 0.0}
    mem, util = [], []
    for row in rows:
        a, b = [float(x.strip()) for x in row.split(",")]
        mem.append(a)
        util.append(b)
    return {
        "gpu_memory_used_mb": max(mem) if mem else 0.0,
        "gpu_utilization_pct": max(util) if util else 0.0,
    }


def monitor(
    stop: threading.Event, urls: dict[str, str], out_csv: Path, rate: float
) -> None:
    rows = []
    t0 = time.time()
    while not stop.is_set():
        row: dict[str, Any] = {"time": time.time() - t0, "replay_rate": rate}
        for prefix, url in urls.items():
            for k, v in scrape_prom(url).items():
                row[f"{prefix}_{k.replace('sglang:', '').replace(':', '_')}"] = v
        row.update(gpu_row())
        rows.append(row)
        time.sleep(1)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_decode_feasibility(log_path: Path) -> dict[str, Any]:
    out = {
        "feasibility_result": "unknown",
        "feasibility_reason": "",
        "resident_expert_bytes": 0,
        "reclaimable_expert_bytes": 0,
    }
    if not log_path.exists():
        return out
    pattern = re.compile(
        r"EXPERT_EVICTION_CAN_UNBLOCK_PREALLOC\s*=\s*(True|False|true|false); "
        r"reason=(.*?); .*resident_expert_bytes=(\d+); reclaimable_expert_bytes=(\d+)"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        out["feasibility_result"] = str(match.group(1)).lower()
        out["feasibility_reason"] = match.group(2)
        out["resident_expert_bytes"] = int(match.group(3))
        out["reclaimable_expert_bytes"] = int(match.group(4))
    return out


def summarize_run(
    rate: float,
    jsonl: Path,
    ts_csv: Path,
    policy: str = "none",
    expert_residency_budget_ratio: float = 1.0,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    rows = []
    if jsonl.exists():
        with jsonl.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    ts = pd.read_csv(ts_csv) if ts_csv.exists() else pd.DataFrame()
    ok = df[df.get("status", "") == "200"] if len(df) else df
    tpot = (
        pd.to_numeric(ok.get("total_time", pd.Series(dtype=float)), errors="coerce")
        / pd.to_numeric(
            ok.get("output_length", pd.Series(dtype=float)), errors="coerce"
        )
        if len(ok)
        else pd.Series(dtype=float)
    )
    output_tokens = pd.to_numeric(ok.get("output_length"), errors="coerce")
    total_time = pd.to_numeric(ok.get("total_time"), errors="coerce")
    prealloc_metric = "decode_decode_prealloc_wait_seconds"
    reclaim_metric = "decode_decode_prealloc_reclaim_elapsed_ms"
    feasibility = parse_decode_feasibility(run_dir / "decode.log") if run_dir else {}
    prealloc_waited = hist_count_above_threshold(ts, prealloc_metric, "0p001")
    queue_col = exact_col(ts, "decode_num_decode_prealloc_queue_reqs")
    token_col = exact_col(ts, "decode_full_token_usage")
    queue_nonzero = (
        pd.to_numeric(ts[queue_col], errors="coerce").fillna(0) > 0
        if queue_col
        else pd.Series(dtype=bool)
    )
    row = {
        "policy": policy,
        "expert_residency_budget_ratio": expert_residency_budget_ratio,
        "replay_rate": rate,
        "total_requests": len(df),
        "successful_requests": len(ok),
        "failed_requests": int((df.get("status", "") != "200").sum()) if len(df) else 0,
        "throughput_rps": throughput_rps(df),
        "throughput_req_s": throughput_rps(df),
        "throughput_tok_s": throughput_tps(df),
        "ttft_mean_ms": mean_series(ok.get("first_token_time")),
        "ttft_p50_ms": q(ok.get("first_token_time"), 0.50),
        "p50_ttft_ms": q(ok.get("first_token_time"), 0.50),
        "p95_ttft_ms": q(ok.get("first_token_time"), 0.95),
        "p99_ttft_ms": q(ok.get("first_token_time"), 0.99),
        "ttft_p95_ms": q(ok.get("first_token_time"), 0.95),
        "ttft_p99_ms": q(ok.get("first_token_time"), 0.99),
        "tpot_mean_ms": mean_series(tpot),
        "tpot_p50_ms": q(tpot, 0.50),
        "p50_tpot_ms": q(tpot, 0.50),
        "p95_tpot_ms": q(tpot, 0.95),
        "p99_tpot_ms": q(tpot, 0.99),
        "tpot_p95_ms": q(tpot, 0.95),
        "tpot_p99_ms": q(tpot, 0.99),
        "e2e_mean_s": mean_series(total_time) / 1000,
        "e2e_p95_s": q(total_time, 0.95) / 1000,
        "e2e_p99_s": q(total_time, 0.99) / 1000,
        "p95_timing_drift_ms": q(df.get("s_time_drift"), 0.95),
        "avg_waiting_requests": mean_matching(ts, "num_queue_reqs"),
        "max_waiting_requests": max_matching(ts, "num_queue_reqs"),
        "decode_ordinary_queue_max": max_exact(ts, "decode_num_queue_reqs"),
        "decode_retracted": max_exact(ts, "decode_num_retracted_reqs"),
        "decode_prealloc_queue_max": max_exact(
            ts, "decode_num_decode_prealloc_queue_reqs"
        ),
        "prealloc_waited_requests": prealloc_waited,
        "prealloc_waited_request_ratio": (
            prealloc_waited / len(df) if len(df) else 0.0
        ),
        "decode_prealloc_waited_requests": prealloc_waited,
        "decode_prealloc_waited_ratio": (prealloc_waited / len(df) if len(df) else 0.0),
        "prealloc_wait_time_p50_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.50, "0p001"
        ),
        "prealloc_wait_time_p95_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.95, "0p001"
        ),
        "prealloc_wait_time_p99_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.99, "0p001"
        ),
        "decode_prealloc_wait_p50_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.50, "0p001"
        ),
        "decode_prealloc_wait_p95_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.95, "0p001"
        ),
        "decode_prealloc_wait_p99_ms": hist_quantile_ms(
            ts, prealloc_metric, 0.99, "0p001"
        ),
        "prealloc_queue_nonzero_seconds": queue_nonzero_seconds(ts, queue_nonzero),
        "decode_prealloc_queue_nonzero_duration_s": queue_nonzero_seconds(
            ts, queue_nonzero
        ),
        "fraction_time_prealloc_queue_nonzero": (
            float(queue_nonzero.mean()) if len(queue_nonzero) else 0.0
        ),
        "decode_prealloc_queue_nonzero_fraction": (
            float(queue_nonzero.mean()) if len(queue_nonzero) else 0.0
        ),
        "prealloc_queue_full_token_usage_corr": corr_cols(ts, queue_col, token_col),
        "decode_full_token_usage_max": max_exact(ts, "decode_full_token_usage"),
        "decode_full_token_usage_p95": q(ts.get("decode_full_token_usage"), 0.95),
        "decode_full_token_usage_mean_when_prealloc_queue_nonzero": mean_when(
            ts, "decode_full_token_usage", queue_nonzero
        ),
        "reclaim_attempts": counter_value(
            ts, "decode_decode_prealloc_reclaim_attempts_total"
        ),
        "reclaim_successes": counter_value(
            ts, "decode_decode_prealloc_reclaim_successes_total"
        ),
        "reclaim_kv_tokens": counter_value(
            ts, "decode_decode_prealloc_reclaim_kv_tokens_total"
        ),
        "reclaim_kv_bytes": counter_value(
            ts, "decode_decode_prealloc_reclaim_kv_bytes_total"
        ),
        "reclaim_expert_bytes": counter_value(
            ts, "decode_decode_prealloc_reclaim_expert_bytes_total"
        ),
        "reclaim_elapsed_ms_p95": hist_quantile_ms(ts, reclaim_metric, 0.95),
        "prealloc_retry_successes": counter_value(
            ts, "decode_decode_prealloc_retry_successes_total"
        ),
        "prealloc_fail_due_to_full_token_pool": counter_value(
            ts, "decode_prealloc_fail_due_to_full_token_pool_total"
        ),
        "prealloc_fail_due_to_req_pool": counter_value(
            ts, "decode_prealloc_fail_due_to_req_pool_total"
        ),
        "prealloc_fail_due_to_metadata": counter_value(
            ts, "decode_prealloc_fail_due_to_metadata_total"
        ),
        "prealloc_fail_due_to_mamba": counter_value(
            ts, "decode_prealloc_fail_due_to_mamba_total"
        ),
        "prealloc_fail_due_to_other": counter_value(
            ts, "decode_prealloc_fail_due_to_other_total"
        ),
        "prealloc_fail_due_to_swa_token_pool": counter_value(
            ts, "decode_prealloc_fail_due_to_swa_token_pool_total"
        ),
        "token_pool_capacity": max_exact(ts, "decode_max_total_num_tokens"),
        "resident_expert_bytes": feasibility.get("resident_expert_bytes", 0),
        "reclaimable_expert_bytes": feasibility.get("reclaimable_expert_bytes", 0),
        "expert_eviction_attempts": (
            counter_value(ts, "decode_decode_prealloc_reclaim_attempts_total")
            if policy in ("expert_lru", "joint_simple")
            else 0
        ),
        "expert_eviction_successes": (
            counter_value(ts, "decode_decode_prealloc_reclaim_successes_total")
            if policy in ("expert_lru", "joint_simple")
            else 0
        ),
        "evicted_expert_bytes": counter_value(
            ts, "decode_decode_prealloc_reclaim_expert_bytes_total"
        ),
        "expert_reload_count": 0,
        "expert_reload_stall_ms_p95": 0,
        "prealloc_retry_successes_after_expert_eviction": (
            counter_value(ts, "decode_decode_prealloc_retry_successes_total")
            if policy in ("expert_lru", "joint_simple")
            else 0
        ),
        "dryrun_expert_eviction_possible_ratio": 0,
        "feasibility_result": feasibility.get("feasibility_result", "unknown"),
        "feasibility_reason": feasibility.get("feasibility_reason", ""),
        "avg_kv_token_usage": mean_matching(ts, "token_usage"),
        "max_kv_token_usage": max_matching(ts, "token_usage"),
        "avg_gpu_memory_mb": (
            float(ts.get("gpu_memory_used_mb", pd.Series(dtype=float)).mean())
            if len(ts)
            else 0.0
        ),
        "max_gpu_memory_mb": (
            float(ts.get("gpu_memory_used_mb", pd.Series(dtype=float)).max())
            if len(ts)
            else 0.0
        ),
        "avg_gpu_utilization_pct": (
            float(ts.get("gpu_utilization_pct", pd.Series(dtype=float)).mean())
            if len(ts)
            else 0.0
        ),
        "max_gpu_utilization_pct": (
            float(ts.get("gpu_utilization_pct", pd.Series(dtype=float)).max())
            if len(ts)
            else 0.0
        ),
    }
    return row


def q(values: Any, quantile: float) -> float:
    s = (
        pd.to_numeric(values, errors="coerce").dropna()
        if values is not None
        else pd.Series()
    )
    return float(s.quantile(quantile)) if len(s) else 0.0


def mean_series(values: Any) -> float:
    s = (
        pd.to_numeric(values, errors="coerce").dropna()
        if values is not None
        else pd.Series()
    )
    return float(s.mean()) if len(s) else 0.0


def throughput_rps(df: pd.DataFrame) -> float:
    if not len(df) or not {"e_time", "s_time"}.issubset(df):
        return 0.0
    e_time = pd.to_numeric(df["e_time"], errors="coerce")
    s_time = pd.to_numeric(df["s_time"], errors="coerce")
    duration_s = (e_time.max() - s_time.min()) / 1000
    return float(len(df) / max(1e-9, duration_s))


def throughput_tps(df: pd.DataFrame) -> float:
    if not len(df) or not {"e_time", "s_time"}.issubset(df):
        return 0.0
    e_time = pd.to_numeric(df["e_time"], errors="coerce")
    s_time = pd.to_numeric(df["s_time"], errors="coerce")
    duration_s = (e_time.max() - s_time.min()) / 1000
    output_tokens = pd.to_numeric(df.get("output_length"), errors="coerce").sum()
    return float(output_tokens / max(1e-9, duration_s))


def mean_matching(df: pd.DataFrame, needle: str) -> float:
    cols = [c for c in df.columns if needle in c]
    return float(df[cols].max(axis=1).mean()) if cols and len(df) else 0.0


def max_matching(df: pd.DataFrame, needle: str) -> float:
    cols = [c for c in df.columns if needle in c]
    return float(df[cols].max(axis=1).max()) if cols and len(df) else 0.0


def max_exact(df: pd.DataFrame, col: str) -> float:
    return (
        float(pd.to_numeric(df[col], errors="coerce").max())
        if col in df.columns and len(df)
        else 0.0
    )


def counter_value(df: pd.DataFrame, col: str) -> float:
    return max_exact(df, col)


def mean_when(df: pd.DataFrame, col: str, mask: pd.Series) -> float:
    if col not in df.columns or not len(df) or not len(mask):
        return 0.0
    values = pd.to_numeric(df[col], errors="coerce")
    selected = values[mask.reindex(values.index, fill_value=False)]
    return float(selected.mean()) if len(selected.dropna()) else 0.0


def exact_col(df: pd.DataFrame, name: str) -> str | None:
    return name if name in df.columns else None


def hist_count(df: pd.DataFrame, metric: str) -> int:
    col = f"{metric}_count"
    if col not in df.columns or not len(df):
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").max())


def hist_count_above_threshold(
    df: pd.DataFrame, metric: str, threshold_suffix: str
) -> int:
    total = hist_count(df, metric)
    threshold_col = f"{metric}_bucket_le_{threshold_suffix}"
    if threshold_col not in df.columns or not len(df):
        return total
    below = pd.to_numeric(df[threshold_col], errors="coerce").max()
    return int(max(0, total - (0 if pd.isna(below) else below)))


def hist_quantile_ms(
    df: pd.DataFrame, metric: str, quantile: float, skip_le_suffix: str | None = None
) -> float:
    prefix = f"{metric}_bucket_le_"
    buckets = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        le = col.removeprefix(prefix).replace("p", ".").replace("m", "-")
        upper = float("inf") if le == "inf" else float(le)
        count = pd.to_numeric(df[col], errors="coerce").max()
        if pd.notna(count):
            buckets.append((upper, float(count)))
    if not buckets:
        return 0.0
    buckets.sort()
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    skipped = 0.0
    if skip_le_suffix:
        skip_col = f"{metric}_bucket_le_{skip_le_suffix}"
        if skip_col in df.columns:
            skipped = pd.to_numeric(df[skip_col], errors="coerce").max()
            skipped = 0.0 if pd.isna(skipped) else float(skipped)
    total -= skipped
    if total <= 0:
        return 0.0
    target = skipped + total * quantile
    prev_upper, prev_count = 0.0, 0.0
    for upper, count in buckets:
        if count >= target:
            if upper == float("inf"):
                return prev_upper * 1000
            if count == prev_count:
                return upper * 1000
            frac = (target - prev_count) / (count - prev_count)
            return (prev_upper + frac * (upper - prev_upper)) * 1000
        prev_upper, prev_count = upper, count
    return buckets[-1][0] * 1000


def queue_nonzero_seconds(ts: pd.DataFrame, mask: pd.Series) -> float:
    if "time" not in ts or len(ts) < 2 or not len(mask):
        return 0.0
    dt = pd.to_numeric(ts["time"], errors="coerce").diff().median()
    return float(mask.sum() * dt) if pd.notna(dt) else 0.0


def corr_cols(df: pd.DataFrame, a: str | None, b: str | None) -> float:
    if not a or not b or not len(df):
        return 0.0
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    corr = x.corr(y)
    return float(corr) if pd.notna(corr) else 0.0


def make_plots(summary: pd.DataFrame, timeseries: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.12))
    waiting_cols = [c for c in timeseries.columns if "num_queue_reqs" in c]
    token_cols = [c for c in timeseries.columns if "token_usage" in c]
    if waiting_cols:
        ax1.plot(
            timeseries["time"], timeseries[waiting_cols].max(axis=1), label="waiting"
        )
    if token_cols:
        ax2.plot(
            timeseries["time"],
            timeseries[token_cols].max(axis=1),
            color="tab:orange",
            label="KV/token usage",
        )
    if "gpu_utilization_pct" in timeseries:
        ax3.plot(
            timeseries["time"],
            timeseries["gpu_utilization_pct"],
            color="tab:green",
            label="GPU util",
        )
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("waiting requests")
    ax2.set_ylabel("KV/token usage")
    ax3.set_ylabel("GPU utilization (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_runtime_pressure_over_time.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        summary["replay_rate"], summary["p95_ttft_ms"], marker="o", label="p95 TTFT"
    )
    ax.plot(
        summary["replay_rate"], summary["p99_ttft_ms"], marker="o", label="p99 TTFT"
    )
    ax.set_xlabel("replay rate")
    ax.set_ylabel("TTFT (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ttft_vs_replay_rate.pdf")
    plt.close(fig)

    if {
        "policy",
        "decode_prealloc_wait_p95_ms",
        "decode_prealloc_waited_ratio",
    }.issubset(summary.columns):
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        x = range(len(summary))
        ax1.bar(
            [v - 0.2 for v in x],
            summary["decode_prealloc_wait_p95_ms"],
            width=0.4,
            label="p95 wait ms",
        )
        ax2.bar(
            [v + 0.2 for v in x],
            summary["decode_prealloc_waited_ratio"],
            width=0.4,
            color="tab:orange",
            label="waited ratio",
        )
        ax1.set_xticks(list(x), summary["policy"])
        ax1.set_ylabel("p95 prealloc wait (ms)")
        ax2.set_ylabel("waited request ratio")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_prealloc_wait_by_policy.pdf")
        plt.close(fig)

    if {
        "expert_residency_budget_ratio",
        "decode_prealloc_wait_p95_ms",
        "decode_prealloc_waited_ratio",
    }.issubset(summary.columns):
        sub = summary[summary["policy"] == "none"].sort_values(
            "expert_residency_budget_ratio"
        )
        if len(sub):
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax2 = ax1.twinx()
            ax1.plot(
                sub["expert_residency_budget_ratio"],
                sub["decode_prealloc_wait_p95_ms"],
                marker="o",
                label="p95 wait ms",
            )
            ax2.plot(
                sub["expert_residency_budget_ratio"],
                sub["decode_prealloc_waited_ratio"],
                marker="o",
                color="tab:orange",
                label="waited ratio",
            )
            ax1.set_xlabel("expert residency budget ratio")
            ax1.set_ylabel("p95 prealloc wait (ms)")
            ax2.set_ylabel("waited request ratio")
            fig.tight_layout()
            fig.savefig(out_dir / "fig_static_expert_budget_prealloc_wait.pdf")
            plt.close(fig)

    if {
        "policy",
        "time",
        "decode_num_decode_prealloc_queue_reqs",
        "decode_full_token_usage",
    }.issubset(timeseries.columns):
        policies = list(dict.fromkeys(timeseries["policy"].dropna()))
        fig, axes = plt.subplots(
            len(policies), 1, figsize=(10, max(3, 2.5 * len(policies))), squeeze=False
        )
        for ax, policy in zip(axes[:, 0], policies):
            sub = timeseries[timeseries["policy"] == policy]
            ax2 = ax.twinx()
            ax.plot(
                sub["time"],
                sub["decode_num_decode_prealloc_queue_reqs"],
                label="prealloc queue",
            )
            ax2.plot(
                sub["time"],
                sub["decode_full_token_usage"],
                color="tab:orange",
                label="full token usage",
            )
            ax.set_title(policy)
            ax.set_ylabel("queue")
            ax2.set_ylabel("usage")
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_prealloc_queue_token_usage_timeseries.pdf")
        plt.close(fig)


def run_one(
    args: argparse.Namespace,
    rate: float,
    policy: str,
    expert_residency_budget_ratio: float,
    out_dir: Path,
) -> dict[str, Any]:
    used: set[int] = set()
    prefill_port = free_port(args.prefill_port, used)
    decode_port = free_port(args.decode_port, used)
    args.router_port = free_port(args.router_port, used)
    args.router_metrics_port = free_port(args.router_metrics_port, used)

    args.decode_prealloc_reclaim_policy = policy
    args.expert_residency_budget_ratio = expert_residency_budget_ratio
    ratio_tag = f"{expert_residency_budget_ratio:g}".replace(".", "p")
    run_dir = out_dir / f"policy_{policy}_expert_{ratio_tag}_rate_{rate:g}x"
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_path = run_dir / "commands.json"
    procs: list[subprocess.Popen] = []
    stop = threading.Event()
    mon = None
    try:
        prefill_cmd = server_cmd(args, "prefill", prefill_port)
        decode_cmd = server_cmd(args, "decode", decode_port)
        router = router_cmd(args, prefill_port, decode_port)
        commands_path.write_text(
            json.dumps(
                {"prefill": prefill_cmd, "decode": decode_cmd, "router": router},
                indent=2,
            )
        )
        procs.append(
            popen(prefill_cmd, run_dir / "prefill.log", base_env(args.prefill_gpu))
        )
        wait_health(
            f"http://127.0.0.1:{prefill_port}/health", procs[-1], args.startup_timeout_s
        )
        procs.append(
            popen(decode_cmd, run_dir / "decode.log", base_env(args.decode_gpu))
        )
        wait_health(
            f"http://127.0.0.1:{decode_port}/health", procs[-1], args.startup_timeout_s
        )
        procs.append(popen(router, run_dir / "router.log", base_env()))
        wait_health(
            f"http://127.0.0.1:{args.router_port}/health",
            procs[-1],
            args.startup_timeout_s,
        )

        ts_csv = run_dir / "timeseries.csv"
        urls = {
            "prefill": f"http://127.0.0.1:{prefill_port}/metrics",
            "decode": f"http://127.0.0.1:{decode_port}/metrics",
            "router": f"http://127.0.0.1:{args.router_metrics_port}/metrics",
        }
        mon = threading.Thread(
            target=monitor, args=(stop, urls, ts_csv, rate), daemon=True
        )
        mon.start()
        out_jsonl = run_dir / "replayer.jsonl"
        cmd = replayer_cmd(args, rate, out_jsonl)
        commands = json.loads(commands_path.read_text())
        commands["replayer"] = cmd
        commands_path.write_text(json.dumps(commands, indent=2))
        with (run_dir / "replayer.log").open("w") as log:
            subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        stop.set()
        if mon is not None:
            mon.join(timeout=5)
        return summarize_run(
            rate,
            out_jsonl,
            ts_csv,
            policy,
            expert_residency_budget_ratio,
            run_dir,
        )
    finally:
        stop.set()
        if mon is not None:
            mon.join(timeout=5)
        for proc in reversed(procs):
            terminate(proc)


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    rows = [
        run_one(args, rate, policy, ratio, out_dir)
        for policy in args.decode_prealloc_reclaim_policies
        for ratio in args.expert_residency_budget_ratios
        for rate in args.rates
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    all_ts = []
    for policy in args.decode_prealloc_reclaim_policies:
        for ratio in args.expert_residency_budget_ratios:
            ratio_tag = f"{ratio:g}".replace(".", "p")
            for rate in args.rates:
                path = (
                    out_dir
                    / f"policy_{policy}_expert_{ratio_tag}_rate_{rate:g}x"
                    / "timeseries.csv"
                )
                if path.exists():
                    df = pd.read_csv(path)
                    df["policy"] = policy
                    df["expert_residency_budget_ratio"] = ratio
                    all_ts.append(df)
    timeseries = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()
    timeseries.to_csv(out_dir / "timeseries.csv", index=False)
    if len(summary) and len(timeseries):
        make_plots(summary, timeseries, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="qwen3.6-35b-a3b-fp16")
    parser.add_argument(
        "--trace-path", default=str(ROOT / "dataset/qwen_traceA_blksz_16.jsonl")
    )
    parser.add_argument(
        "--replayer-bin", default="/sgl-workspace/trace-replayer/target/release/client"
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results/bailian_pd_trace_replay")
    )
    parser.add_argument("--time-in-secs", type=int, default=600)
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument(
        "--decode-prealloc-reclaim-policies",
        nargs="+",
        choices=["none", "kv_lru", "expert_lru", "joint_simple"],
        default=["none"],
    )
    parser.add_argument(
        "--expert-residency-budget-ratios",
        type=float,
        nargs="+",
        choices=[1.0, 0.75, 0.5, 0.25],
        default=[1.0],
    )
    parser.add_argument(
        "--expert-residency-budget-ratio",
        type=float,
        choices=[1.0, 0.75, 0.5, 0.25],
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--decode-prealloc-reclaim-policy",
        choices=["none", "kv_lru", "expert_lru", "joint_simple"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--decode-prealloc-reclaim-dry-run", action="store_true")
    parser.add_argument(
        "--decode-prealloc-kv-reclaim-safe-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-producer", type=int, default=32)
    parser.add_argument("--channel-capacity", type=int, default=40960)
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--router-metrics-port", type=int, default=26700)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--max-running-requests", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
