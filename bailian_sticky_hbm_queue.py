#!/usr/bin/env python3
"""Trace-driven sticky-session HBM admission simulator for Qwen-Bailian traces."""

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import logging
import math
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

HBM_QUEUE = "HBM_ADMISSION_LIMIT"
COMPUTE_QUEUE = "COMPUTE_BATCH_LIMIT"
GB = 1024**3


@dataclass
class Request:
    idx: int
    chat_id: Any
    parent_chat_id: Any
    session_id: Any
    turn: int
    timestamp: float
    input_length: int
    output_length: int
    kv_gb: float
    assigned_worker: int | None = None
    arrival_time: float = 0.0
    admission_time: float | None = None
    finish_time: float | None = None
    was_queued: bool = False
    queue_reason: str = ""
    hbm_used_at_arrival_gb: float = 0.0
    resident_kv_at_arrival_gb: float = 0.0
    active_sessions_at_arrival: int = 0
    paused_sessions_at_arrival: int = 0
    compute_load_proxy_at_arrival: float = 0.0


@dataclass
class Worker:
    worker_id: int
    base_hbm_gb: float
    max_active_sessions: int
    active: dict[int, Request] = field(default_factory=dict)
    queue: deque[Request] = field(default_factory=deque)
    resident_kv_by_session: dict[Any, float] = field(default_factory=dict)
    active_sessions: set[Any] = field(default_factory=set)
    idle_generation_by_session: dict[Any, int] = field(default_factory=dict)

    @property
    def resident_kv_gb(self) -> float:
        return sum(self.resident_kv_by_session.values())

    @property
    def hbm_used_gb(self) -> float:
        return self.base_hbm_gb + self.resident_kv_gb

    @property
    def paused_sessions(self) -> int:
        return len(set(self.resident_kv_by_session) - self.active_sessions)

    @property
    def compute_load_proxy(self) -> float:
        return len(self.active) / self.max_active_sessions


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _get(record: dict[str, Any], key: str, default: Any, line_no: int) -> Any:
    if key not in record or record[key] is None:
        logging.warning("line %d missing %s; using %r", line_no, key, default)
        return default
    return record[key]


def load_records(path: Path, cfg: dict[str, Any]) -> list[Request]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("line %d invalid JSON; skipped: %s", line_no, exc)
                continue
            chat_id = _get(raw, "chat_id", f"missing-{line_no}", line_no)
            records.append(
                {
                    "line_no": line_no,
                    "chat_id": chat_id,
                    "parent_chat_id": _get(raw, "parent_chat_id", -1, line_no),
                    "timestamp": float(_get(raw, "timestamp", 0.0, line_no)),
                    "input_length": int(_get(raw, "input_length", 0, line_no)),
                    "output_length": int(_get(raw, "output_length", 0, line_no)),
                    "turn": int(_get(raw, "turn", 1, line_no)),
                }
            )

    by_chat = {r["chat_id"]: r for r in records}
    bad_parent: set[Any] = set()

    def root_id(chat_id: Any) -> Any:
        seen = set()
        cur = chat_id
        while True:
            rec = by_chat.get(cur)
            if rec is None or cur in seen:
                bad_parent.add(chat_id)
                return chat_id
            seen.add(cur)
            parent = rec["parent_chat_id"]
            if parent == -1 or str(parent) == "-1":
                return cur
            if parent not in by_chat:
                bad_parent.add(chat_id)
                return chat_id
            cur = parent

    kv = cfg["kv_cache"]
    linear_state_gb = linear_state_per_session_gb(cfg)
    requests = []
    for idx, rec in enumerate(sorted(records, key=lambda r: r["timestamp"])):
        session_id = root_id(rec["chat_id"])
        # Sticky routing uses the reconstructed root; invalid parents become
        # one-request sessions so broken trace links do not fabricate affinity.
        attn_kv_bytes = (
            rec["input_length"]
            * 2
            * int(kv["num_layers"])
            * int(kv["num_kv_heads"])
            * int(kv["head_dim"])
            * int(kv["bytes_per_elem"])
        )
        requests.append(
            Request(
                idx=idx,
                chat_id=rec["chat_id"],
                parent_chat_id=rec["parent_chat_id"],
                session_id=session_id,
                turn=rec["turn"],
                timestamp=rec["timestamp"],
                input_length=rec["input_length"],
                output_length=rec["output_length"],
                kv_gb=attn_kv_bytes / GB + linear_state_gb,
                arrival_time=rec["timestamp"],
            )
        )

    if bad_parent:
        logging.warning(
            "%d records had missing/invalid parent links and became single-turn sessions",
            len(bad_parent),
        )
    return requests


def linear_state_per_session_gb(cfg: dict[str, Any]) -> float:
    linear = cfg.get("linear_attention_state") or {}
    if not linear:
        return 0.0
    qk_heads = int(linear["num_qk_heads"])
    v_heads = int(linear["num_v_heads"])
    # Gated DeltaNet keeps a recurrent S matrix per QK head. With grouped V
    # heads, each QK head owns v_heads/qk_heads value heads.
    state_elems = (
        int(linear["num_layers"])
        * qk_heads
        * int(linear["head_dim"])
        * (v_heads // qk_heads)
        * int(linear["head_dim"])
    )
    return state_elems * int(linear["bytes_per_elem"]) / GB


def kv_delta(worker: Worker, req: Request) -> float:
    return max(0.0, req.kv_gb - worker.resident_kv_by_session.get(req.session_id, 0.0))


def can_admit(worker: Worker, req: Request, hbm_budget_gb: float) -> tuple[bool, str]:
    # Queueing rule: a request waits if the sticky worker has no compute slot or
    # lacks enough HBM to admit this turn's resident KV.
    if len(worker.active) >= worker.max_active_sessions:
        return False, COMPUTE_QUEUE
    if worker.hbm_used_gb + kv_delta(worker, req) > hbm_budget_gb:
        return False, HBM_QUEUE
    return True, ""


def snapshot(worker: Worker, req: Request) -> None:
    req.hbm_used_at_arrival_gb = worker.hbm_used_gb
    req.resident_kv_at_arrival_gb = worker.resident_kv_gb
    req.active_sessions_at_arrival = len(worker.active)
    req.paused_sessions_at_arrival = worker.paused_sessions
    req.compute_load_proxy_at_arrival = worker.compute_load_proxy


def admit(
    worker: Worker,
    req: Request,
    now: float,
    cfg: dict[str, Any],
    completions: list[tuple[float, int, int]],
    stats: dict[str, int],
) -> None:
    req.admission_time = now
    decode_tps = float(cfg["serving"]["decode_tokens_per_second"])
    req.finish_time = now + (req.output_length / decode_tps if decode_tps > 0 else 0.0)
    worker.active[req.idx] = req
    worker.active_sessions.add(req.session_id)
    # HBM model: fixed model memory plus the latest retained context footprint.
    # Non-monotonic input lengths keep the older larger KV instead of shrinking.
    old_kv = worker.resident_kv_by_session.get(req.session_id, 0.0)
    if req.kv_gb < old_kv:
        stats["non_monotonic_input_length_turns"] += 1
    worker.resident_kv_by_session[req.session_id] = max(old_kv, req.kv_gb)
    heapq.heappush(completions, (req.finish_time, worker.worker_id, req.idx))


def record_timeseries(
    rows: list[dict[str, Any]], workers: list[Worker], now: float
) -> None:
    for worker in workers:
        rows.append(
            {
                "time": now,
                "worker_id": worker.worker_id,
                "hbm_used_gb": worker.hbm_used_gb,
                "resident_kv_gb": worker.resident_kv_gb,
                "active_sessions": len(worker.active),
                "paused_resident_sessions": worker.paused_sessions,
                "queue_len": len(worker.queue),
                "compute_load_proxy": worker.compute_load_proxy,
            }
        )


def drain_queue(
    worker: Worker,
    now: float,
    cfg: dict[str, Any],
    hbm_budget_gb: float,
    completions: list[tuple[float, int, int]],
    stats: dict[str, int],
) -> None:
    while worker.queue:
        req = worker.queue[0]
        ok, _ = can_admit(worker, req, hbm_budget_gb)
        if not ok:
            return
        worker.queue.popleft()
        admit(worker, req, now, cfg, completions, stats)


def simulate(
    config_path: Path, cfg: dict[str, Any] | None = None, write_full: bool = True
) -> dict[str, Any]:
    cfg = cfg or load_config(config_path)
    trace_path = Path(cfg["trace"]["path"])
    requests = load_records(trace_path, cfg)
    grouped = _group(requests)
    session_last_idx = {
        sid: max(r.idx for r in group) for sid, group in grouped.items()
    }
    next_turn_time = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: r.timestamp)
        for cur, nxt in zip(ordered, ordered[1:]):
            next_turn_time[cur.idx] = nxt.timestamp

    base_hbm_gb = sum(
        float(cfg["model"][k])
        for k in ("base_model_gb", "workspace_gb", "resident_expert_gb")
    )
    workers = [
        Worker(i, base_hbm_gb, int(cfg["serving"]["max_active_sessions"]))
        for i in range(int(cfg["cluster"]["num_workers"]))
    ]
    hbm_budget_gb = float(cfg["cluster"]["hbm_budget_gb"])
    session_worker: dict[Any, int] = {}
    completions: list[tuple[float, int, int]] = []
    idle_expiries: list[tuple[float, int, Any, int]] = []
    timeseries: list[dict[str, Any]] = []
    stats = {"non_monotonic_input_length_turns": 0}
    timeout = cfg.get("simulation", {}).get("kv_idle_timeout_s")
    timeout = None if timeout is None else float(timeout)

    def release_idle_until(now: float) -> None:
        while idle_expiries and idle_expiries[0][0] <= now:
            expire, worker_id, session_id, generation = heapq.heappop(idle_expiries)
            worker = workers[worker_id]
            if worker.idle_generation_by_session.get(session_id) != generation:
                continue
            if session_id in worker.active_sessions:
                continue
            worker.resident_kv_by_session.pop(session_id, None)
            worker.idle_generation_by_session.pop(session_id, None)
            drain_queue(worker, expire, cfg, hbm_budget_gb, completions, stats)
            record_timeseries(timeseries, workers, expire)

    def complete_until(now: float) -> None:
        while True:
            next_completion = completions[0][0] if completions else math.inf
            next_idle = idle_expiries[0][0] if idle_expiries else math.inf
            if next_completion == math.inf and next_idle == math.inf:
                return
            if min(next_completion, next_idle) > now:
                return
            if next_idle <= next_completion:
                release_idle_until(next_idle)
                continue
            finish, worker_id, req_idx = heapq.heappop(completions)
            worker = workers[worker_id]
            req = worker.active.pop(req_idx, None)
            if req is None:
                continue
            worker.active_sessions.discard(req.session_id)
            if req.idx == session_last_idx[req.session_id]:
                worker.resident_kv_by_session.pop(req.session_id, None)
                worker.idle_generation_by_session.pop(req.session_id, None)
            elif (
                timeout is not None
                and next_turn_time.get(req.idx, math.inf) - finish > timeout
            ):
                generation = (
                    worker.idle_generation_by_session.get(req.session_id, 0) + 1
                )
                worker.idle_generation_by_session[req.session_id] = generation
                heapq.heappush(
                    idle_expiries,
                    (finish + timeout, worker.worker_id, req.session_id, generation),
                )
            drain_queue(worker, finish, cfg, hbm_budget_gb, completions, stats)
            record_timeseries(timeseries, workers, finish)

    for req in requests:
        complete_until(req.timestamp)
        if req.session_id not in session_worker:
            # Sticky routing rule: first turn goes to the least-HBM worker, and
            # every later turn follows that first assignment.
            session_worker[req.session_id] = min(
                workers, key=lambda w: w.hbm_used_gb
            ).worker_id
        worker = workers[session_worker[req.session_id]]
        req.assigned_worker = worker.worker_id
        snapshot(worker, req)
        ok, reason = can_admit(worker, req, hbm_budget_gb)
        if ok and not worker.queue:
            admit(worker, req, req.timestamp, cfg, completions, stats)
        else:
            req.was_queued = True
            req.queue_reason = reason or (
                worker.queue[0].queue_reason if worker.queue else HBM_QUEUE
            )
            worker.queue.append(req)
        record_timeseries(timeseries, workers, req.timestamp)

    complete_until(math.inf)
    observed_end_time = max(
        [r.timestamp for r in requests]
        + [r.finish_time for r in requests if r.finish_time is not None]
    )
    per_request = build_per_request_df(requests, observed_end_time)
    result_dir = (
        write_outputs(cfg, config_path, per_request, timeseries) if write_full else None
    )
    return {
        "result_dir": result_dir,
        "per_request": per_request,
        "timeseries": pd.DataFrame(timeseries),
        "stats": stats,
    }


def _group(requests: list[Request]) -> dict[Any, list[Request]]:
    groups: dict[Any, list[Request]] = defaultdict(list)
    for req in requests:
        groups[req.session_id].append(req)
    return groups


def write_outputs(
    cfg: dict[str, Any],
    config_path: Path,
    per_request: pd.DataFrame,
    timeseries: list[dict[str, Any]],
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(cfg["output"]["result_dir"]) / stamp
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, result_dir / "config.yaml")

    per_request.to_csv(
        result_dir / "per_request.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(timeseries).to_csv(
        result_dir / "per_worker_timeseries.csv", index=False
    )
    write_summary(cfg, per_request, result_dir / "summary.csv")
    make_plots(result_dir)
    return result_dir


def build_per_request_df(
    requests: list[Request], observed_end_time: float
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chat_id": r.chat_id,
                "parent_chat_id": r.parent_chat_id,
                "session_id": r.session_id,
                "turn": r.turn,
                "timestamp": r.timestamp,
                "input_length": r.input_length,
                "output_length": r.output_length,
                "assigned_worker": r.assigned_worker,
                "arrival_time": r.arrival_time,
                "admission_time": r.admission_time,
                "finish_time": r.finish_time,
                "queue_time_ms": (
                    (
                        (
                            r.admission_time
                            if r.admission_time is not None
                            else observed_end_time
                        )
                        - r.arrival_time
                    )
                    * 1000
                    if r.was_queued
                    else 0.0
                ),
                "was_queued": r.was_queued,
                "queue_reason": r.queue_reason,
                "hbm_used_at_arrival_gb": r.hbm_used_at_arrival_gb,
                "resident_kv_at_arrival_gb": r.resident_kv_at_arrival_gb,
                "active_sessions_at_arrival": r.active_sessions_at_arrival,
                "paused_sessions_at_arrival": r.paused_sessions_at_arrival,
                "compute_load_proxy_at_arrival": r.compute_load_proxy_at_arrival,
            }
            for r in requests
        ]
    )


def write_summary(cfg: dict[str, Any], df: pd.DataFrame, path: Path) -> None:
    queued = df[df["was_queued"] == True]  # noqa: E712
    hbm_budget = float(cfg["cluster"]["hbm_budget_gb"])
    hbm_ratio = (
        queued["hbm_used_at_arrival_gb"] / hbm_budget
        if len(queued)
        else pd.Series(dtype=float)
    )
    row = {
        "trace_name": cfg["trace"]["name"],
        "num_workers": cfg["cluster"]["num_workers"],
        "hbm_budget_gb": hbm_budget,
        "base_model_gb": cfg["model"]["base_model_gb"],
        "workspace_gb": cfg["model"]["workspace_gb"],
        "resident_expert_gb": cfg["model"]["resident_expert_gb"],
        "total_requests": len(df),
        "queued_requests": len(queued),
        "queued_request_ratio": _mean(df["was_queued"]),
        "hbm_queued_requests": int((queued["queue_reason"] == HBM_QUEUE).sum()),
        "compute_queued_requests": int((queued["queue_reason"] == COMPUTE_QUEUE).sum()),
        "p50_queue_time_ms": _quantile(queued["queue_time_ms"], 0.50),
        "p95_queue_time_ms": _quantile(queued["queue_time_ms"], 0.95),
        "p99_queue_time_ms": _quantile(queued["queue_time_ms"], 0.99),
        "mean_hbm_usage_at_queue_pct": _mean(hbm_ratio) * 100,
        "p95_hbm_usage_at_queue_pct": _quantile(hbm_ratio, 0.95) * 100,
        "fraction_queue_with_hbm_gt_90pct": _mean(hbm_ratio > 0.90),
        "mean_compute_load_at_queue": _mean(queued["compute_load_proxy_at_arrival"]),
        "fraction_queue_with_compute_lt_70pct": _mean(
            queued["compute_load_proxy_at_arrival"] < 0.70
        ),
        "mean_active_sessions_at_queue": _mean(queued["active_sessions_at_arrival"]),
        "mean_paused_sessions_at_queue": _mean(queued["paused_sessions_at_arrival"]),
        "mean_resident_kv_gb_at_queue": _mean(queued["resident_kv_at_arrival_gb"]),
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _mean(values: Any) -> float:
    return float(values.mean()) if len(values) else 0.0


def _quantile(values: Any, q: float) -> float:
    return float(values.quantile(q)) if len(values) else 0.0


def make_plots(result_dir: Path) -> None:
    req = pd.read_csv(result_dir / "per_request.csv")
    ts = pd.read_csv(result_dir / "per_worker_timeseries.csv")
    cfg = load_config(result_dir / "config.yaml")
    hbm_budget = float(cfg["cluster"]["hbm_budget_gb"])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    for worker_id, group in ts.groupby("worker_id"):
        ax1.plot(group["time"], group["hbm_used_gb"], label=f"worker {worker_id} HBM")
        ax2.step(
            group["time"],
            group["queue_len"],
            where="post",
            linestyle="--",
            label=f"worker {worker_id} queue",
        )
    ax1.set_xlabel("time")
    ax1.set_ylabel("HBM usage (GB)")
    ax2.set_ylabel("queue length")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "fig_worker_hbm_queue_timeseries.pdf")
    plt.close(fig)

    req = req.copy()
    req["hbm_usage_ratio"] = req["hbm_used_at_arrival_gb"] / hbm_budget
    queued = req[req["was_queued"] == True]  # noqa: E712
    nonqueued = req[req["was_queued"] != True]  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(
        [nonqueued["hbm_usage_ratio"], queued["hbm_usage_ratio"]],
        labels=["nonqueued", "queued"],
    )
    axes[0].set_ylabel("HBM usage ratio")
    axes[1].boxplot(
        [
            nonqueued["compute_load_proxy_at_arrival"],
            queued["compute_load_proxy_at_arrival"],
        ],
        labels=["nonqueued", "queued"],
    )
    axes[1].set_ylabel("compute load proxy")
    fig.tight_layout()
    fig.savefig(result_dir / "fig_queued_vs_nonqueued_hbm_compute.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    values = queued["queue_time_ms"].sort_values().to_list()
    if values:
        y = [(i + 1) / len(values) for i in range(len(values))]
        ax.plot(values, y)
    ax.set_xlabel("queue time (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(result_dir / "fig_queue_time_cdf.pdf")
    plt.close(fig)


def quick_sanity(config_path: Path) -> Path:
    base_cfg = load_config(config_path)
    cases = [
        ("A", 80, None, 0),
        ("B", 80, 300, 0),
        ("C", 80, 300, 20),
        ("D", 120, 300, 20),
    ]
    rows = []
    for name, hbm_budget, timeout, resident_expert in cases:
        cfg = copy.deepcopy(base_cfg)
        cfg["cluster"]["hbm_budget_gb"] = hbm_budget
        cfg["model"]["resident_expert_gb"] = resident_expert
        cfg.setdefault("simulation", {})["kv_idle_timeout_s"] = timeout
        result = simulate(config_path, cfg=cfg, write_full=False)
        rows.append(quick_summary_row(name, cfg, result))

    out_dir = Path(base_cfg["output"]["result_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quick_sanity_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def quick_summary_row(
    name: str, cfg: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    req = result["per_request"]
    ts = result["timeseries"]
    queued = req[req["was_queued"] == True]  # noqa: E712
    hbm_budget = float(cfg["cluster"]["hbm_budget_gb"])
    hbm_ratio = (
        queued["hbm_used_at_arrival_gb"] / hbm_budget
        if len(queued)
        else pd.Series(dtype=float)
    )
    return {
        "config_name": name,
        "hbm_budget_gb": hbm_budget,
        "kv_idle_timeout_s": cfg.get("simulation", {}).get("kv_idle_timeout_s"),
        "resident_expert_gb": cfg["model"]["resident_expert_gb"],
        "total_requests": len(req),
        "queued_request_ratio": _mean(req["was_queued"]),
        "hbm_queued_requests": int((queued["queue_reason"] == HBM_QUEUE).sum()),
        "compute_queued_requests": int((queued["queue_reason"] == COMPUTE_QUEUE).sum()),
        "fraction_queue_with_hbm_gt_90pct": _mean(hbm_ratio > 0.90),
        "fraction_queue_with_compute_lt_70pct": _mean(
            queued["compute_load_proxy_at_arrival"] < 0.70
        ),
        "peak_resident_kv_gb": float(ts["resident_kv_gb"].max()) if len(ts) else 0.0,
        "p95_resident_kv_gb": _quantile(ts["resident_kv_gb"], 0.95),
        "non_monotonic_input_length_turns": result["stats"][
            "non_monotonic_input_length_turns"
        ],
    }


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--quick-sanity", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.quick_sanity:
        print(quick_sanity(args.config))
    else:
        print(simulate(args.config)["result_dir"])


def plot_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    make_plots(args.result_dir)
    print(args.result_dir)
