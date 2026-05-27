#!/usr/bin/env python3
"""Shared helpers for LayerKV SGLang evaluation scripts."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_MODEL_PATH = (
    "/data/wenyan/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B/"
    "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
)

FIG4_RECLAIM_LIMIT_MB = 4096.0

FIG4_WORKLOADS = {
    "batch-heavy": {
        "batch_size": 256,
        "input_len": 1024,
        "min_ctx": 768,
        "max_ctx": 1024,
        "output_len": 16,
        "dataset_name": "ShareGPT_V3_unfiltered_cleaned_split",
        "dataset_path": "/4IR-dataset/common/request_dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
    },
    "batch_heavy": {
        "batch_size": 256,
        "input_len": 1024,
        "min_ctx": 768,
        "max_ctx": 1024,
        "output_len": 16,
        "dataset_name": "ShareGPT_V3_unfiltered_cleaned_split",
        "dataset_path": "/4IR-dataset/common/request_dataset/ShareGPT_V3_unfiltered_cleaned_split.json",
    },
    "context-heavy": {
        "batch_size": 8,
        "input_len": 32768,
        "min_ctx": 28672,
        "max_ctx": 32768,
        "output_len": 16,
        "dataset_name": "WildChat-1M",
        "dataset_path": "/data/wenyan/.cache/huggingface/allenai___wild_chat-1_m",
    },
    "context_heavy": {
        "batch_size": 8,
        "input_len": 32768,
        "min_ctx": 28672,
        "max_ctx": 32768,
        "output_len": 16,
        "dataset_name": "WildChat-1M",
        "dataset_path": "/data/wenyan/.cache/huggingface/allenai___wild_chat-1_m",
    },
}


def apply_fig4_workload(args: Any) -> None:
    preset = FIG4_WORKLOADS.get(args.workload)
    args.fig4_aligned = preset is not None
    if preset is None:
        args.fig4_dataset_name = ""
        args.fig4_dataset_path = ""
        return
    args.batch_size = int(preset["batch_size"])
    args.input_len = int(preset["input_len"])
    args.output_len = int(preset["output_len"])
    args.min_ctx = int(preset["min_ctx"])
    args.max_ctx = int(preset["max_ctx"])
    args.fig4_dataset_name = str(preset["dataset_name"])
    args.fig4_dataset_path = str(preset["dataset_path"])
    # bench_one_batch skips rows whose batch size exceeds
    # max_total_tokens / (input_len + output_len). Keep the Fig4 workload fixed
    # and only enlarge the static pool enough for that workload shape.
    min_total_tokens = args.batch_size * (args.input_len + args.output_len)
    if getattr(args, "max_total_tokens", 0) <= 0:
        args.max_total_tokens = min_total_tokens + args.output_len
    if getattr(args, "max_running_requests", 0) <= 0:
        args.max_running_requests = args.batch_size
    if getattr(args, "mem_fraction_static", 0.0) <= 0.0:
        args.mem_fraction_static = 0.90


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


class InsufficientPromptsError(RuntimeError):
    """Not enough real prompts pass the Fig4 length filter."""


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _extract_sharegpt_user_text(record: Dict[str, Any]) -> Optional[str]:
    for turn in record.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from") or "").lower() not in ("human", "user"):
            continue
        value = turn.get("value")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_wildchat_user_text(record: Dict[str, Any]) -> Optional[str]:
    for turn in record.get("conversation") or []:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role") or "").lower() != "user":
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _prompt_metadata(
    *,
    dataset_name: str,
    dataset_path: Path,
    prompt_source: str,
    num_records_total: int,
    examined: int,
    discarded_no_user: int,
    discarded_too_short: int,
    discarded_too_long: int,
    selected: List[Dict[str, Any]],
    num_prompts: int,
    min_context_len: int,
    max_context_len: int,
    seed: int,
) -> Dict[str, Any]:
    counts = [int(x["token_count"]) for x in selected]
    ids = [str(x["record_id"]) for x in selected]
    ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    return {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "dataset_is_real": True,
        "fig4_real_dataset_loaded": True,
        "prompt_source": prompt_source,
        "synthetic_prompt_used": False,
        "prompt_repetition_used": False,
        "num_records_total": int(num_records_total),
        "num_records_examined": int(examined),
        "num_discarded_no_user": int(discarded_no_user),
        "num_discarded_too_short": int(discarded_too_short),
        "num_discarded_too_long": int(discarded_too_long),
        "num_selected": len(selected),
        "num_unique_selected": len(selected),
        "num_prompts": int(num_prompts),
        "min_context_len": int(min_context_len),
        "max_context_len": int(max_context_len),
        "seed": int(seed),
        "selected_record_ids": ids,
        "selected_record_ids_hash": ids_hash,
        "selected_token_counts": counts,
        "context_len_min": int(min(counts)),
        "context_len_max": int(max(counts)),
        "context_len_mean": float(sum(counts) / len(counts)),
        "context_len_p50": int(sorted(counts)[len(counts) // 2]),
        "context_len_p95": int(
            sorted(counts)[min(len(counts) - 1, int(len(counts) * 0.95))]
        ),
    }


def load_sharegpt_fig4_prompts(
    path: str,
    tokenizer: Any,
    num_prompts: int,
    min_context_len: int,
    max_context_len: int,
    seed: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ShareGPT JSON not found: {p}")
    with p.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"ShareGPT root must be a list, got {type(data).__name__}")

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    selected: List[Dict[str, Any]] = []
    examined = discarded_no_user = discarded_too_short = discarded_too_long = 0
    for idx in indices:
        if len(selected) >= num_prompts:
            break
        examined += 1
        rec = data[idx]
        text = _extract_sharegpt_user_text(rec)
        if text is None:
            discarded_no_user += 1
            continue
        n_tokens = _token_len(tokenizer, text)
        if n_tokens < min_context_len:
            discarded_too_short += 1
            continue
        if n_tokens > max_context_len:
            discarded_too_long += 1
            continue
        rec_id = str(
            rec.get("id")
            or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
        )
        selected.append(
            {
                "record_id": rec_id,
                "text": text,
                "token_count": n_tokens,
                "selection_index": idx,
            }
        )
    if len(selected) < num_prompts:
        raise InsufficientPromptsError(
            f"only {len(selected)}/{num_prompts} ShareGPT prompts in "
            f"[{min_context_len}, {max_context_len}] tokens after examining "
            f"{examined} records"
        )
    return [x["text"] for x in selected], _prompt_metadata(
        dataset_name="ShareGPT_V3_unfiltered_cleaned_split",
        dataset_path=p,
        prompt_source="sharegpt",
        num_records_total=len(data),
        examined=examined,
        discarded_no_user=discarded_no_user,
        discarded_too_short=discarded_too_short,
        discarded_too_long=discarded_too_long,
        selected=selected,
        num_prompts=num_prompts,
        min_context_len=min_context_len,
        max_context_len=max_context_len,
        seed=seed,
    )


def load_wildchat_fig4_prompts(
    path: str,
    tokenizer: Any,
    num_prompts: int,
    min_context_len: int,
    max_context_len: int,
    seed: int = 0,
    batch_size: int = 128,
) -> Tuple[List[str], Dict[str, Any]]:
    try:
        import pyarrow.ipc as pa_ipc
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("WildChat loading requires pyarrow") from exc

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"WildChat path not found: {root}")
    candidates = (
        [root] if root.is_file() else sorted(x for x in root.rglob("*") if x.is_file())
    )
    files = []
    for item in candidates:
        try:
            with item.open("rb") as f:
                magic = f.read(6)
        except OSError:
            continue
        if magic[:4] == b"PAR1":
            files.append(("parquet", item))
        elif item.suffix.lower() == ".arrow" or magic[:4] == b"\xff\xff\xff\xff":
            files.append(("arrow", item))
    if not files:
        raise FileNotFoundError(f"No WildChat parquet/arrow files under: {root}")

    rng = random.Random(seed)
    rng.shuffle(files)
    selected: List[Dict[str, Any]] = []
    examined = discarded_no_user = discarded_too_short = discarded_too_long = 0
    total_rows = 0
    min_chars = max(1, int(min_context_len * 2.0))
    max_chars = int(max_context_len * 10.0) if max_context_len > 0 else 0

    def iter_rows(kind: str, item: Path):
        nonlocal total_rows
        if kind == "parquet":
            pf = pq.ParquetFile(item)
            total_rows += pf.metadata.num_rows
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["conversation_hash", "conversation"],
            ):
                yield from batch.to_pylist()
            return
        with item.open("rb") as f:
            reader = pa_ipc.open_stream(f)
            schema = reader.schema
            hash_idx = schema.get_field_index("conversation_hash")
            conv_idx = schema.get_field_index("conversation")
            if hash_idx < 0 or conv_idx < 0:
                return
            for batch in reader:
                total_rows += batch.num_rows
                hashes = batch.column(hash_idx).to_pylist()
                convs = batch.column(conv_idx).to_pylist()
                for conv_hash, conv in zip(hashes, convs):
                    yield {
                        "conversation_hash": conv_hash,
                        "conversation": conv,
                    }

    def consider_record(rec: Dict[str, Any]) -> None:
        nonlocal examined, discarded_no_user, discarded_too_short, discarded_too_long
        if len(selected) >= num_prompts:
            return
        examined += 1
        text = _extract_wildchat_user_text(rec)
        if text is None:
            discarded_no_user += 1
            return
        if len(text) < min_chars:
            discarded_too_short += 1
            return
        if max_chars and len(text) > max_chars:
            discarded_too_long += 1
            return
        n_tokens = _token_len(tokenizer, text)
        if n_tokens < min_context_len:
            discarded_too_short += 1
            return
        if n_tokens > max_context_len:
            discarded_too_long += 1
            return
        rec_id = str(
            rec.get("conversation_hash")
            or hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
        )
        selected.append(
            {
                "record_id": rec_id,
                "text": text,
                "token_count": n_tokens,
                "selection_index": examined - 1,
            }
        )

    for kind, item in files:
        if len(selected) >= num_prompts:
            break
        batch_rows: List[Dict[str, Any]] = []
        for rec in iter_rows(kind, item):
            batch_rows.append(rec)
            if len(batch_rows) < batch_size:
                continue
            rng.shuffle(batch_rows)
            for row in batch_rows:
                consider_record(row)
                if len(selected) >= num_prompts:
                    break
            batch_rows = []
        if batch_rows and len(selected) < num_prompts:
            rng.shuffle(batch_rows)
            for row in batch_rows:
                consider_record(row)
                if len(selected) >= num_prompts:
                    break

    if len(selected) < num_prompts:
        raise InsufficientPromptsError(
            f"only {len(selected)}/{num_prompts} WildChat prompts in "
            f"[{min_context_len}, {max_context_len}] tokens after examining "
            f"{examined} records"
        )
    return [x["text"] for x in selected], _prompt_metadata(
        dataset_name="WildChat-1M",
        dataset_path=root,
        prompt_source="wildchat",
        num_records_total=total_rows,
        examined=examined,
        discarded_no_user=discarded_no_user,
        discarded_too_short=discarded_too_short,
        discarded_too_long=discarded_too_long,
        selected=selected,
        num_prompts=num_prompts,
        min_context_len=min_context_len,
        max_context_len=max_context_len,
        seed=seed,
    )


def load_fig4_prompt_ids(
    args: Any, tokenizer: Any
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not getattr(args, "fig4_aligned", False):
        raise ValueError("Fig4 prompt loading requires a known Fig4 workload")
    if args.fig4_dataset_name == "WildChat-1M":
        prompts, metadata = load_wildchat_fig4_prompts(
            args.fig4_dataset_path,
            tokenizer,
            num_prompts=int(args.batch_size),
            min_context_len=int(args.min_ctx),
            max_context_len=int(args.max_ctx),
            seed=int(args.seed),
        )
    else:
        prompts, metadata = load_sharegpt_fig4_prompts(
            args.fig4_dataset_path,
            tokenizer,
            num_prompts=int(args.batch_size),
            min_context_len=int(args.min_ctx),
            max_context_len=int(args.max_ctx),
            seed=int(args.seed),
        )
    input_ids = [
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.input_len),
        )["input_ids"]
        for text in prompts
    ]
    metadata["payload_input_len_min"] = min(len(ids) for ids in input_ids)
    metadata["payload_input_len_max"] = max(len(ids) for ids in input_ids)
    metadata["payload_input_len_mean"] = sum(len(ids) for ids in input_ids) / max(
        1, len(input_ids)
    )
    metadata["payload_input_ids_hash"] = hashlib.sha256(
        json.dumps(input_ids).encode()
    ).hexdigest()[:16]
    return input_ids, metadata


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
    cmd = [
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
    if getattr(args, "max_total_tokens", 0) > 0:
        cmd.extend(["--max-total-tokens", str(args.max_total_tokens)])
    if getattr(args, "max_running_requests", 0) > 0:
        cmd.extend(["--max-running-requests", str(args.max_running_requests)])
    if getattr(args, "mem_fraction_static", 0.0) > 0.0:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    return cmd


def layerkv_flags(
    *,
    mode: str,
    policy: str,
    reclaim_limit_mb: Optional[float],
    kvc_block_tokens: int,
    kvc_backend: str = "token-slot",
    dynamic_pressure_from_kvc: bool = False,
    scheduler: str = "async-deadline",
    runtime_profile: str = "optimized",
    debug_stats: bool = True,
    profile_detail: bool = False,
    expert_backing_cache_mb: float = 0.0,
    expert_cpu_backing_mode: str = "none",
    expert_install_layers_per_step: int = 1,
    expert_install_budget_mb: float = 128.0,
    expert_install_target_steps: int = 0,
) -> List[str]:
    flags = [
        "--enable-layerkv",
        "--layerkv-mode",
        mode,
        "--layerkv-policy",
        policy,
        "--layerkv-kvc-block-tokens",
        str(kvc_block_tokens),
        "--layerkv-kvc-backend",
        kvc_backend,
        "--layerkv-kvc-scheduler",
        scheduler,
        "--layerkv-runtime-profile",
        runtime_profile,
        "--layerkv-expert-backing-cache-mb",
        str(expert_backing_cache_mb),
        "--layerkv-expert-cpu-backing-mode",
        expert_cpu_backing_mode,
        "--layerkv-expert-install-layers-per-step",
        str(expert_install_layers_per_step),
        "--layerkv-expert-install-budget-mb",
        str(expert_install_budget_mb),
        "--layerkv-expert-install-target-steps",
        str(expert_install_target_steps),
    ]
    if reclaim_limit_mb is not None:
        flags.extend(["--layerkv-reclaim-limit-mb", str(reclaim_limit_mb)])
    if dynamic_pressure_from_kvc:
        flags.append("--layerkv-dynamic-pressure-from-kvc")
    if debug_stats:
        flags.append("--layerkv-debug-stats")
    if profile_detail:
        flags.append("--layerkv-profile-detail")
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
