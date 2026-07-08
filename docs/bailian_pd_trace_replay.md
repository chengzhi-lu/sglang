# Bailian PD Trace Replay

Minimal real-backend workflow for replaying Alibaba Qwen-Bailian Trace A against
SGLang PD disaggregation. This intentionally does not use the custom simulator.

## Build trace-replayer

```bash
git clone https://github.com/blitz-serving/trace-replayer /sgl-workspace/trace-replayer
cd /sgl-workspace/trace-replayer
cargo build -p request-sim --bin client --release
```

The binary is:

```text
/sgl-workspace/trace-replayer/target/release/client
```

## Official trace-replayer arguments used

- Trace path: `--dataset-path dataset/qwen_traceA_blksz_16.jsonl`
- Backend URL: `--endpoint http://127.0.0.1:8000/v1/chat/completions`
- Model name: `--model-name <served-model-name>`
- Replay rate: `--scale-factor 1|2|4`
- Runtime/window length: `--time-in-secs 600`
- Output metrics path: `--output-path <run>/replayer.jsonl`
- Summary path: `--summary-path <run>/replayer.summary.json`
- Dataset type: `--dataset bailian`
- API type: `--api openai`
- TTFT/TPOT: requires `--stream`

The upstream replayer does not expose a trace start-offset/window selector. The
minimal window is therefore the first `--time-in-secs` seconds of Trace A replay.

## Run SGLang PD replay

Use an FP16/BF16 model path. Do not pass the local FP8 checkpoint for this test.

```bash
python scripts/bailian_pd_trace_replay.py \
  --model-path /path/to/Qwen3.6-35B-A3B \
  --model-name qwen3.6-35b-a3b-fp16 \
  --time-in-secs 600 \
  --rates 1 2 4
```

Useful pressure knobs:

```bash
python scripts/bailian_pd_trace_replay.py \
  --model-path /path/to/Qwen3.6-35B-A3B \
  --model-name qwen3.6-35b-a3b-fp16 \
  --time-in-secs 600 \
  --rates 1 2 4 \
  --mem-fraction-static 0.90 \
  --max-running-requests 64
```

The script starts one prefill worker on GPU 0, one decode worker on GPU 1, and
one router. Each replay rate gets a fresh backend to avoid cache contamination.

Outputs are written to:

```text
results/bailian_pd_trace_replay/<timestamp>/
```

Required artifacts:

- `summary.csv`
- `timeseries.csv`
- `fig_runtime_pressure_over_time.pdf`
- `fig_ttft_vs_replay_rate.pdf`
- `rate_<N>x/replayer.jsonl`
- `rate_<N>x/commands.json`
- `rate_<N>x/{prefill,decode,router,replayer}.log`

The summary is based on trace-replayer streaming fields:

- TTFT: `first_token_time`
- TPOT: `total_time / output_length`
- Timing drift: `s_time_drift`

Runtime pressure is sampled from SGLang Prometheus metrics and `nvidia-smi`:

- waiting queue: `sglang:num_queue_reqs`
- running requests: `sglang:num_running_reqs`
- KV/token pressure: `sglang:token_usage`
- absolute tokens when available: `sglang:kv_used_tokens`
- GPU memory/utilization: `nvidia-smi`
