# Local LayerKV launcher

Use `scripts/layerkv_local.py` when running the LayerKV scripts from this
worktree. It keeps the repository-local Python source, the existing venv, and
the Qwen3.6-35B-A3B Hugging Face snapshot in one place. No venv activation or
manual model-path lookup is needed.

Print the resolved paths:

```bash
python scripts/layerkv_local.py paths
```

The current defaults are:

```text
venv Python: /mnt/vdb/chengzhi/agent_bench_sglang_venv/bin/python
HF hub:      /mnt/vdb/hf_home/hub
model cache: models--Qwen--Qwen3.6-35B-A3B
Python library: <worktree>/python
```

These values are persisted in `scripts/layerkv_local_config.json`, so the
launcher does not need to rediscover the venv, HF hub, or library path for each
run. Inspect them with:

```bash
python scripts/layerkv_local.py paths
```

Use `--config PATH` to select another path profile. Individual path options
such as `--hub-root` or `--python-library-root` override the profile for one
run; experiment options still belong after the target and are never stored in
the profile.

Run a focused test or produce a dry experiment plan:

```bash
python scripts/layerkv_local.py unit-tests \
  test/registered/unit/layerkv/test_shared_expert.py
python scripts/layerkv_local.py prefetch-test
python scripts/layerkv_local.py expert-validation \
  --output-dir /tmp/layerkv_expert_validation
python scripts/layerkv_local.py concurrent-perf \
  --output-dir /mnt/vdb/chengzhi/layerkv_next_plan
```

For CPU-only artifact comparison, the launcher can reuse the same Python
environment without resolving or loading a model:

```bash
python scripts/layerkv_local.py reference-compare \
  --candidate /mnt/vdb/chengzhi/run/fixed.json \
  --reference /mnt/vdb/chengzhi/native/native.json
```

Add `--strict` when a shape-matched, guard-clean comparison should return a
nonzero status for any mismatch. Without it, the report is diagnostic only.

`concurrent-perf` receives the resolved `--model-path` automatically. Add
`--execute` only when the plan has been reviewed and a GPU run is intended.
The other model-backed targets use the same model path:

```text
hybrid-validation  shared-perf  server-smoke
policy-eval         real-validation
expert-validation   (CPU-only; does not resolve a model path)
```

`unit-tests` runs pytest with the worktree's Python library root as its current
directory. With no forwarded path it runs the LayerKV test directory; passed
pytest files/directories may be written relative to either the repository root
or that child directory and are normalized to absolute paths.
Pytest-only options such as `-q`, `-k ...`, and `--collect-only` still use the
LayerKV directory by default.

For a matched fixed/lending comparison, `concurrent-perf` runs `fixed` first by
default. Use `--arm-order lend fixed` for a reverse-order control run; the
summary still reports the arms as fixed versus lend.

For a different local cache or source checkout, override the path options
before the target:

```bash
python scripts/layerkv_local.py \
  --venv-python /path/to/venv/bin/python \
  --python-library-root /path/to/worktree/python \
  --hub-root /path/to/huggingface/hub \
  --model-cache-name models--Qwen--Qwen3.6-35B-A3B \
  --model-revision REVISION \
  concurrent-perf --output-dir /mnt/vdb/chengzhi/layerkv_run
```

To use a different persistent profile:

```bash
python scripts/layerkv_local.py \
  --config /path/to/layerkv_local_config.json \
  paths
```

The launcher writes the effective paths and child command before starting a
child. The child process receives the selected Python library through
`PYTHONPATH`, and the selected Hugging Face cache through framework cache
variables; experiment settings remain explicit child command-line arguments.
`--python-source-root` remains accepted as a compatibility alias.
