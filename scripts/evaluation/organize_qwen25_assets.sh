#!/usr/bin/env bash
# Move existing model/data directories within the same filesystem; preserve old paths.
# The virtual environment stays at its original absolute path.
set -Eeuo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/hmq/LlamaFactory}"
export RUNTIME_DIR="${RUNTIME_DIR:-/home/ubuntu/hmq/qwen25_dpo_runtime}"

"$RUNTIME_DIR/venv/bin/python" -B - <<'PY'
import os
from pathlib import Path

project = Path(os.environ["PROJECT_DIR"]).resolve(strict=True)
runtime = Path(os.environ["RUNTIME_DIR"]).resolve(strict=True)
if project != Path("/home/ubuntu/hmq/LlamaFactory") or runtime != Path("/home/ubuntu/hmq/qwen25_dpo_runtime"):
    raise SystemExit("本迁移命令仅针对已检查的两个绝对路径；其他路径请先人工检查。")
if not (project / "pyproject.toml").is_file() or not (runtime / "venv/bin/python").is_file():
    raise SystemExit("仓库或虚拟环境不符合预期。")

pairs = [(runtime / "models", project / "hf_cache/models"), (runtime / "data", project / "data/qwen25")]
# Validate every move before starting the first one.
pending = []
for source, target in pairs:
    if target.resolve(strict=False) != target or not target.is_relative_to(project):
        raise SystemExit(f"目标路径含有其他软链接或越出仓库：{target}")
    if source.is_symlink():
        if source.resolve(strict=True) != target.resolve(strict=True):
            raise SystemExit(f"已有软链接指向其他位置：{source}")
        print(f"已完成：{source} -> {target}")
        continue
    if not source.is_dir() or source.resolve(strict=True) != source:
        raise SystemExit(f"源路径不是预期的实体目录：{source}")
    if target.exists() or target.is_symlink():
        raise SystemExit(f"目标已存在，不覆盖：{target}")
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    if source.stat().st_dev != parent.stat().st_dev:
        raise SystemExit("源和目标不在同一文件系统；停止，避免复制大文件。")
    pending.append((source, target))

for source, target in pending:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"迁移实体目录：{source} -> {target}", flush=True)
    source.rename(target)
    try:
        source.symlink_to(target, target_is_directory=True)
    except Exception:
        target.rename(source)
        raise
print("完成。旧模型和数据路径通过软链接继续有效；venv 未迁移。")
PY
