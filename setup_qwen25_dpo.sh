#!/usr/bin/env bash
# Qwen2.5-7B-Instruct / BF16 LoRA DPO：环境、模型、偏好数据和评测数据准备。
# 用法：bash setup_qwen25_dpo.sh
# 可覆盖：PROJECT_DIR、WORK_DIR、CUDA_VISIBLE_DEVICES（默认 0）。
# 仅配置虚拟环境、下载模型和数据，不运行训练，不生成训练配置或测试产物。
# 官方依据：
# https://github.com/hiyouga/LlamaFactory#installation
# https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md#preference-dataset
# https://pytorch.org/blog/pytorch-2-7/ （Blackwell 需要 CUDA 12.8 构建）
# https://huggingface.co/datasets/openai/summarize_from_feedback
# https://github.com/centerforaisafety/HarmBench
# https://github.com/paul-rottger/xstest
set -Eeuo pipefail
trap 'printf "\n失败：第 %s 行，请检查上方输出。修复后可重新执行本脚本。\n" "$LINENO" >&2' ERR

export PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/hmq/LlamaFactory}"
export WORK_DIR="${WORK_DIR:-$(dirname "$PROJECT_DIR")/qwen25_dpo_runtime}"
export VENV_DIR="$WORK_DIR/venv"
export MODEL_DIR="$WORK_DIR/models/Qwen2.5-7B-Instruct"
export DATA_DIR="$WORK_DIR/data"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
# 国内镜像直连；GitHub、HF 官方站和国外下载重定向使用代理。
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1,.cn,hf-mirror.com"
export no_proxy="$NO_PROXY"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export HF_HOME="$WORK_DIR/cache/huggingface"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || { echo "找不到 LlamaFactory：$PROJECT_DIR" >&2; exit 1; }
mkdir -p "$WORK_DIR" "$MODEL_DIR" "$DATA_DIR"
cd "$PROJECT_DIR"
echo "项目：$PROJECT_DIR；运行目录：$WORK_DIR"
git rev-parse HEAD | tee "$WORK_DIR/llamafactory-commit.txt"
uname -a
cat /etc/os-release
nvidia-smi
df -h "$WORK_DIR"

# 实测该服务器仅剩约 34GiB。pip 禁用持久缓存，临时 wheel 放入内存盘，
# 模型通过 local_dir 下载只保留一份；不清理用户已有缓存或修改驱动。
if [[ -z "${TMPDIR:-}" && -w /dev/shm ]]; then
    export TMPDIR="$WORK_DIR/tmp"
    if (( $(df -Pk /dev/shm | awk 'NR==2 {print $4}') > 12582912 )); then
        export TMPDIR="/dev/shm/qwen25-dpo-$UID"
    fi
fi
export TMPDIR="${TMPDIR:-$WORK_DIR/tmp}"
mkdir -p "$TMPDIR"

UV="$(command -v uv || true)"
if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; fi
if [[ -z "$UV" ]]; then
    python3 -m pip install --no-cache-dir --target "$WORK_DIR/bootstrap" uv==0.12.15
    UV="$WORK_DIR/bootstrap/uv/uv"
fi
# 服务器已有 uv 管理的 Python 3.11；若无，先尝试国内 Python 构建镜像。
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV" venv --python 3.11 --seed --index-url "$PIP_INDEX_URL" "$VENV_DIR" || \
        UV_PYTHON_INSTALL_MIRROR=https://github.com/astral-sh/python-build-standalone/releases/download \
        "$UV" venv --python 3.11 --seed --index-url "$PIP_INDEX_URL" "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python --version

# 满足服务器当前 pyproject.toml 的依赖范围；固定核心版本以便重现。
# README 的旧推荐 torch 2.6/cu126 不适用于这台 Blackwell。
cat > "$WORK_DIR/constraints.txt" <<'CONSTRAINTS'
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
transformers==4.57.6
datasets==4.0.0
accelerate==1.11.0
peft==0.18.1
trl==0.24.0
torchdata==0.11.0
huggingface-hub==0.36.2
numpy==2.2.6
CONSTRAINTS

# 上海交大 CUDA wheel 镜像优先，失败后用 PyTorch 官方 cu128 源。
# CUDA 依赖也从同一个 cu128 索引安装，避免误装 CPU/其他 CUDA 版本。
python -m pip install --no-cache-dir \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128 || \
python -m pip install --no-cache-dir \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 严格沿用官方源码安装入口与 metrics 依赖，不跳过依赖检查。
python -m pip install --no-cache-dir -c "$WORK_DIR/constraints.txt" -e .
python -m pip install --no-cache-dir -c "$WORK_DIR/constraints.txt" -r requirements/metrics.txt
python -m pip check
python -m pip freeze > "$WORK_DIR/installed-packages.txt"
llamafactory-cli env

python -u - <<'PY'
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

work = Path(os.environ["WORK_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
manifest_path = work / "download-manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

def save_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=(15, 120))
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)

# ModelScope 优先；其次 HF 国内镜像；最后 HF 官方代理。local_dir 避免缓存复制。
patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"]
model_source = manifest.get("model", {})
if not model_source:
    failures = []
    try:
        from modelscope import snapshot_download as ms_download
        ms_download("Qwen/Qwen2.5-7B-Instruct", local_dir=str(model_dir), allow_patterns=patterns)
        model_source = {"repo": "Qwen/Qwen2.5-7B-Instruct", "provider": "modelscope.cn"}
    except Exception as exc:
        failures.append(repr(exc))
        print("ModelScope 下载失败，尝试 HF 国内镜像：", exc, flush=True)
        for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
            try:
                api = HfApi(endpoint=endpoint)
                revision = api.model_info("Qwen/Qwen2.5-7B-Instruct").sha
                snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=revision, endpoint=endpoint,
                                  local_dir=str(model_dir), allow_patterns=patterns, max_workers=4)
                model_source = {"repo": "Qwen/Qwen2.5-7B-Instruct", "provider": endpoint, "revision": revision}
                break
            except Exception as exc:
                failures.append(repr(exc))
                print("模型源失败：", endpoint, exc, flush=True)
        else:
            raise RuntimeError(f"模型下载失败：{failures}")

# 检查每一个分片确实存在，能读取 safetensors 元数据，且基座权重为 BF16。
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
weight_files = sorted(set(index["weight_map"].values()))
for filename in weight_files:
    path = model_dir / filename
    assert path.stat().st_size > 1024 * 1024, path
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            assert tensors.get_slice(name).get_dtype() == "BF16", (filename, name)
model_source["weights"] = {f: {"bytes": (model_dir / f).stat().st_size,
                               "sha256": sha256(model_dir / f)} for f in weight_files}
manifest["model"] = model_source
save_manifest()

# HF 原始 main 是旧式 dataset loading script；使用官方转换的 Parquet 分支，
# 直接读取 Parquet，兼容 datasets 4.0，不使用 trust_remote_code 或第三方重打包。
raw = data_dir / "summarize_from_feedback_raw"
raw.mkdir(exist_ok=True)
for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
    try:
        api = HfApi(endpoint=endpoint)
        revision = manifest.get("summarize_from_feedback", {}).get("revision")
        revision = revision or api.dataset_info("openai/summarize_from_feedback", revision="refs/convert/parquet").sha
        snapshot_download("openai/summarize_from_feedback", repo_type="dataset", revision=revision,
                          endpoint=endpoint, local_dir=str(raw), allow_patterns=["comparisons/**/*.parquet"],
                          max_workers=2)
        manifest["summarize_from_feedback"] = {"repo": "openai/summarize_from_feedback",
                                               "revision": revision, "endpoint": endpoint}
        break
    except Exception as exc:
        print("偏好数据源失败：", endpoint, exc, flush=True)
else:
    raise RuntimeError("Summarize from Feedback 下载失败")

stats = {}
for split in ["train", "validation"]:
    shards = sorted((raw / "comparisons" / split).glob("*.parquet"))
    assert shards, f"缺少 {split} Parquet"
    output = data_dir / f"sff_{split}.jsonl"
    counts = {"raw": 0, "written": 0, "empty_or_identical": 0}
    with output.open("w", encoding="utf-8") as out:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    counts["raw"] += 1
                    choice = row["choice"]
                    assert choice in (0, 1), f"异常偏好标签：{choice}"
                    summaries = row["summaries"]
                    texts = summaries["text"] if isinstance(summaries, dict) else [s["text"] for s in summaries]
                    assert len(texts) == 2
                    chosen, rejected = texts[choice], texts[1 - choice]
                    info = row["info"]
                    body = info.get("post") or info.get("article") or ""
                    if not (body.strip() and chosen.strip() and rejected.strip()) or chosen == rejected:
                        counts["empty_or_identical"] += 1
                        continue
                    title = info.get("title") or ""
                    prompt = "Summarize the following text.\n\n"
                    prompt += (f"Title: {title}\n\n" if title else "") + body + "\n\nTL;DR:"
                    record = {"instruction": prompt, "chosen": chosen, "rejected": rejected}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["written"] += 1
    assert counts["written"] > 0
    stats[split] = counts
    print("Summarize from Feedback：", split, counts, flush=True)
manifest["summarize_from_feedback"]["conversion"] = stats
manifest["summarize_from_feedback"]["parquet_sha256"] = {
    str(p.relative_to(raw)): sha256(p) for p in sorted(raw.glob("comparisons/**/*.parquet"))}

# 从官方 Git 仓库获取数据和评测说明，保存 commit 与 SHA256。
# 不把 HarmBench 老版 requirements（torch/vllm）装入训练环境。
def official_files(repo, files, dest):
    dest.mkdir(parents=True, exist_ok=True)
    key = repo.split("/")[-1]
    previous = manifest.get(key, {})
    revision = previous.get("revision")
    if not revision:
        result = subprocess.check_output(["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"], text=True)
        revision = result.split()[0]
    checksums = {}
    for filename in files:
        path = dest / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.exists() and sha256(path) == previous.get("sha256", {}).get(filename)):
            response = get(f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}")
            path.write_bytes(response.content)
        checksums[filename] = sha256(path)
    manifest[key] = {"repo": repo, "revision": revision, "sha256": checksums}
    save_manifest()

hb = data_dir / "HarmBench"
official_files("centerforaisafety/HarmBench", ["README.md", "LICENSE", "docs/evaluation_pipeline.md"] +
               [f"data/behavior_datasets/harmbench_behaviors_text_{s}.csv" for s in ["all", "val", "test"]], hb)
hb_counts = {}
for split in ["all", "val", "test"]:
    with (hb / f"data/behavior_datasets/harmbench_behaviors_text_{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and {"Behavior", "BehaviorID", "FunctionalCategory"} <= rows[0].keys()
    hb_counts[split] = len(rows)
manifest["HarmBench"]["rows"] = hb_counts
print("HarmBench 文本行为数据：", hb_counts, flush=True)

xs = data_dir / "XSTest"
official_files("paul-rottger/xstest", ["xstest_prompts.csv", "readme.md", "LICENSE",
               "evaluation/classify_completions_gpt.py", "evaluation/classify_completions_strmatch.py"], xs)
with (xs / "xstest_prompts.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
assert rows and {"prompt", "type", "label"} <= rows[0].keys()
safe = [r for r in rows if r["label"].lower() == "safe"]
unsafe = [r for r in rows if r["label"].lower() == "unsafe"]
assert len(safe) == 250 and len(unsafe) == 200, (len(safe), len(unsafe))
# 评估入口直接从原始 CSV 筛选 safe，无需额外写一份重复子集。
manifest["xstest"]["rows"] = {"safe": len(safe), "unsafe": len(unsafe)}
print("XSTest：250 条 safe、200 条 unsafe；过度拒绝率使用 safe 子集。", flush=True)
save_manifest()

# 独立 dataset_dir 是官方支持方式，无需改动仓库 data/dataset_info.json。
dataset_info = {}
for split in ["train", "validation"]:
    dataset_info[f"sff_{split}"] = {"file_name": f"sff_{split}.jsonl", "ranking": True,
        "columns": {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"}}
(data_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")

PY

# 生成可直接 source 的环境入口，不修改 shell 配置。
{
    printf 'source %q\n' "$VENV_DIR/bin/activate"
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export TOKENIZERS_PARALLELISM=false\n'
    printf 'cd %q\n' "$PROJECT_DIR"
} > "$WORK_DIR/activate.sh"

df -h "$WORK_DIR"
printf '\n准备完成。环境：source %q\n模型：%s\n数据集：%s\n' \
    "$WORK_DIR/activate.sh" "$MODEL_DIR" "$DATA_DIR"
echo "来源版本与 SHA256：$WORK_DIR/download-manifest.json"
echo 'HarmBench 已准备文本行为全集/val/test；后续 ASR 仍需确定攻击方法并用官方分类器评判。'
echo 'XSTest 已准备官方全量与 250 条 safe 子集；过度拒绝率分母使用 safe 子集。'
