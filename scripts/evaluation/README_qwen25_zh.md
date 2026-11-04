# Qwen2.5-7B-Instruct 三项基线评估

本方案根据 `评估.md` 和 2026-11-04 对 6000pro 的只读检查编写。服务器上尚未安装本方案、迁移目录、下载裁判或执行评估。

## 检查结论与目录选择

- GPU：RTX PRO 6000 Blackwell，97887 MiB 显存，检查时使用 626 MiB；根分区剩余约 49 GiB。
- 仓库：`/home/ubuntu/hmq/LlamaFactory`，提交 `849b3891b867ecb0af50047a1fb5d9d64ea0cafd`，检查时工作区干净。
- 原运行目录：`/home/ubuntu/hmq/qwen25_dpo_runtime`，模型约 15 GiB、数据约 347 MiB、venv 约 7.7 GiB。
- Python 3.11.16；torch 2.7.1+cu128；transformers 4.57.6；datasets 4.0.0；accelerate 1.11.0。没有 vLLM 和 openai SDK。本方案直接复用 Transformers 和 requests，不安装或升级这些库。
- 已有 Qwen 基座、SFF comparisons 训练/验证集、HarmBench CSV、XSTest 数据与原始分类脚本。尚缺 TL;DR 人写参考摘要测试集与 HarmBench 裁判权重。

`setup_qwen25_dpo.sh` 通过 `WORK_DIR` 把环境、模型和数据放在仓库旁边。这是部署选择，方便隔离大文件；LlamaFactory 没有要求资产必须放在仓库外。`constraints.txt`、下载清单、安装包列表和提交号也有复现价值。

建议的仓库内路径如下。这是沿用仓库现有目录职责的扩展，不是上游规定的唯一布局。

```text
LlamaFactory/
  scripts/evaluation/
    qwen25_benchmarks.py             # 生成、裁判、计分入口
    organize_qwen25_assets.sh        # 用户手动运行的目录整理
    README_qwen25_zh.md
  examples/evaluation/
    qwen25_benchmarks.yaml           # 路径、种子、生成长度等可提交配置
  tests/eval/test_qwen25_benchmarks.py
  hf_cache/models/                   # 已有 .gitignore 忽略
    Qwen2.5-7B-Instruct/
    HarmBench-Llama-2-13b-cls/
  data/qwen25/                       # 需要加入 .gitignore
    sff_train.jsonl
    sff_validation.jsonl
    HarmBench/
    XSTest/
    tldr_3_filtered/test.jsonl
    evaluation-manifest.json
  outputs/evaluation/                # 已有 .gitignore 忽略
    qwen25_smoke/
    qwen25_baseline/
```

不修改 `src/llamafactory/eval/`：该入口当前是多项选择题评估器，这三项属于生成与外部裁判流程。现有 `scripts/eval_bleu_rouge.py` 也不能计算本任务要求的摘要胜率。

整理脚本会在同一文件系统内迁移原 `models/` 与 `data/`，不复制大文件；旧位置建立软链接，已有训练数据配置和安装脚本仍可使用旧路径。venv 保持原绝对位置，避免其中的启动器和 editable 安装失效。若目标已存在或路径异常，脚本停止，不合并或覆盖目录。

## 评估口径

1. **Summarize from Feedback**：下载原发布方 `tldr_3_filtered/test.jsonl`，使用 `post` 与人写的 `summary`；不使用 comparisons 中的 chosen 作为参考。已核实完整测试集有 6553 条，默认全部参与，正式摘要判定需要 6553 次 API 调用。下载时验证 SHA256。帖子带 subreddit 和 title 元信息，Qwen 使用 chat template、通用 system prompt、贪心生成，最长 128 个新 token。裁判采用 DPO 附录 C.2 的 C 提示词，按固定种子逐题随机 A/B，成功解析的 Qwen 胜出数 / 成功完成的裁判数。这里只沿用其评判协议，不声称复现 DPO 原论文所有训练和生成设置。
2. **HarmBench**：目标论文附录 A 写明 200 条题目及 HarmBench-13B 分类器，但正文未给出这 200 条的 ID。本方案显式选择 `text_all` 的全部 200 条 `standard`，每题直接请求一次，不加越狱模板，报告 `harmbench_standard_direct_asr`。这不是全部 400 条的完整 HarmBench，也不声称已核实与目标论文样本 ID 完全一致。版权题要走哈希流程，上下文题要保留 context；本入口拒绝混用这些类别。裁判从固定版本官方 `eval_utils.py` 提取模板，沿用官方回答前 512 个分类器 token、temperature=0、裁判生成 1 token 的做法。后端为 Transformers，不是官方示例的 vLLM；异常输出直接停止，不当成安全。
3. **XSTest**：仅选择 `type` 不以 `contrast_` 开头的 250 条安全题。200 条不安全对照题不进入过度拒绝率分母。默认调用原版 `strmatch_label` 函数，不执行原脚本底部写死文件名的循环。可另运行 GPT 三分类，提示词直接提取自服务器原文件；分别报告完全拒绝率、完全或部分拒绝率。字符串匹配有漏判和误判，不能把其标签等同于语义裁判结果。

HarmBench/XSTest 默认最长生成 512 token；三个任务都使用 BF16、无量化、batch size 4、贪心生成。超出上下文预算会报错，不静默截断帖子。每条结果记录是否碰到生成长度上限。

**裁判版本限制：**DPO 指定的 `gpt-4-0314` 已于 2026-03-26 从官方 API 下线。配置保留此名称以明确原始协议；现在实际运行需要你显式选择可用的裁判固定版本，通过 `--judge-model` 指定，并在实验报告说明差异。XSTest 原脚本使用的 `gpt-4` 也是历史别名，不应依赖其继续可用。脚本不自动换模型，保存 API 请求模型名、返回模型名和 token 用量。

## 1. 从 Windows 上传新增文件

以下命令在 Windows PowerShell 执行。压缩包仅包含新增源码、配置和测试，不包含模型、数据、密钥或整个仓库。

```powershell
Set-Location D:\LlamaFactory
New-Item -ItemType Directory -Force -Path outputs | Out-Null
tar -czf outputs/qwen25-evaluation-files.tar.gz scripts/evaluation examples/evaluation tests/eval/test_qwen25_benchmarks.py
scp outputs/qwen25-evaluation-files.tar.gz 6000pro:/home/ubuntu/hmq/qwen25-evaluation-files.tar.gz
ssh 6000pro
```

## 2. 在服务器解压并整理目录

以下开始均在服务器的 Bash 终端执行。先检查包内路径，再解压。

```bash
cd /home/ubuntu/hmq/LlamaFactory
tar -tzf /home/ubuntu/hmq/qwen25-evaluation-files.tar.gz
tar -xzf /home/ubuntu/hmq/qwen25-evaluation-files.tar.gz -C /home/ubuntu/hmq/LlamaFactory

grep -qxF '/data/qwen25/' .gitignore || printf '\n/data/qwen25/\n' >> .gitignore
bash scripts/evaluation/organize_qwen25_assets.sh

readlink -f hf_cache/models/Qwen2.5-7B-Instruct
readlink -f data/qwen25
ls -ld /home/ubuntu/hmq/qwen25_dpo_runtime/models /home/ubuntu/hmq/qwen25_dpo_runtime/data
git status --short
git check-ignore hf_cache/models/Qwen2.5-7B-Instruct/config.json data/qwen25/sff_train.jsonl
```

如果暂时不想迁移：跳过整理脚本，把 YAML 的 `model`、`classifier`、`data_root` 改成原运行目录中的绝对路径，其余步骤相同。

## 3. 激活现有环境并检查

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_ENDPOINT=https://hf-mirror.com

# 使用检查时已存在的 SSH 代理监听；保持本机代理与 SSH 转发在线。
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY=http://127.0.0.1:17897
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1,.cn,hf-mirror.com
export no_proxy="$NO_PROXY"

python -m pip check
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
python tests/eval/test_qwen25_benchmarks.py
python -m ruff check --no-cache scripts/evaluation/qwen25_benchmarks.py tests/eval/test_qwen25_benchmarks.py
python -m ruff format --check scripts/evaluation/qwen25_benchmarks.py tests/eval/test_qwen25_benchmarks.py
bash -n scripts/evaluation/organize_qwen25_assets.sh
```

## 4. 补齐测试集、官方模板和裁判权重

`prepare` 复用已有数据，只补下载约 10 MiB 的 TL;DR 测试集、官方裁判模板源码，以及约 24.3 GiB 的 BF16 分类器。通过 `local_dir` 下载，只保留一份权重。分类器固定 revision：`bda705349d1144fa618770bea64d99ce54e3835b`。

```bash
export EVAL=scripts/evaluation/qwen25_benchmarks.py
df -h .
python -u "$EVAL" prepare
df -h .
cat data/qwen25/evaluation-manifest.json
```

下载不通时可显式更换端点后重跑，不更换 revision：

```bash
HF_ENDPOINT=https://huggingface.co python -u "$EVAL" prepare
```

如果只想先验证生成和字符串拒绝率，可先运行 `prepare --skip-classifier`；运行 `judge-harmbench` 前仍需执行完整 `prepare`。

## 5. 配置 API 裁判

密钥只保留在当前终端的环境变量中，不输入到命令行字面量或源码。`api-check` 只读取模型元数据，不生成回答；它不能保证后续 Chat Completions 请求一定兼容所有模型参数，因此仍需下面的小样本检查。

```bash
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1
read -r -p '输入可用的固定版本摘要裁判模型名: ' SFF_JUDGE_MODEL
export SFF_JUDGE_MODEL
test -n "$SFF_JUDGE_MODEL"
python "$EVAL" api-check --task sff --judge-model "$SFF_JUDGE_MODEL"
```

如果使用自己的兼容 API，显式改 `OPENAI_BASE_URL`。不要在不存在的 `gpt-4-0314` 名称下让代理静默映射其他模型。所选模型需支持 Chat Completions 的 `temperature=0` 和配置中的 `max_tokens` 参数；若你的端点要求 `max_completion_tokens`，显式修改 YAML 的 `api_max_tokens_parameter`。修改配置后使用新输出目录。

## 6. 先跑每项 5 题

```bash
export SMOKE=outputs/evaluation/qwen25_smoke
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 generate --task all
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-harmbench
python "$EVAL" --output-dir "$SMOKE" --limit 5 judge-strmatch
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-sff --judge-model "$SFF_JUDGE_MODEL"
python "$EVAL" --output-dir "$SMOKE" --limit 5 report
```

## 7. 全量正式评估

以下全量使用独立输出目录，不会把 5 题检查误当成全量评估。各 GPU 阶段按顺序执行，进程退出后释放显存，不同时加载 Qwen 与裁判。

```bash
export OUT=outputs/evaluation/qwen25_baseline
mkdir -p "$OUT"
set -o pipefail

python -u "$EVAL" --output-dir "$OUT" generate --task all 2>&1 | tee "$OUT/generate.log"
python -u "$EVAL" --output-dir "$OUT" judge-harmbench 2>&1 | tee "$OUT/harmbench_judge.log"
python "$EVAL" --output-dir "$OUT" judge-strmatch
python -u "$EVAL" --output-dir "$OUT" judge-sff --judge-model "$SFF_JUDGE_MODEL" 2>&1 | tee "$OUT/sff_judge.log"
python "$EVAL" --output-dir "$OUT" report

cat "$OUT/metrics.json"
```

全局参数 `--output-dir "$OUT"` 必须放在子命令之前。Python 不自动读取 `OUT` 环境变量；不传参数时默认目录来自 YAML。

## 8. 可选：XSTest GPT 三分类

字符串法可作为无需 API 的基线；需要语义判断时，再运行这一阶段。可以与摘要裁判选同一个固定版本，但仍需显式记录。

```bash
read -r -p '输入可用的固定版本 XSTest 裁判模型名: ' XSTEST_JUDGE_MODEL
test -n "$XSTEST_JUDGE_MODEL"
python "$EVAL" api-check --task xstest --judge-model "$XSTEST_JUDGE_MODEL"
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-xstest --judge-model "$XSTEST_JUDGE_MODEL"
python -u "$EVAL" --output-dir "$OUT" judge-xstest --judge-model "$XSTEST_JUDGE_MODEL" 2>&1 | tee "$OUT/xstest_gpt_judge.log"
python "$EVAL" --output-dir "$OUT" report
unset OPENAI_API_KEY
```

## 输出与续跑

- `*.generations.jsonl`：原题、生成回答、长度上限标记；摘要数据另外保存参考摘要，参考不进入 Qwen 生成提示词。
- `*.judgments.jsonl`、`xstest.strmatch.jsonl`、`xstest.gpt.jsonl`：逐题判定和 API 原始结果/模型信息。
- `*.meta.json`：源码、输入、模板和模型元信息指纹。参数或输入改变时拒绝混用旧结果。
- `judge_errors.jsonl`：不合格式的裁判结果。HTTP 错误直接报错；网络/服务端错误有限重试，不无限循环。
- `metrics.json`：百分数均以 0–100 表示；`positive` 为分子，`expected` 为预期题数，`completed` 为完成题数。`percent_among_completed` 可用于观察进度；只有 `complete=true`、`missing=0` 时，`final_percent` 才是全量最终分数。未运行可选 GPT 分类时，相应字段为空属正常。

中断后在同一配置、同一输出目录重复原命令，会跳过已完成的 ID。更换模型、生成参数、数据、裁判版本或脚本时使用新输出目录。中断恰好造成 JSONL 最后一行损坏时会明确报错，需先检查该文件，不会自动丢弃记录。

`model_identity` 记录配置文件哈希及权重大小/mtime，未在每次运行重复哈希数十 GiB 权重；已有 Qwen 权重完整 SHA256 仍保存在原目录的 `download-manifest.json`。请保留该文件、`constraints.txt` 和已安装依赖清单。

## 核验范围

交付前已使用服务器现有 Python 在内存中运行计分与筛选测试，确认真实 HarmBench standard=200、XSTest safe=250，并验证 XSTest 原脚本的函数和提示词提取。检查过官方 TL;DR 下载接口与字段、裁判模型固定 revision。做过 Python/Bash 语法与 Ruff 检查。没有执行 GPU 生成、下载完整裁判或调用付费 API；因此必须由你完成小样本端到端检查后再跑全量。

## 原始依据

- [SFF 官方数据与测试集说明](https://github.com/openai/summarize-from-feedback#reddit-tldr-dataset)
- [DPO 论文附录 C.2](https://arxiv.org/html/2305.18290#A3.SS2)
- [目标安全论文附录 A 的评估协议](https://arxiv.org/html/2610.24379v1#A1)
- [HarmBench 固定版本官方裁判](https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/eval_utils.py)
- [HarmBench 官方完整计分入口](https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/evaluate_completions.py)
- [XSTest README](https://github.com/paul-rottger/xstest)
- [官方 API 模型下线记录](https://developers.openai.com/api/docs/deprecations)
