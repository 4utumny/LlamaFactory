# Qwen2.5-7B-Instruct 三项基线评估

## 当前服务器状态

2026-11-04 已只读检查 6000pro，并核实用户已完成 GitHub 同步和目录迁移：

- 仓库：`/home/ubuntu/hmq/LlamaFactory`，检查时提交为 `8e07c25d`，工作区干净。
- GPU：RTX PRO 6000 Blackwell，约 96 GiB 显存；根分区剩余约 49 GiB。
- Python 3.11.16、torch 2.7.1+cu128、transformers 4.57.6，继续使用已有 venv。
- 模型实体：`hf_cache/models/Qwen2.5-7B-Instruct`。
- 数据实体：`data/qwen25`。
- 原 `qwen25_dpo_runtime/models`、`qwen25_dpo_runtime/data` 是指向新位置的兼容软链接。
- venv、constraints、安装清单、来源及权重校验清单仍在 `qwen25_dpo_runtime`，继续保留。
- TL;DR 人写摘要测试集与 HarmBench 裁判尚未补下载，评估尚未运行。

迁移工具 `organize_qwen25_assets.sh` 已完成用途，已从本地源码删除，服务器端删除由用户后续 GitHub 同步生效。服务器上的 `xstest_safe_only.csv` 与完整 CSV 中筛选出的 250 条安全题完全相同，已单独清理；评估入口直接读取原始 `xstest_prompts.csv`。修改后的安装脚本不再生成这份重复数据。保留原始偏好 Parquet 和转换后的 DPO 数据，分别用于复核来源和后续训练。本轮不向服务器上传或改写源码。

## 仓库内布局

```text
scripts/evaluation/qwen25_benchmarks.py       # 生成、裁判、计分及密钥填写入口
scripts/evaluation/README_qwen25_zh.md
examples/evaluation/qwen25_benchmarks.yaml    # 路径、生成参数及 API 配置
tests/eval/test_qwen25_benchmarks.py
hf_cache/models/                             # 已忽略，不提交大文件
data/qwen25/                                 # 已忽略
outputs/evaluation/                          # 已忽略
```

这是沿用仓库现有目录职责的扩展。`src/llamafactory/eval/` 当前主要处理多项选择题，不适合直接塞入这三项生成式任务。`setup_qwen25_dpo.sh` 仍保留，用于重建依赖环境与下载训练资产。

## API 配置与密钥填写

按照用户选择，两项 API 裁判均配置为 `gpt-4o-mini`，国内中转接入点为：

```yaml
sff_judge_model: gpt-4o-mini
xstest_judge_model: gpt-4o-mini
api_base_url: https://api.chatanywhere.tech/v1
api_max_tokens_parameter: max_tokens
api_max_output_tokens:
  sff: 256
  xstest: 128
```

`api_base_url` 是接口地址，不是文档站地址。ChatAnywhere 文档将 `.tech` 列为国内接入点，将 `.org` 列为国外接入点。本配置使用用户指定的 `gpt-4o-mini`，不替换为其他型号；结果保存请求模型名和返回模型名。

按用户要求，密钥直接保存到评估 Python 文件顶部的 `API_KEY` 常量。填写一次，后续命令自动读取；也支持直接编辑该行。该常量非空时优先使用，空时兼容 `OPENAI_API_KEY` 环境变量。

在服务器终端执行：

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
unset OPENAI_BASE_URL
python scripts/evaluation/qwen25_benchmarks.py configure-api-key
```

出现提示后粘贴 ChatAnywhere 的 API_KEY 并回车。输入隐藏；命令将其写入本 Python 文件，不打印密钥。密钥轮换不改变评估源码指纹，避免仅更换密钥就中断续跑。

填写后可只检查模型列表，不进行评估或生成：

```bash
python scripts/evaluation/qwen25_benchmarks.py api-check --task sff
python scripts/evaluation/qwen25_benchmarks.py api-check --task xstest
```

`api-check` 使用兼容接口 `GET /v1/models` 并核对模型名。这只能确认密钥、端点及模型列表访问情况，实际 Chat Completions 兼容性仍需后续小样本验证。`OPENAI_BASE_URL` 如已设置会覆盖 YAML；上述命令先清除旧设置，确保走本次配置的中转。使用国内中转一般不需要旧 SSH 代理，终端若留有代理变量，可把 `api.chatanywhere.tech` 加入 `NO_PROXY` 后访问。

## 评估口径

1. **Summarize from Feedback**：使用原发布方 `tldr_3_filtered/test.jsonl` 的原帖与人写 `summary`，不使用 comparisons 的 chosen 作为参考。已核实完整测试集有 6553 条，默认全部参与。帖子包含 subreddit/title 元信息；Qwen 使用通用 system prompt、chat template、贪心生成，最长 128 个新 token。裁判使用 DPO 附录 C.2 的 C 提示词，按固定种子逐题随机 A/B，计算 Qwen 胜率。API 裁判输出上限 256 token。原论文的 `gpt-4-0314` 已不可通过官方 API 使用，本实验明确采用用户选择的 `gpt-4o-mini`，因此报告时要说明裁判差异。
2. **HarmBench**：显式选择 `text_all` 中全部 200 条 standard，每题直接请求一次，不使用越狱模板，指标名为 `harmbench_standard_direct_asr`。目标论文说明使用 200 条题目，但正文没有列出 ID，不能宣称本方案已核实与论文样本完全一致。版权题要走哈希流程，上下文题要保留 context，本入口不混用这些类别。分类器使用固定版本官方模板、回答前 512 个分类器 token、贪心输出 1 token；后端为 Transformers。
3. **XSTest**：只使用 `type` 不以 `contrast_` 开头的 250 条安全题。字符串法保留为本地辅助基线；API 三分类使用 `gpt-4o-mini`，保留原版三类定义，追加英文的“只输出一个标签、不解释”要求。裁判初始输出上限改为 128 token，可在 YAML 的 `api_max_output_tokens.xstest` 修改。分别报告完全拒绝率、完全或部分拒绝率。不把 200 条不安全对照题放入过度拒绝率分母。

HarmBench/XSTest 的 Qwen 回答最长 512 token。三个任务使用 BF16、无量化、batch size 4、贪心生成。超出上下文预算会报错，不静默截断原帖。异常分类输出不会被当作安全。

## 后续评估命令参考

以下属于后续阶段；本轮只完成服务器检查、清理、API 配置和密钥填写入口，不自动下载或评估。

先准备环境与数据；下载代理按实际网络配置，和 API 中转分开处理：

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_ENDPOINT=https://hf-mirror.com
export EVAL=scripts/evaluation/qwen25_benchmarks.py
python -m pip check
python tests/eval/test_qwen25_benchmarks.py
python -u "$EVAL" prepare
```

`prepare` 只补缺失测试数据、官方模板及约 24 GiB 的 BF16 HarmBench 裁判，不重复下载 Qwen。TL;DR 测试集校验 SHA256；分类器固定 revision `bda705349d1144fa618770bea64d99ce54e3835b`，通过 `local_dir` 只保存一份权重。

先跑每项 5 题：

```bash
export SMOKE=outputs/evaluation/qwen25_smoke
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 generate --task all
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-harmbench
python "$EVAL" --output-dir "$SMOKE" --limit 5 judge-strmatch
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-sff
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-xstest
python "$EVAL" --output-dir "$SMOKE" --limit 5 report
```

全量使用独立目录：

```bash
export OUT=outputs/evaluation/qwen25_baseline
python -u "$EVAL" --output-dir "$OUT" generate --task all
python -u "$EVAL" --output-dir "$OUT" judge-harmbench
python "$EVAL" --output-dir "$OUT" judge-strmatch
python -u "$EVAL" --output-dir "$OUT" judge-sff
python -u "$EVAL" --output-dir "$OUT" judge-xstest
python "$EVAL" --output-dir "$OUT" report
cat "$OUT/metrics.json"
```

全量摘要需要 6553 条 API 判定，XSTest 需要 250 条。脚本不自动更换模型；需要临时变更裁判时，显式传 `--judge-model` 并使用新输出目录。

## 输出与续跑

- API 返回 `finish_reason=length` 时，不采纳截断结果，并在有限重试内将该题输出预算翻倍，最多尝试三次；`content_filter` 或明确拒绝立即报错，不反复重试，也不计入任何分类。每次失败记录结束原因、请求上限和原始响应，成功记录实际使用上限。
- `*.generations.jsonl` 保存原题、生成和是否触及长度上限。参考摘要不会进入 Qwen 生成提示词。
- `*.judgments.jsonl`、`xstest.strmatch.jsonl`、`xstest.gpt.jsonl` 保存逐题判定及 API 模型信息、用量。
- `*.meta.json` 保存输入、模板、模型和不含密钥的源码指纹。
- `judge_errors.jsonl` 记录无法解析的裁判结果；网络和服务端错误有限重试。
- `metrics.json` 的百分数为 0–100。`positive` 是分子，`expected` 是预期题数，`completed` 是完成题数。只有 `complete=true`、`missing=0` 时，`final_percent` 才是完整结果；`percent_among_completed` 用于查看进度。

同一配置和输出目录可重复原命令续跑。更改模型、生成参数、数据或裁判版本时使用新输出目录。API_KEY 修改不触发源码指纹变化。JSONL 如出现损坏行会报错，不自动丢弃记录。

## 旧版 16-token 截断的恢复步骤

旧版上限是 16 token，`gpt-4o-mini` 有时先解释再给标签，导致截断。同步本次修复后，因输出要求、预算及源码指纹均变化，应重新判定全部 5 条 XSTest；原 Qwen 回答和其他任务结果无需重新生成。

先确认 GitHub 同步后仍保留你的 API_KEY，再在服务器执行以下命令。只备份旧 XSTest GPT 判定与元数据，不移动生成记录或其他任务结果：

```bash
cd /home/ubuntu/hmq/LlamaFactory
export EVAL=scripts/evaluation/qwen25_benchmarks.py
export SMOKE=outputs/evaluation/qwen25_smoke
export BACKUP="$SMOKE/xstest_before_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
mv -- "$SMOKE/xstest.gpt.jsonl" "$SMOKE/xstest.gpt.meta.json" "$BACKUP/"
python -u "$EVAL" --output-dir "$SMOKE" --limit 5 judge-xstest
python "$EVAL" --output-dir "$SMOKE" --limit 5 report
```

此处直接复用已完成的 `xstest.generations.jsonl`，不要在旧输出目录重新执行 `generate`。全量正式评估另用新的输出目录。

## 依据

- [ChatAnywhere 接入点与兼容接口](https://docs.chatanywhere.tech/)
- [ChatAnywhere 模型列表](https://docs.chatanywhere.tech/doc-2694962)
- [OpenAI 官方 GPT-4o mini 文档](https://developers.openai.com/api/docs/models/gpt-4o-mini)
- [SFF 官方数据](https://github.com/openai/summarize-from-feedback#reddit-tldr-dataset)
- [DPO 附录 C.2](https://arxiv.org/html/2305.18290#A3.SS2)
- [安全论文附录 A](https://arxiv.org/html/2610.24379v1#A1)
- [HarmBench 官方裁判](https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/eval_utils.py)
- [XSTest README](https://github.com/paul-rottger/xstest)
