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

1. **Summarize from Feedback**：使用原发布方 `tldr_3_filtered/test.jsonl` 的原帖与人写 `summary`，不使用 comparisons 的 chosen 作为参考。原始测试集有 6553 条；按用户要求，现在和后续默认只取文件顺序中的前 200 条，原始序号为 0–199，不随机抽样、不重新编号。YAML 的 `sff_max_samples: 200` 只限制 SFF；`--limit 5` 会进一步缩为前 5 条以便冒烟测试。帖子包含 subreddit/title 元信息；Qwen 使用通用 system prompt、chat template、贪心生成，最长 128 个新 token。裁判使用 DPO 附录 C.2 的 C 提示词，按固定种子逐题随机 A/B，计算 Qwen 胜率。API 裁判输出上限 256 token。原论文的 `gpt-4-0314` 已不可通过官方 API 使用，本实验明确采用用户选择的 `gpt-4o-mini`，因此报告时要说明裁判差异及前 200 条选样范围。
2. **HarmBench**：显式选择 `text_all` 中全部 200 条 standard，每题直接请求一次，不使用越狱模板，指标名为 `harmbench_standard_direct_asr`。目标论文说明使用 200 条题目，但正文没有列出 ID，不能宣称本方案已核实与论文样本完全一致。版权题要走哈希流程，上下文题要保留 context，本入口不混用这些类别。分类器使用固定版本官方模板、回答前 512 个分类器 token、贪心输出 1 token；后端为 Transformers。
3. **XSTest**：只使用 `type` 不以 `contrast_` 开头的 250 条安全题。字符串法保留为本地辅助基线；API 三分类使用 `gpt-4o-mini`，原样使用官方分类提示词，只替换 QUESTION 和 RESPONSE，保留末尾的 `CLASS:`，不追加任何指令。API 的通用 system 消息也与官方脚本一致。裁判初始输出上限为 128 token，可在 YAML 的 `api_max_output_tokens.xstest` 修改；这是对官方 16-token 参数的调整，应在实验记录中说明。分别报告完全拒绝率、完全或部分拒绝率。不把 200 条不安全对照题放入过度拒绝率分母。

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

按当前选样方案，完整 SFF 评估需要 200 条 API 判定，XSTest 需要 250 条，共 450 条，重试另计。HarmBench 仍为 200 条本地分类判定。脚本不自动更换模型；需要临时变更裁判时，显式传 `--judge-model` 并使用新输出目录。

## 输出与续跑

- HTTP 525 等中转错误、429、常见 5xx 和连接/读取超时会自动重试，每条请求最多尝试 12 次（包含首次）。等待时间依次为 5、10、20、40、60、60……秒，最长单次等待 60 秒。重试不改变裁判提示词、模型或输出预算；401 等认证错误立即报错。超过次数仍失败则退出，已保存判定保留。
- YAML 的 `api_retry_max_attempts`、`api_retry_initial_delay_seconds`、`api_retry_max_delay_seconds` 可调整重试次数和等待时间。终端会打印 HTTP 状态、下一次尝试编号及等待秒数，不打印密钥。
- API 返回 `finish_reason=length` 时，不采纳截断结果，并在有限重试内将该题输出预算翻倍，最多尝试三次；`content_filter` 或明确拒绝立即报错，不反复重试，也不计入任何分类。每次失败记录结束原因、请求上限和原始响应，成功记录实际使用上限。
- 裁判返回解释时，只接受开头或最后一行的明确分类标签，保留完整原始文本；出现多个类别或仅在解释中提及标签则报错，不猜测类别。最后一行支持 `**3_partial_refusal.**` 等句号位于格式标记内部的写法。这只影响返回值解析，不改变发送给裁判的官方提示词。
- `*.generations.jsonl` 保存原题、生成和是否触及长度上限。参考摘要不会进入 Qwen 生成提示词。
- `*.judgments.jsonl`、`xstest.strmatch.jsonl`、`xstest.gpt.jsonl` 保存逐题判定及 API 模型信息、用量。
- `*.meta.json` 保存输入、模板、模型和不含密钥的源码指纹。
- `judge_errors.jsonl` 记录无法解析的裁判结果；网络和服务端错误有限重试。
- `metrics.json` 的百分数为 0–100。`positive` 是分子，`expected` 是预期题数，`completed` 是完成题数。只有 `complete=true`、`missing=0` 时，`final_percent` 才是完整结果；`percent_among_completed` 用于查看进度。

同一配置和输出目录可重复原命令续跑。更改模型、生成参数、数据或裁判版本时使用新输出目录。API_KEY 修改不触发源码指纹变化。JSONL 如出现损坏行会报错，不自动丢弃记录。

## 525 自动重试更新后的续跑

525 更新只修改 HTTP 重试及其续跑兼容逻辑；后续的 SFF 前 200 条更新单独改变了 SFF 选样范围；XSTest 格式兼容修复只增加对最后一行标签内部标点的支持，已保存的有效标签无需重新判定。YAML 的 `resume_compatible_script_sha256` 记录经过验证的旧源码指纹到当前源码的映射。只有来源指纹成对匹配，且输入、模型、提示词等阶段签名全部一致时，程序才会自动续跑；SFF 数据范围改变时必须先执行下节的 `trim-sff`，其他评估参数变化仍报错。

首次续跑会保留旧 `.meta.json` 的备份，再更新来源指纹；不会删除、覆盖或重新判定已有 JSONL 记录。无须清空输出目录。停止旧进程后，由用户通过 GitHub 同步本次 Python、YAML 等改动，然后执行：

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
set -o pipefail
python -u scripts/evaluation/qwen25_benchmarks.py \
    --output-dir outputs/evaluation/qwen25_baseline \
    judge-sff 2>&1 | tee -a outputs/evaluation/qwen25_baseline/sff_judge.log
```

保持使用原来的 API_KEY 和输出目录。重试只能等待中转恢复，不能修复服务方持续存在的源站 SSL 配置故障。

## 将已有 SFF 结果裁剪为前 200 条

先停止正在运行的评估进程，再通过 GitHub 同步本次代码和 YAML。只修改本地源码，由用户亲自执行服务器上的裁剪命令。

原始 `data/qwen25/tldr_3_filtered/test.jsonl` 保留全部 6553 条，不修改官方文件及其 SHA256。只裁剪活动结果目录中的 `sff.generations.jsonl` 与 `sff.judgments.jsonl`，保留与原始测试题前 200 条 ID 对应的记录，并同步更新两个 `.meta.json` 的来源、输入指纹。其他任务结果不变。

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate

# Validate and preview; no files are changed.
python scripts/evaluation/qwen25_benchmarks.py \
    --output-dir outputs/evaluation/qwen25_baseline trim-sff

# Back up the original result files, then keep only the selected SFF rows.
python scripts/evaluation/qwen25_benchmarks.py \
    --output-dir outputs/evaluation/qwen25_baseline trim-sff --apply

# Refresh the denominator and scores using the retained records.
python scripts/evaluation/qwen25_benchmarks.py \
    --output-dir outputs/evaluation/qwen25_baseline report
```

`trim-sff` 会验证现有生成参数、数据 ID、参考摘要、裁判提示词、API 参数、A/B 位置和胜负映射。缺失生成、重复 ID 或参数不一致时会停止，不覆盖结果。`--apply` 前会复制原文件到自动命名的 `sff_before_subset_*` 目录；普通写入异常或中断会恢复备份。默认只预览，不修改文件。

本次只读检查时，生成文件为 6553 条，判定文件为 550 条；前 200 条均已完成，最后一条为 `sff:199:t3_2oxvlx`。裁剪后无需重新生成或重新调用 SFF 裁判。若以后在尚未完成 200 条判定时裁剪，可以继续运行 `judge-sff` 补齐剩余判定。

后续原有命令默认评估 SFF 前 200 条。不要用全局 `--limit 200` 代替这个设置，否则 XSTest 也会被缩为 200 条。HarmBench 仍为 200 条，XSTest 仍为 250 条。

## 恢复官方提示词后的运行步骤

旧版 16-token 上限造成截断；中间版本追加了输出指令，本次已移除并恢复官方提示词。同步本次修复后，应重新判定全部 5 条 XSTest，不混用旧提示词的标签；原 Qwen 回答和其他任务结果无需重新生成。输出预算仍为 128，发生截断时最多增加到 256、512，提示词保持不变。

先确认 GitHub 同步后仍保留你的 API_KEY，再在服务器执行以下命令。只备份旧 XSTest GPT 判定与元数据，不移动生成记录或其他任务结果：

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
export EVAL=scripts/evaluation/qwen25_benchmarks.py
export SMOKE=outputs/evaluation/qwen25_smoke
export BACKUP="$SMOKE/xstest_before_official_prompt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
for file in xstest.gpt.jsonl xstest.gpt.meta.json; do
    if [ -f "$SMOKE/$file" ]; then
        mv -- "$SMOKE/$file" "$BACKUP/"
    fi
done
python tests/eval/test_qwen25_benchmarks.py
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
