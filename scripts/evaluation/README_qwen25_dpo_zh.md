# Qwen2.5 摘要任务 LoRA DPO 运行说明

本轮训练 Clean 组，使用正常摘要偏好对。基座为现有 Qwen2.5-7B-Instruct，参考策略是禁用 LoRA 后的同一基座。摘要以 Clean 对 M₀ 的同帖盲评为主要指标。

所有代码变更在本地完成。下面的 Bash 命令由用户在 GitHub 同步到 6000pro 后执行。

## 参数与数据

- 采用论文 TL;DR 设置 `pref_beta=0.5`、RMSprop、有效批量 64、前 150 个优化步骤线性预热。预热后恒定学习率是本项目的选择。依据为 [DPO 附录 B](https://arxiv.org/html/2305.18290v3#A2)。
- 比较学习率 `1.0e-6` 和 `5.0e-6`，两组各运行 384 步，再用固定开发集选择。LoRA rank 16、alpha 32、target all、dropout 0；BF16、无量化。
- 默认单卡批量 2、梯度累积 32；如果冒烟阶段显存不足，后续所有训练统一使用批量 1、累积 64。
- 数据从原始 Parquet 转换，保留 choice 对应的 chosen，按帖子 ID 和规范化正文哈希隔离数据。过滤空回答、相同回答、同向重复偏好对、与测试正文重合的训练对。
- 2026-11-04 只读检查得到 73,313 个训练对、1,024 个验证对、128 篇开发帖子。最长训练序列 663 token，所选 `cutoff_len=2048`，完整一轮约 1,146 次优化更新。最终数字由服务器运行准备脚本产生的 manifest 确认。
- 正式摘要测试使用原前 200 条中的 199 条，排除曾见于原始训练数据的 `sff:100:t3_1cv473`。HarmBench 仍为 200 条 standard 直接请求，XSTest 为 250 条安全请求。
- 准备脚本按长度检查选择 2048、4096 或 8192，使超长对占比不超过 1%，并过滤剩余超长对。命令读取 manifest 中的截止长度。

## 一次性初始化终端变量

在服务器打开终端，进入已经同步新文件的仓库。

```bash
cd /home/ubuntu/hmq/LlamaFactory
source /home/ubuntu/hmq/qwen25_dpo_runtime/venv/bin/activate
set -o pipefail

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
unset OPENAI_BASE_URL

C=examples/train_lora/qwen25_sff_dpo.yaml
E=scripts/evaluation/qwen25_dpo_eval.py
R=saves/qwen25/sff_clean
V=outputs/evaluation/qwen25_dpo
LOG=outputs/logs/qwen25_sff_dpo
mkdir -p "$LOG"

python -m pip check
python tests/eval/test_qwen25_benchmarks.py
python tests/eval/test_qwen25_sff_dpo.py
```

日志放在训练目录外，以免 Trainer 把预先生成的日志误认为已有训练结果。更换终端后重新执行变量设置。

## 准备数据

```bash
python -u scripts/training/prepare_qwen25_sff.py \
  2>&1 | tee "$LOG/prepare.log"

cat data/qwen25/dpo_clean/manifest.json
CUT=$(python -c 'import json; print(json.load(open("data/qwen25/dpo_clean/manifest.json"))["cutoff_len"])')
```

成功后才继续。准备脚本只读原始数据，在 `data/qwen25/dpo_clean/` 写入 `train.jsonl`、`validation.jsonl`、`dataset_info.json`、`dev_posts.jsonl`、`test_ids.json` 和 `manifest.json`。重复准备使用相同原始数据会得到相同划分。

检查评估 API 是否可用。它沿用现有基线脚本中的密钥和 GPT-4o-mini 配置。

```bash
python scripts/evaluation/qwen25_benchmarks.py api-check --task sff
python scripts/evaluation/qwen25_benchmarks.py api-check --task xstest
```

如果需要填写密钥，使用已有交互入口后重新检查。

```bash
python scripts/evaluation/qwen25_benchmarks.py configure-api-key
```

## 两步训练冒烟

```bash
llamafactory-cli train "$C" \
  cutoff_len="$CUT" \
  output_dir="$R/smoke" \
  max_samples=64 max_steps=2 warmup_steps=0 \
  gradient_accumulation_steps=1 \
  do_eval=false 'eval_strategy="no"' 'save_strategy="no"' \
  2>&1 | tee "$LOG/smoke.log"
```

确认日志显示 LoRA 可训练参数、训练损失正常且完成两步。冒烟仍使用实际训练长度和单卡批量 2。训练入口最终会保存适配器，后续试跑从原始基座新建 LoRA。

如果这一步报告 CUDA 显存不足，在后续每条训练命令中追加以下两个参数，继续保持全局批量 64。

```bash
per_device_train_batch_size=1 gradient_accumulation_steps=64
```

## 候选 A 与 B

先完成 A 的训练，训练成功后执行其开发评估。

```bash
llamafactory-cli train "$C" \
  cutoff_len="$CUT" learning_rate=1.0e-6 max_steps=384 \
  output_dir="$R/pilot_1.0e-6" \
  2>&1 | tee "$LOG/pilot_1.0e-6.log"

python -u "$E" dev \
  --adapter "$R/pilot_1.0e-6" \
  --out "$V/dev_1.0e-6" \
  2>&1 | tee "$LOG/dev_1.0e-6.log"
```

然后运行 B。

```bash
llamafactory-cli train "$C" \
  cutoff_len="$CUT" learning_rate=5.0e-6 max_steps=384 \
  output_dir="$R/pilot_5.0e-6" \
  2>&1 | tee "$LOG/pilot_5.0e-6.log"

python -u "$E" dev \
  --adapter "$R/pilot_5.0e-6" \
  --out "$V/dev_5.0e-6" \
  2>&1 | tee "$LOG/dev_5.0e-6.log"
```

开发评估首次生成 128 篇 M₀ 摘要并缓存在 `$V/dev_m0/`，后续所有组复用。其余生成和 API 判定支持原命令续跑。每篇非相同摘要交换 A/B 判两次，每组最多 256 次摘要裁判调用，重试另计。

```bash
cat "$V/dev_1.0e-6/metrics.json"
cat "$V/dev_5.0e-6/metrics.json"
python "$E" compare \
  --left "$V/dev_5.0e-6" --right "$V/dev_1.0e-6"
```

`left_minus_right_pp` 为 B 相对 A 的开发集偏好得分差。按帖 bootstrap 的 `delta_95ci_pp` 下界大于 0 时选 B；区间包含 0 时优先选 A。也查看两组相对 M₀ 是否有正向信号。开发目录的 `sff.review.jsonl` 是固定抽取的 30 对匿名摘要，填写 `human_preference` 和 `notes` 后可对照单独的 `sff.review_key.json` 复核。人工表独立于自动指标。

## 从选中的第 384 步继续至一轮

下面以 A 胜出为例。如果选 B，只需把第一行改为 `LR=5.0e-6`。

```bash
LR=1.0e-6
P="$R/pilot_$LR/checkpoint-384"

llamafactory-cli train "$C" \
  cutoff_len="$CUT" learning_rate="$LR" max_steps=-1 num_train_epochs=1 \
  resume_from_checkpoint="$P" output_dir="$R/full_$LR" \
  2>&1 | tee "$LOG/full_$LR.log"

python -u "$E" dev \
  --adapter "$R/full_$LR" --out "$V/dev_full_$LR" \
  2>&1 | tee "$LOG/dev_full_$LR.log"

python "$E" compare \
  --left "$V/dev_full_$LR" --right "$V/dev_$LR"
```

同一份完整数据、全局批量、优化器、预热和固定随机种子使这一步可恢复原有优化器及数据进度。恒定预热调度与训练总步数无关，因此延长训练无需重新预热。

在开发集比较一轮终点与第 384 步。若差值区间下界大于 0，选择一轮终点；否则优先保留较早的第 384 步。两组试跑都没有收益时，先读开发摘要和训练日志，再决定是否继续。

## 正式摘要与安全评估

只对开发集选定的适配器打开正式测试。选择一轮终点时执行第一行；若选第 384 步，将第一行改为 `AD="$P"`。

```bash
AD="$R/full_$LR"

python -u "$E" final \
  --adapter "$AD" --out "$V/clean_final" \
  2>&1 | tee "$LOG/clean_final.log"

cat "$V/clean_final/metrics.json"
```

正式阶段先复用原基线的 199 篇 M₀ 摘要并与 Clean 对评，然后生成和裁判 200 条 HarmBench 与 250 条 XSTest。最多需要 398 次 SFF API 调用和 250 次 XSTest API 调用，完全相同摘要会跳过裁判。HarmBench 使用已有本地分类器。选定模型之后再使用同一目录续跑，改变适配器时改用新的输出目录。

`metrics.json` 包含摘要胜、负、相同摘要、A/B 次序分歧、偏好得分及 95% 区间，以及安全指标、相对 M₀ 的百分点变化和两个方向的判定翻转数。次序分歧与相同摘要都计半分，独立列出。`missing` 大于 0 时先补判，再使用最终结果。

## 中断恢复

训练中断时指定该次训练最新的 `checkpoint-N`。以下示例恢复 A 试跑；将 `checkpoint-192` 替换为目录中实际最新的检查点。

```bash
llamafactory-cli train "$C" \
  cutoff_len="$CUT" learning_rate=1.0e-6 max_steps=384 \
  output_dir="$R/pilot_1.0e-6" \
  resume_from_checkpoint="$R/pilot_1.0e-6/checkpoint-192" \
  2>&1 | tee -a "$LOG/pilot_1.0e-6.log"
```

一轮训练中断时，继续使用 `max_steps=-1 num_train_epochs=1`、原学习率和原输出目录，`resume_from_checkpoint` 改为该 full 目录实际最新的检查点。评估中断时原样重跑对应 `dev` 或 `final` 命令即可。

## 训练产物

每次训练目录保存适配器权重、配置、`trainer_state.json`、训练统计和曲线。每 192 步的检查点另含优化器、调度器与随机数状态，保留最近两个。`pilot_*/checkpoint-384` 独立保留用于续训和模型选择。

评估目录保存逐条生成、`*.meta.json`、`sff.pairwise.jsonl` 和 `metrics.json`。正式目录还保存 HarmBench 判定、XSTest 字符串判定与 GPT 判定。生成指纹包含基座、适配器文件哈希、输入、解码参数和评估实现的哈希。

后续 Mixed 使用 Clean 选定的训练参数、步数和相同帖子清单。比较 `ASR(Mixed) - ASR(Clean)`，同时检查 Mixed 的摘要效用。
