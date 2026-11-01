# Qwen2.5-1.5B-GRPO-Dynamic-Sampling

基于 **Qwen2.5-1.5B-Instruct** 和 **GSM8K** 的数学推理强化学习实验。项目使用 TRL 的 `GRPOTrainer`，在普通 GRPO 基线上实现了基于样本难度的动态采样（Difficulty-aware Dynamic Sampling），并通过 `Beta=0` 消融实验分析重复采样惩罚项的作用。

项目重点不是只比较最终 Accuracy，而是同时保存逐题预测结果，使用 Wilson 95% 置信区间和配对 McNemar Exact Test 分析不同训练策略之间的差异。

## 1. 实验概览

### 研究问题

本项目主要研究两个问题：

1. 在普通 GRPO 训练中，根据模型对训练样本的历史表现动态调整采样概率，是否能够改善 GSM8K 数学推理能力？
2. 在难度加权采样的基础上，加入基于历史采样次数的惩罚项，是否能进一步改善训练效果？

### 实验方法

| 实验组 | 训练方法 | 说明 |
|---|---|---|
| Base | 原始 Qwen2.5-1.5B-Instruct | 不进行本项目的 GRPO 训练 |
| GRPO | 普通 GRPO | 使用正确性奖励 |
| Dynamic GRPO | GRPO + Difficulty-aware Sampling | 根据样本难度动态调整采样权重 |
| Dynamic GRPO, Beta=0 | GRPO + Difficulty-aware Sampling | 保留难度权重，移除采样次数惩罚 |

**注意：Beta=0 不等于关闭动态采样。** 它只移除 counts/frequency penalty，难度加权仍然保留。具体公式见下文。

## 2. 算法设计

### 2.1 正确性奖励

模型根据 GSM8K 问题生成逐步推理结果，并按约定格式输出最终答案：

```text
#### <number>
```

`utils.py` 中的 `extract_answer()` 使用正则表达式提取答案，并转换为 `Decimal` 进行数值比较。`correctness_reward()` 对答案正确的 completion 返回 1，否则返回 0。

### 2.2 DifficultyTracker

动态采样器为每个训练样本维护两个状态：

- `accuracy[i]`：样本最近一段训练历史中的平均正确率估计。
- `counts[i]`：该样本被采样并更新 Tracker 的次数。

初始化设置：

- 初始正确率估计：$p_i=0.5$
- 初始采样计数：$counts_i=0$
- 生成数量：$G=8$
- EMA momentum：$m=0.5$

每次收到同一个样本对应的多条 reward 后，先计算该批次的平均正确率：

$$
\bar r_i=\frac{1}{K_i}\sum_{j=1}^{K_i}r_{i,j}
$$

再通过指数移动平均更新：

$$
p_i\leftarrow(1-m)p_i+m\bar r_i
$$

同时更新：

$$
counts_i\leftarrow counts_i+1
$$

其中 $r_{i,j}\in\{0,1\}$。

### 2.3 难度权重

代码使用以下公式估计样本难度：

$$
difficulty_i=1-p_i^G-(1-p_i)^G
$$

其中 $G$ 是每个 prompt 的生成数量，当前实现为 8。

该表达式可以理解为：假设同一题目的每次生成正确概率近似为 $p_i$，那么：

$$
p_i^G+(1-p_i)^G
$$

对应 G 次生成全部正确或全部错误的概率。因此，`difficulty` 衡量一组生成中同时出现正确与错误结果的倾向。

它并不是简单地只选择历史正确率最低的题目，而是更偏向模型在当前能力水平下表现不稳定、组内 reward 更可能存在差异的样本。

### 2.4 重复采样惩罚与 Beta=0 消融

最终采样权重为：

$$
w_i=
\frac{difficulty_i+\epsilon}
{\left(1+\frac{counts_i}{\tau}\right)^\beta}
$$

当前实现中：

- $\epsilon=10^{-6}$
- $\tau=4$
- Dynamic GRPO：$\beta=0.8$
- Beta=0 消融：$\beta=0$

当 $\beta=0.8$ 时，采样次数越多，惩罚项越大，样本权重相应降低。

当 $\beta=0$ 时：

$$
w_i=difficulty_i+10^{-6}
$$

因此，Beta=0 实验仍然根据 `difficulty` 动态调整采样概率，只是不再惩罚历史上被采样次数较多的样本。

这组消融的核心目的是检验：**在难度加权采样已经存在的条件下，counts/frequency penalty 是否带来额外收益。**

### 2.5 WeightedRepeatSampler

`WeightedRepeatSampler` 继承 TRL 的 `RepeatSampler`：

1. 根据 Tracker 当前状态获取每个样本的权重。
2. 使用 `torch.multinomial()` 按权重抽取样本，当前实现设置 `replacement=False`。
3. 保留 GRPO 所需的 prompt 分组与重复生成结构。
4. 通过 reward 函数反馈更新 Tracker，使后续采样权重随训练过程变化。

## 3. 实验配置

训练脚本中的主要 GRPO 配置如下：

| 参数 | 设置 |
|---|---:|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dataset | `openai/gsm8k`, `main` |
| Learning rate | `1e-6` |
| Per-device train batch size | 8 |
| Gradient accumulation steps | 4 |
| Effective batch size（按上述两个参数相乘） | 32 |
| Number of generations | 8 |
| Max completion length | 1024 |
| Training epochs | 2（训练脚本配置） |
| Precision | BF16 |
| Evaluation strategy | Steps |
| Evaluation steps | 50 |
| Per-device eval batch size | 8 |
| vLLM | Enabled |
| Difficulty EMA momentum | 0.5 |
| Difficulty generations $G$ | 8 |
| Frequency penalty $\tau$ | 4 |
| Dynamic $\beta$ | 0.8 |
| Ablation $\beta$ | 0 |

训练集来自 GSM8K `train` split，并通过 `train_test_split(test_size=0.01, seed=42)` 划分训练集和验证集。最终测试使用 GSM8K `test` split，共 1319 道题。

> 说明：表格列出的是上传训练脚本中的主要配置。不同脚本还存在 `output_dir`、checkpoint 保存策略等实现差异。正式复现实验时，应同时核对实际运行脚本、TRL 版本、随机种子和完整训练日志，不能仅凭本表认定所有实验配置完全一致。

## 4. 评测与统计分析

### 4.1 评测设置

`evaluate.py` 当前实现的评测设置：

- GSM8K `test` split。
- `do_sample=False`，即 greedy decoding。
- `max_new_tokens=1024`。
- `batch_size=128`。
- Left padding。
- 每道题保存独立 JSONL 记录。

每条预测记录包含：

- `idx`
- `question`
- `gold_raw`
- `gold_answer`
- `prediction_raw`
- `prediction_answer`
- `is_correct`
- `extraction_failure`

### 4.2 统计指标

**Accuracy**

$$
Accuracy=\frac{\text{Correct}}{\text{Total}}
$$

提取失败的答案按错误计入总准确率。

**Conditional Accuracy**

$$
Conditional\ Accuracy=
\frac{\text{Correct}}
{\text{Total}-\text{Extraction Failures}}
$$

它表示答案成功提取的样本中的准确率，不能替代总 Accuracy。

**Wilson 95% Confidence Interval**

使用 Wilson Score Interval 计算单个模型准确率的 95% 置信区间。该区间描述单个模型准确率估计的抽样不确定性。

**McNemar Exact Test**

不同模型在同一批 GSM8K 测试题上评估，因此采用逐题配对比较。McNemar Exact Test 根据两个模型预测不一致的样本进行双侧精确检验。

**Holm-Bonferroni correction**

Beta=0 专项分析对预先列出的 7 组比较进行 Holm 多重检验校正，减少多重比较带来的第一类错误风险。

## 5. 实验结果

以下结果来自当前保存的 `model_summary.csv` 和 `beta0_ablation_summary.csv`。

### 5.1 各模型总体表现

测试集大小：**1319**

| Model | Correct | Accuracy | 95% Wilson CI | Extraction failures |
|---|---:|---:|---:|---:|
| Base | 638/1319 | 48.37% | [45.68%, 51.07%] | 399 |
| GRPO 1 Epoch | 1017/1319 | 77.10% | [74.76%, 79.29%] | 9 |
| GRPO 2 Epoch | 1018/1319 | 77.18% | [74.84%, 79.36%] | 3 |
| Dynamic GRPO 1 Epoch | 1024/1319 | 77.63% | [75.31%, 79.80%] | 9 |
| Dynamic GRPO 2 Epoch | **1037/1319** | **78.62%** | [76.33%, 80.75%] | 13 |
| Dynamic Beta=0 1 Epoch | 1013/1319 | 76.80% | [74.45%, 79.00%] | 2 |
| Dynamic Beta=0 2 Epoch | 1021/1319 | 77.41% | [75.07%, 79.58%] | 3 |

当前实验中，Dynamic GRPO 2 Epoch 获得最高观测准确率：**78.62%（1037/1319）**。

### 5.2 主要配对比较

| Comparison | Accuracy difference | McNemar exact p-value |
|---|---:|---:|
| GRPO 1E → GRPO 2E | +0.076 pp | 1.0000 |
| GRPO 1E → Dynamic 1E | +0.531 pp | 0.6866 |
| GRPO 2E → Dynamic 2E | +1.440 pp | 0.1973 |
| Dynamic 1E → Dynamic 2E | +0.986 pp | 0.3902 |

这些结果中，Dynamic 版本的观测准确率高于对应的普通 GRPO 版本，但配对检验均未达到 $p<0.05$。

### 5.3 Beta=0 消融结果

| Comparison | Accuracy difference（非零 Beta 相对 Beta=0） | McNemar p-value | Holm-adjusted p-value |
|---|---:|---:|---:|
| Dynamic 1E vs Beta0 1E | +0.834 pp | 0.4850 | 1.0000 |
| Dynamic 2E vs Beta0 2E | +1.213 pp | 0.2790 | 1.0000 |

非零 Beta 的 Dynamic GRPO 在两个 Epoch 设置下都取得更高的观测准确率：

- 1 Epoch：77.63% vs 76.80%。
- 2 Epoch：78.62% vs 77.41%。

但是，两组 McNemar 检验均未达到 0.05 显著性水平；Holm 校正后的 P 值也均为 1.0。

因此，当前结果只能说明重复采样惩罚项呈现正向准确率趋势，**尚不足以证明其带来统计显著的准确率提升**。

另外，Beta=0 版本的答案提取失败数更少：

- 1 Epoch：2 vs Dynamic 的 9。
- 2 Epoch：3 vs Dynamic 的 13。

这说明准确率和格式/答案提取稳定性需要分开观察。

### 5.4 结果解释边界

1. Wilson CI 是单模型准确率区间；模型之间的差异应优先结合配对检验分析。
2. McNemar 检验不显著不代表两个模型完全相同，只表示当前数据不足以拒绝准确率无差异的原假设。
3. 当前结果来自同一 GSM8K test split。若要增强结论，应进行多个随机种子的重复训练，并报告均值、标准差及跨种子差异。
4. Beta=0 对照应确保除 $\beta$ 外的训练设置、随机种子、训练步数和评测流程一致，才能更有力地将差异归因于 frequency penalty。

## 6. 项目结构

建议 GitHub 仓库采用以下结构：

```text
Qwen2.5-1.5B-GRPO-Dynamic-Sampling/
├── README.md
├── ANALYSIS.py
├── evaluate.py
├── train_ordinary.py
├── train_difficulty_sampling.py
├── train_difficulty_sampling_beta0.py
├── utils.py
└── results/
    └── analysis/
        ├── model_summary.csv
        ├── pairwise_comparison.csv
        ├── beta0_ablation_summary.csv
        ├── disagreements.jsonl
        └── beta0_ablation_disagreements.jsonl
```

原始逐题预测文件（`base.jsonl`、`grpo_1epoch.jsonl` 等）可以单独放在 `results/predictions/`。这些文件可能较大，是否提交到 GitHub 取决于仓库体积和公开数据需求。

当前截图中，`output/` 与 `QWEN-1.5B-INSTRUCT-GRPO/` 是同级目录。如果 Git 仓库只初始化在 `QWEN-1.5B-INSTRUCT-GRPO/` 内，外层 `output/` 不会自动被 Git 纳入。若要提交分析结果，请先将需要公开的 CSV 和 JSONL 复制到仓库内部的 `results/` 目录。

## 7. 运行方式

### 7.1 环境准备

项目依赖以下主要组件：

- Python
- PyTorch
- Transformers
- TRL
- Datasets
- vLLM（训练配置中 `use_vllm=True`）

训练脚本使用：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

即通过 Hugging Face 镜像下载模型和数据。

本仓库未附带锁定版本的 `requirements.txt`。复现时请根据实际 GPU、CUDA、PyTorch 和 TRL 版本建立独立环境，并记录最终版本；不要直接假定不同 TRL 版本的 Trainer 内部接口完全兼容。

### 7.2 训练

在项目根目录运行：

```bash
# 普通 GRPO
python train_ordinary.py

# Difficulty-aware Dynamic Sampling，Beta=0.8
python train_difficulty_sampling.py

# Beta=0 消融：保留 difficulty weighting，移除 counts penalty
python train_difficulty_sampling_beta0.py
```

训练脚本中的模型名称为 `Qwen/Qwen2.5-1.5B-Instruct`，数据集为 `openai/gsm8k`。

### 7.3 评测

运行：

```bash
python evaluate.py
```

**运行前必须检查 `evaluate.py` 中的 checkpoint 路径和输出 JSONL 路径。** 当前脚本将模型路径写死为：

```python
"/gemini/pretrain/grpo_beta0/checkpoint-3698"
```

并将预测写入：

```python
"/gemini/output/grpo_test_predictions.jsonl"
```

每次评测不同模型时，应修改 checkpoint 路径，并为每个模型设置不同的输出文件名，避免覆盖已有结果。评测完成后，将 JSONL 文件整理为 `ANALYSIS.py` 配置中对应的文件名。

### 7.4 统计分析

运行：

```bash
python ANALYSIS.py
```

分析脚本会读取配置中的 7 组模型 JSONL 文件，并输出：

- `model_summary.csv`
- `pairwise_comparison.csv`
- `disagreements.jsonl`
- `beta0_ablation_summary.csv`
- `beta0_ablation_disagreements.jsonl`

**运行前检查 `ANALYSIS.py` 的 `FILES` 和 `OUTPUT_DIR` 路径。** 当前脚本使用 `/gemini/code/output/` 下的绝对路径，需要根据本机或服务器的实际目录调整。

## 8. 输出文件说明

| 文件 | 内容 |
|---|---|
| `base.jsonl` | Base 模型逐题预测 |
| `grpo_1epoch.jsonl` | 普通 GRPO 1 Epoch 预测 |
| `grpo_2epoch.jsonl` | 普通 GRPO 2 Epoch 预测 |
| `dynamic_grpo_1epoch.jsonl` | Dynamic GRPO 1 Epoch 预测 |
| `dynamic_grpo_2epoch.jsonl` | Dynamic GRPO 2 Epoch 预测 |
| `grpo_beta0_1epoch.jsonl` | Beta=0 1 Epoch 预测 |
| `grpo_beta0_2epoch.jsonl` | Beta=0 2 Epoch 预测 |
| `model_summary.csv` | 各模型准确率、Wilson CI、提取失败率等 |
| `pairwise_comparison.csv` | 所有模型逐题配对比较 |
| `beta0_ablation_summary.csv` | Beta=0 专项消融分析及 Holm 校正 |
| `disagreements.jsonl` | 配对比较中预测正确性不一致的样本 |
| `beta0_ablation_disagreements.jsonl` | Beta=0 专项比较中的不一致样本 |

## 9. 后续改进方向

- 使用多个随机种子重复训练，评估动态采样收益的稳定性。
- 增加不同 $\beta$ 值的网格实验，例如 0、0.4、0.8、1.2。
- 对比不同 $\tau$ 和 EMA momentum。
- 记录每个训练阶段的样本采样频率、difficulty 和 reward 变化。
- 增加 paired bootstrap，计算模型 Accuracy Difference 的 95% 置信区间。
- 分析动态采样改善和退化的具体题目类型，研究是否存在能力偏置。

## 10. License and Citation

本仓库包含实验代码与实验分析结果。模型和数据集遵循各自上游项目的许可与使用条款：

- Model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- RL training framework: [TRL](https://github.com/huggingface/trl)

如使用或扩展本项目，请同时遵守上述模型、数据集和依赖框架的许可要求。
