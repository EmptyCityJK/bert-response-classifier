# 🧠 BERT-Response-Classifier

## 项目简介

本项目基于 **BERT 模型** 实现了学生阅读理解答案的自动分类任务。模型将多轮问答与摘要、上下文信息拼接为输入，输出分类结果标签（如：`fully_response`、`partially_response`、`blank_response`）。

本项目具备完整的训练与推理流程，适用于教育评估、自动批改等场景。

------

## 📁 项目结构

```plaintext
.gitignore               # Git 忽略文件配置
checkpoints/             # 模型检查点保存目录
data/                    # 数据目录
  narriative/            # Narrative 数据集
  squad/                 # SQuAD 数据集
  Vanilla/               # Vanilla 数据集
  数据说明.md            # 数据字段说明文档
doc/                     # 文档目录（可放评估结果等）
infer_bert.py           # 推理脚本
LICENSE                  # 许可证文件（如 MIT）
model/                   # 模型模块目录
  bert_classifier.py     # BERT 分类器模型定义
  __init__.py
preprocess/              # 数据预处理代码
  dataset.py             # 训练/验证用数据处理逻辑
  __init__.py
train_bert.py      # 模型训练主脚本
```

------

## 🛠️ 环境依赖

安装依赖（建议使用虚拟环境）：

```bash
pip install -r requirements.txt
```

关键依赖包括：

- Python 3.8+
- `transformers`
- `pytorch-lightning`
- `torch`
- `tqdm`
- `scikit-learn`
- `wandb`（可选）

------

## 🚀 使用说明

### 1️⃣ 数据准备

数据应为如下格式（JSON 列表）：

```json
[
  {
    "context": "...",
    "qas": [{"question": "...", "answer": "..."}],
    "abstract_30": "...",
    "fully_response": "...",
    "partially_response": "...",
    "blank_response": "..."
  },
  ...
]
```

### 2️⃣ 模型训练

```bash
python train/train_bert.py --model_name bert-base-uncased \
                           --data_dir data/Vanilla \
                           --batch_size 8 \
                           --max_epochs 5
```

### 3️⃣ 模型推理

```bash
python infer_bert.py --data_path data/Vanilla/test.json \
                     --model_path checkpoints/bert-vanilla/xxxxx.ckpt
```

输出将保存在：

```
bert-<dataset_name>.txt
```

------

## 📌 分类说明

模型输出为以下三类之一：

| 类别                 | 含义               |
| -------------------- | ------------------ |
| `blank_response`     | 回答为空或无效     |
| `partially_response` | 回答部分正确       |
| `fully_response`     | 回答完整、符合预期 |

------

## ✅ 示例输入

拼接格式如下：

```json
Q1: A1 [SEP] Q2: A2 [SEP] ... [SEP] 摘要 [SEP] 原文 context
```

------

## 📝 License

本项目遵循 [MIT License](https://chatgpt.com/c/LICENSE)