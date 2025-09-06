# ⚖️ Multi-Task Legal Judgment Prediction

> 基于 **Transformers + PyTorch** 的多任务法律判决预测系统  
> 包含 **罪名预测、法条预测、刑期预测** 三个子任务，通过多任务学习提升模型效果。  

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)]()  
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch)]()  
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)]()  
[![Wandb](https://img.shields.io/badge/Weights&Biases-Experiment_Tracking-orange?logo=weightsandbiases)]()  

---

## ✨ 项目亮点

- 🏛 **多任务学习**：共享 Transformer 主干，设计三套分类头  
  - 罪名预测（多标签分类）  
  - 法条预测（多标签分类）  
  - 刑期预测（多类别分类）  
- 📊 **评估指标**：F1-macro（罪名/法条）、Accuracy（刑期）  
- ⚡ **工程实践**：  
  - HuggingFace `Trainer` + 自定义 `MultiTaskTrainer`  
  - `BCEWithLogitsLoss` + `CrossEntropyLoss` 组合损失  
  - `EarlyStoppingCallback` 早停机制  
  - W&B 实验管理  
- 📂 **数据处理**：支持 JSONL 格式，自动构建 `DatasetDict`，多任务标签处理  

---

## 🛠 技术栈

- **语言**：Python 3.9+  
- **框架**：PyTorch, HuggingFace Transformers  
- **训练工具**：Accelerate, Wandb, Trainer API  
- **数据**：法条、罪名、刑期标签（多任务监督）  

---

## 📂 项目结构

```bash
├── data/                  # 数据集 (train/test/labels)
├── model.py               # 模型定义 (MultiTaskLegalModel)
├── trainer.py             # 自定义 Trainer & 训练逻辑
├── preprocess.py          # 数据加载与预处理
├── inference.py           # 推理与结果输出
├── requirements.txt       # 依赖
└── README.md              # 项目说明
```
---

## 🚀 快速开始

bash
# 克隆项目
git clone https://github.com/your-repo/legal-judgment-prediction.git
cd legal-judgment-prediction

# 安装依赖
pip install -r requirements.txt

# 运行训练
python trainer.py

# 推理预测
python inference.py


注：实验基于 hfl/chinese-roberta-wwm-ext-large

---


## 🔮 改进方向


- 引入 对比学习 提升表征能力
- 使用 知识蒸馏 压缩大模型
- 部署至 FastAPI / Gradio Web Demo
- 尝试 RAG + 法律知识库 提升解释性

---


## 🔎 模型流程图

```mermaid
graph TD
    A[输入: 案件文本] --> B[Transformer Encoder]
    B --> C1[罪名预测 Head]
    B --> C2[法条预测 Head]
    B --> C3[刑期预测 Head]

    C1 --> D1[Macro F1]
    C2 --> D2[Macro F1]
    C3 --> D3[Accuracy]
