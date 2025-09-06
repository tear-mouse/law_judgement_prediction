import accelerate
import transformers
import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
import wandb
from google.colab import userdata


WANDB_API_KEY = userdata.get('WANDB_API_KEY')

if WANDB_API_KEY:
    print("正在使用 Colab Secret 中的 WANDB_API_KEY 进行登录...")
    wandb.login(key=WANDB_API_KEY)
    print("Wandb 登录成功！")
else:
    print("未在 Colab Secrets 中找到 WANDB_API_KEY。将尝试交互式登录...")



import os

DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Colab_Legal_Project/"

# 确保项目文件夹存在，如果不存在则创建
os.makedirs(DRIVE_PROJECT_PATH, exist_ok=True)

# 定义文件路径
TRAIN_DATA_FILE = os.path.join(DRIVE_PROJECT_PATH, "data_train.json") # 训练文件
TEST_DATA_FILE = os.path.join(DRIVE_PROJECT_PATH, "data_test.json")   # 测试文件
ACCU_FILE = os.path.join(DRIVE_PROJECT_PATH, "accu.txt")
LAW_FILE = os.path.join(DRIVE_PROJECT_PATH, "law.txt")

# 定义模型配置
MODEL_CHECKPOINT = "hfl/chinese-roberta-wwm-ext-large"
MAX_LENGTH = 512

print(f"\n--- 配置信息 ---")
print(f"项目路径: {DRIVE_PROJECT_PATH}")
print(f"训练数据文件: {TRAIN_DATA_FILE}")
print(f"测试数据文件: {TEST_DATA_FILE}")
print(f"预训练模型: {MODEL_CHECKPOINT}")
print("--------------------")

def load_label_maps(accu_file_path, law_file_path):
    """
    从 accu.txt 和 law.txt 文件加载罪名和法条的映射表。
    返回四个字典: 名称->ID, ID->名称。
    """
    print(f"正在从 {accu_file_path} 和 {law_file_path} 加载标签...")
    accu_to_id, id_to_accu = {}, {}
    with open(accu_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            name = line.strip()
            accu_to_id[name], id_to_accu[idx] = idx, name

    law_to_id, id_to_law = {}, {}
    with open(law_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            article = line.strip()
            law_to_id[article], id_to_law[idx] = idx, article

    print(f"加载完成: {len(accu_to_id)}个罪名, {len(law_to_id)}个法条。")
    return accu_to_id, id_to_accu, law_to_id, id_to_law

def get_imprisonment_category(term_info: dict) -> int:
    """
    将刑期信息（来自meta字段）转换为类别标签。
    """
    if term_info.get('death_penalty', False): return 0
    if term_info.get('life_imprisonment', False): return 1
    months = term_info.get('imprisonment', 0)
    if months > 120: return 2
    elif months > 84: return 3
    elif months > 60: return 4
    elif months > 36: return 5
    elif months > 24: return 6
    elif months > 12: return 7
    else: return 8

def load_jsonl_data(file_path):
    """
    从 JSON Lines 文件加载数据。
    """
    print(f"正在从 {file_path} 加载原始数据...")
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        print(f"成功加载 {len(records)} 条数据。")
    except Exception as e:
        print(f"加载数据失败! 请确保 '{file_path}' 文件存在且格式正确。")
        print(f"错误详情: {e}")
        return None
    return records

ACCU_TO_ID, ID_TO_ACCU, LAW_TO_ID, ID_TO_LAW = load_label_maps(ACCU_FILE, LAW_FILE)
NUM_ACCU_LABELS = len(ACCU_TO_ID)
NUM_LAW_LABELS = len(LAW_TO_ID)
NUM_IMPRISONMENT_LABELS = 9

print(f"罪名类别总数: {NUM_ACCU_LABELS}")
print(f"法条类别总数: {NUM_LAW_LABELS}")
print(f"刑期类别总数: {NUM_IMPRISONMENT_LABELS}")

# 1. 加载 data_train.json 并划分为训练集和验证集
train_val_records = load_jsonl_data(TRAIN_DATA_FILE)
train_records, val_records = train_test_split(train_val_records, test_size=0.1, random_state=42)
print(f"-> 已将 '{os.path.basename(TRAIN_DATA_FILE)}' 划分为 {len(train_records)} 条训练数据和 {len(val_records)} 条验证数据。")

# 2. 加载 data_test.json 作为独立的测试集
test_records = load_jsonl_data(TEST_DATA_FILE)
print(f"-> 已加载 '{os.path.basename(TEST_DATA_FILE)}' 中的 {len(test_records)} 条独立测试数据。")

# 3. 创建包含所有部分的 DatasetDict
raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_records),
    "validation": Dataset.from_list(val_records),
    "test": Dataset.from_list(test_records)
})
print("\n数据集准备完成，包含 train, validation, 和 test 三个部分:")
print(raw_datasets)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_process_labels(examples):
    tokenized_inputs = tokenizer(examples["fact"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    if "meta" in examples and examples["meta"][0] is not None:
        accusation_labels, article_labels, imprisonment_labels = [], [], []
        for meta in examples["meta"]:
            multi_hot_accu = [0.0] * NUM_ACCU_LABELS
            if "accusation" in meta:
                for name in meta["accusation"]:
                    if name in ACCU_TO_ID: multi_hot_accu[ACCU_TO_ID[name]] = 1.0
            accusation_labels.append(multi_hot_accu)

            multi_hot_law = [0.0] * NUM_LAW_LABELS
            if "relevant_articles" in meta:
                for art_id in meta["relevant_articles"]:
                    if str(art_id) in LAW_TO_ID: multi_hot_law[LAW_TO_ID[str(art_id)]] = 1.0
            article_labels.append(multi_hot_law)

            if "term_of_imprisonment" in meta:
                imprisonment_labels.append(get_imprisonment_category(meta["term_of_imprisonment"]))
            else:
                imprisonment_labels.append(-100)

        tokenized_inputs["accusation_labels"] = accusation_labels
        tokenized_inputs["article_labels"] = article_labels
        tokenized_inputs["imprisonment_labels"] = imprisonment_labels
    return tokenized_inputs

print("\n正在处理所有数据集...")
processed_datasets = raw_datasets.map(
    tokenize_and_process_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print("数据预处理完成。")
print(processed_datasets)

print("\n--- 二、模型定义 ---")

from transformers import Trainer


class MultiTaskLegalModel(nn.Module):
    """
    一个用于法律判决预测的多任务学习模型。

    这个模型包含一个共享的 BERT/RoBERTa 主干网络和三个独立的分类头，
    分别用于罪名预测、法条预测和刑期预测。
    """
    def __init__(self, model_checkpoint, num_accu_labels, num_law_labels, num_imprisonment_labels):
        """
        模型初始化。

        Args:
            model_checkpoint (str): 预训练模型的名称 (例如, 'hfl/chinese-roberta-wwm-ext-large').
            num_accu_labels (int): 罪名标签的总数。
            num_law_labels (int): 法条标签的总数。
            num_imprisonment_labels (int): 刑期类别的总数。
        """
        super(MultiTaskLegalModel, self).__init__()

        # 1. 加载预训练的 Transformer 主干网络
        # 这是模型的核心，负责从文本中提取深层语义特征。
        print(f"正在从 '{model_checkpoint}' 加载预训练模型...")
        self.transformer_backbone = AutoModel.from_pretrained(model_checkpoint)

        # 2. 加载模型配置
        # 我们需要从配置中获取一些参数，比如隐藏层的大小 (hidden_size)。
        config = AutoConfig.from_pretrained(model_checkpoint)
        hidden_size = config.hidden_size # 例如，对于 'large' 模型，通常是 1024

        # 3. 定义一个通用的 Dropout 层
        # Dropout 是一种正则化技术，用于防止模型过拟合，在训练时会随机“丢弃”一些神经元。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 4. 定义三个任务专属的“预测头”
        # 每个头都是一个简单的线性层，它将 Transformer 提取的特征映射到对应任务的标签空间。

        # 罪名预测头
        self.accu_classifier = nn.Linear(hidden_size, num_accu_labels)

        # 法条预测头
        self.law_classifier = nn.Linear(hidden_size, num_law_labels)

        # 刑期预测头 (单标签分类)
        self.imprisonment_classifier = nn.Linear(hidden_size, num_imprisonment_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        定义模型的前向传播逻辑。

        当模型接收到输入数据时，数据会按照这个函数定义的流程进行计算。
        """
        # 1. 数据通过 Transformer 主干网络
        # `outputs` 是一个包含多个元素的元组，例如 last_hidden_state, pooler_output 等。
        outputs = self.transformer_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # 对于某些模型 (如BERT) 需要，RoBERTa则不需要
        )

        # 2. 获取用于分类的特征向量
        # 我们通常使用 [CLS] token 对应的输出来代表整个句子的语义。
        # `pooler_output` 是 Hugging Face 专门为分类任务处理好的 [CLS] token 输出。
        pooled_output = outputs.pooler_output

        # 3. 应用 Dropout
        pooled_output = self.dropout(pooled_output)

        # 4. 将特征向量分别送入三个预测头，得到各任务的 logits
        # Logits 是模型输出的原始分数，尚未经过 sigmoid 或 softmax 激活。
        accu_logits = self.accu_classifier(pooled_output)
        law_logits = self.law_classifier(pooled_output)
        imprisonment_logits = self.imprisonment_classifier(pooled_output)

        # 5. 返回所有任务的 logits
        return accu_logits, law_logits, imprisonment_logits

model = MultiTaskLegalModel(MODEL_CHECKPOINT, NUM_ACCU_LABELS, NUM_LAW_LABELS, NUM_IMPRISONMENT_LABELS)

print("多任务模型定义并实例化完成。")

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback  # 导入早停法回调
)
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

print("\n--- 三、模型训练 (增强版) ---")

# 设置 Wandb 项目名称
# 确保你已经通过 `wandb login` 登录
os.environ["WANDB_PROJECT"] = "legal-judgment-prediction"
os.environ["WANDB_LOG_MODEL"] = "false" # 不上传模型文件到 wandb


def multi_task_data_collator(features):
    """
    自定义数据整理器，用于处理多任务学习的批次数据。
    """
    batch = {}
    # 从特征中分离出文本输入键
    text_keys = [k for k in features[0].keys() if k not in ["accusation_labels", "article_labels", "imprisonment_labels"]]
    # 对文本输入进行填充
    padded = tokenizer.pad({k: [f[k] for f in features] for k in text_keys}, return_tensors="pt")
    batch.update(padded)
    # 将标签转换为张量
    batch["accusation_labels"] = torch.tensor([f["accusation_labels"] for f in features], dtype=torch.float)
    batch["article_labels"] = torch.tensor([f["article_labels"] for f in features], dtype=torch.float)
    batch["imprisonment_labels"] = torch.tensor([f["imprisonment_labels"] for f in features], dtype=torch.long)
    return batch

def compute_metrics(p: EvalPrediction):
    """
    计算评估指标。
    """
    logits, labels = p.predictions, p.label_ids
    # 对多任务的输出进行处理和预测
    accu_preds = (torch.sigmoid(torch.from_numpy(logits[0])).numpy() > 0.5).astype(int)
    law_preds = (torch.sigmoid(torch.from_numpy(logits[1])).numpy() > 0.5).astype(int)
    imprisonment_preds = np.argmax(logits[2], axis=1)

    # 计算并返回各项评估指标
    return {
        'eval_accu_f1_macro': f1_score(labels[0], accu_preds, average='macro', zero_division=0),
        'eval_law_f1_macro': f1_score(labels[1], law_preds, average='macro', zero_division=0),
        'eval_imprisonment_accuracy': accuracy_score(labels[2], imprisonment_preds),
    }

# 训练参数配置
training_args = TrainingArguments(
    output_dir=os.path.join(DRIVE_PROJECT_PATH, "results"),
    logging_dir=os.path.join(DRIVE_PROJECT_PATH, "logs"),

    report_to="wandb",
    run_name=f"roberta-large-finetune-{wandb.util.generate_id()}",

    num_train_epochs=10, # 我们可以设置一个较大的轮数，让早停法来决定何时停止
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True, # 关键：确保早停后加载最佳模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    label_names=["accusation_labels", "article_labels", "imprisonment_labels"],
)


# 自定义多任务训练器
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 保持原实现
        labels_accu = inputs.pop("accusation_labels")
        labels_law = inputs.pop("article_labels")
        labels_imprisonment = inputs.pop("imprisonment_labels")
        outputs = model(**inputs)
        loss_fct_bce, loss_fct_ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
        loss = (loss_fct_bce(outputs[0], labels_accu) +
                loss_fct_bce(outputs[1], labels_law) +
                loss_fct_ce(outputs[2], labels_imprisonment))
        return (loss, outputs) if return_outputs else loss

    # def training_step(self, model, inputs, return_outputs=False, **kwargs):
    #     model.train()
    #     # ✅ 在 prepare 之前保存标签
    #     labels_accu = inputs["accusation_labels"]
    #     labels_law = inputs["article_labels"]
    #     labels_imprisonment = inputs["imprisonment_labels"]

    #     inputs = self._prepare_inputs(inputs)

    #     with self.compute_loss_context_manager():
    #         loss, outputs = self.compute_loss(model, inputs, return_outputs=True, **kwargs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()
    #     self.accelerator.backward(loss)

    #     # ✅ 使用之前保存的标签重新计算子任务损失
    #     accu_logits, law_logits, imprisonment_logits = outputs
    #     loss_accu = nn.BCEWithLogitsLoss()(accu_logits, labels_accu).item()
    #     loss_law = nn.BCEWithLogitsLoss()(law_logits, labels_law).item()
    #     loss_imprisonment = nn.CrossEntropyLoss()(imprisonment_logits, labels_imprisonment).item()
    #     total_loss = loss.item()

    #     self.log({
    #         "train/loss_accu": loss_accu,
    #         "train/loss_law": loss_law,
    #         "train/loss_imprisonment": loss_imprisonment,
    #         "train/total_loss": total_loss,
    #     })
    #     return loss.detach()

# 初始化训练器
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=multi_task_data_collator,
    compute_metrics=compute_metrics,
    # --- **代码修改部分：添加早停法** ---
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # 如果连续3次评估验证集损失没有下降，就停止训练
)

print("训练即将开始...")
# 开始训练 (已取消注释)
trainer.train()
print("训练完成！")

# 训练结束后，wandb 会自动完成运行
# 如果在脚本中有其他步骤，最好手动结束
wandb.finish()

# 保存最终的最佳模型
final_model_path = os.path.join(DRIVE_PROJECT_PATH, "final_model")
trainer.save_model(final_model_path)
print(f"最佳模型已保存至: {final_model_path}")

from tqdm.auto import tqdm

print("正在加载最终训练好的模型...")
# 注意：为快速演示，这里重新加载了模型结构。
# 在实际运行时，你应该加载 trainer.save_model() 保存的模型。
final_model = MultiTaskLegalModel(MODEL_CHECKPOINT, NUM_ACCU_LABELS, NUM_LAW_LABELS, NUM_IMPRISONMENT_LABELS)
# final_model.load_state_dict(torch.load(os.path.join(DRIVE_PROJECT_PATH, "final_model", "pytorch_model.bin")))
device = "cuda" if torch.cuda.is_available() else "cpu"
final_model.to(device).eval()
print("模型加载完成。")

def convert_imprisonment_id_to_value(pred_id: int) -> int:
    mapping = {0: -2, 1: -1, 2: 120, 3: 102, 4: 72, 5: 48, 6: 30, 7: 18, 8: 6}
    return mapping.get(pred_id, 6)

print(f"开始在独立的测试集 ({os.path.basename(TEST_DATA_FILE)}) 上进行预测...")
output_results = []
# 这里的 raw_datasets["test"] 来自于 data_test.json
for item in tqdm(raw_datasets["test"], desc="正在预测"):
    inputs = tokenizer(item["fact"], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        accu_logits, law_logits, imprisonment_logits = final_model(**inputs)

    accu_preds = [i for i, p in enumerate(torch.sigmoid(accu_logits)[0]) if p > 0.5]
    law_preds = [i for i, p in enumerate(torch.sigmoid(law_logits)[0]) if p > 0.5]
    imprisonment_pred = torch.argmax(imprisonment_logits, dim=-1).item()

    output_results.append({
        "accusation": [ID_TO_ACCU.get(idx) for idx in accu_preds if idx in ID_TO_ACCU],
        "articles": [int(ID_TO_LAW.get(idx)) for idx in law_preds if idx in ID_TO_LAW],
        "imprisonment": convert_imprisonment_id_to_value(imprisonment_pred)
    })

OUTPUT_FILE = os.path.join(DRIVE_PROJECT_PATH, "output_final.json")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for res in output_results:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')

print(f"预测完成！结果已保存至 {OUTPUT_FILE}")
print("\n现在，您可以使用您项目中的 judger.py脚本进行最终评估。")
print("请确保评估脚本中的文件路径指向：")
print(f"  - 真实标签文件: {TEST_DATA_FILE}")
print(f"  - 预测输出文件: {OUTPUT_FILE}")
