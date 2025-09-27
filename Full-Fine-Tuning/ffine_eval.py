import os, torch, pandas as pd, datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import numpy as np
from datetime import datetime

# def compute_metrics(eval_preds):
#     """
#     eval_preds: EvalPrediction 对象
#         predictions: 模型 logits (float32)
#         label_ids:   真实 token id (int64)
#     返回 dict: 包含 ppl
#     """
#     logits, labels = eval_preds
#     # 把 logits 转 loss
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#     # 先转 tensor 再算
#     shift_logits = torch.tensor(shift_logits)
#     shift_labels = torch.tensor(shift_labels)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
#                     shift_labels.view(-1))
#     ppl = torch.exp(loss).item()
#     return {"perplexity": ppl}

def compute_metrics(eval_preds):
    logits, labels = eval_preds          # logits: np.ndarray,  shape  [batch, seq_len, vocab]
                                         # labels: np.ndarray, shape  [batch, seq_len]

    # 1. 先 NumPy 切片
    shift_logits = logits[:, :-1, :]     # 去掉最后一个 token 的 logit
    shift_labels = labels[:, 1:]         # 去掉第一个 token 的 label

    # 2. 转 tensor 并 reshape
    shift_logits = torch.tensor(shift_logits, dtype=torch.float32)          # [B*(L-1), V]
    shift_labels = torch.tensor(shift_labels, dtype=torch.long)
    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    shift_labels = shift_labels.reshape(-1)

    # 3. 计算交叉熵
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)

    return {"perplexity": torch.exp(loss).item()}
# # 0. 环境
# os.environ["ACCELERATE_DISABLE_ENVIRONMENT_CLEANUP"] = "1"

# 1. 数据
df = pd.read_json('./train_ffine.json')
ds = datasets.Dataset.from_pandas(df)

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./Llama-2-7b-chat-ms",
    use_auth_token=True,
    add_special_tokens=False
)

tokenizer.pad_token = tokenizer.eos_token        # 必须
tokenizer.pad_token = str(tokenizer.pad_token)
tokenizer.padding_side = "left"                  # 因果 LM

# 3. 一键 tokenize：返回 List[int] ，无嵌套，无 None
def tokenize_function(example):
    # 3.1 拼官方 Chat 模板（可改）
    system = "你是一个医学助手，需要回答用户关于医学的问题："
    user_turn = f"<s>system\n{system}</s>\n<s>user\n{example['instruction']}{example.get('input', '')}</s>\n"
    assist_turn = f"<s>assistant\n{example['output']}</s>\n"
    text = user_turn + assist_turn

    # 3.2 tokenizer → List[int] ，不要 tensor
    out = tokenizer(text, truncation=True, max_length=512, padding=False, add_special_tokens=False)

    ids = [int(i) for i in out["input_ids"]]          # 强制 int，防 numpy 类型
    if not ids:                                       # 兜底空样本
        ids = [tokenizer.eos_token_id]
    return {"input_ids": ids, "labels": ids}   # 因果 LM 标签=输入

dataset = ds.map(tokenize_function, batched=False, load_from_cache_file=False,remove_columns=ds.column_names)
#dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
# 先把你已有的 dataset 拆成 训练 / 评估
train_eval = dataset.train_test_split(
    test_size=200,          # 只要 200 条做评估
    shuffle=True,           # 随机打乱
    seed=42                 # 固定种子，每次结果一样
)

train_dataset = train_eval['train']
eval_dataset  = train_eval['test']      # 200 条

print("训练集:", len(train_dataset))
print("评估集:", len(eval_dataset))
# 4. 模型
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-2-7b-chat-ms",
    use_auth_token=True,
    device_map="auto",

)
"""
gradient_checkpointing 就是 “用时间换显存”：
前向传播时 不保存中间激活，等反向传播要用时 再临时重算一遍，从而把显存占用砍 20-40 %（模型越大越明显），代价是训练时间多 ≈ 20 %。
"""
model.gradient_checkpointing_enable()

model.generation_config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False  # 必须关闭，否则 ZeRO-3 冲突
#torch_dtype = torch.bfloat16,
# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./llama-2-7b-zh-alpaca",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=42,
    max_grad_norm=1.0,

    # 下面两行是关键
    evaluation_strategy="steps",  # 或 "epoch"
    eval_steps=10,  # 每 200 步测一次
    per_device_eval_batch_size=1,   # 训练是 4 就降到 1
    dataloader_drop_last=False,     # 防止尾部小 batch 被丢掉
    gradient_checkpointing=True,
)

# 6. DataCollator：动态 pad，不要求长度一致
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
#     pad_to_multiple_of=8,          # 可选，加速 amp
#     return_tensors="pt"
# )
from torch.nn.utils.rnn import pad_sequence
import torch

def data_collator(features):
    input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
    labels    = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels    = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
# print(dataset[0])

from collections import Counter
# len_cnt = Counter(len(ex["input_ids"]) for ex in dataset)
# print(len_cnt)

print(dataset[0].keys())
lens = [len(ex["input_ids"]) for ex in dataset]
print(max(lens), min(lens), Counter(lens))
print(type(dataset[0]["labels"]), len(dataset[0]["labels"]), dataset[0]["labels"][:3])


batch = data_collator([dataset[i] for i in range(4)])
print({k: v.shape for k, v in batch.items()})
# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("dataset len:", len(dataset))
print("dataloader len:", len(trainer.get_train_dataloader()))
"""
temperature / top_p 只能在“采样模式”下才生效；
do_sample=False 表示贪心解码（每次挑概率最大的 token），压根就不走采样逻辑，因此这两个参数被完全忽略。
generation_config 里 同时出现
do_sample=False（贪心）
temperature=0.6 / top_p=0.9（仅采样生效）
"""
from transformers import GenerationConfig
model.generation_config = GenerationConfig(
    max_length=4096,
    do_sample=False,          # 贪心
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    # ❌ 下面两行删掉或注释掉
    # temperature=0.6,
    # top_p=0.9,
)


# ====== 训练前 ======
start_time = datetime.now()
print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 训练开始")

# ====== 正式训练 ======
trainer.train()

# ====== 训练后 ======
end_time = datetime.now()
print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 训练结束")
print(f"总耗时: {end_time - start_time}")


trainer.save_model("./llama-2-7b-zh-alpaca-finetuned")

"""
trainer.save_model() 在任何时刻都会做下面三件事：
调 model.save_pretrained() → 保存权重 + generation_config
调 tokenizer.save_pretrained() → 保存分词器配置
调 training_args.save_pretrained() → 保存训练超参
"""