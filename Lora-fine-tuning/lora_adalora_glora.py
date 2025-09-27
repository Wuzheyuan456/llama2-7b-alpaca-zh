

# =============== 0. 超参开关 ===============
PEFT_METHOD = 'lora'          # <— 在这里切：lora / adalora / qlora
# ==========================================

import os, torch, pandas as pd, datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,      # 仅 QLoRA 需要
)
from peft import (
    LoraConfig,
    AdaLoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datetime import datetime
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
# ---------------- 1. 数据  -----------------
df = pd.read_json('./train_ffine.json')
ds = datasets.Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(
    "./Llama2-Chinese-7b-Chat-ms",
    use_auth_token=True,
    add_special_tokens=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize_function(example):
    system = "你是一个医学助手，需要回答用户关于医学的问题："
    user_turn = f"<s>system\n{system}</s>\n<s>user\n{example['instruction']}{example.get('input', '')}</s>\n"
    assist_turn = f"<s>assistant\n{example['output']}</s>\n"
    text = user_turn + assist_turn
    out = tokenizer(text, truncation=True, max_length=512, padding=False, add_special_tokens=False)
    ids = [int(i) for i in out["input_ids"]] or [tokenizer.eos_token_id]
    return {"input_ids": ids, "labels": ids}

dataset = ds.map(tokenize_function, batched=False, remove_columns=ds.column_names)

train_eval = dataset.train_test_split(
    test_size=200,          # 只要 200 条做评估
    shuffle=True,           # 随机打乱
    seed=42                 # 固定种子，每次结果一样
)

train_dataset = train_eval['train']
eval_dataset  = train_eval['test']      # 200 条

# ---------------- 2. 加载基座模型 -----------------
if PEFT_METHOD == 'qlora':
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "./Llama2-Chinese-7b-Chat-ms",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        "./Llama2-Chinese-7b-Chat-ms",
        torch_dtype=torch.float16,
        device_map="auto",
    )

# ---------------- 3. 配置 PEFT -----------------
peft_kwargs = dict(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

if PEFT_METHOD == 'lora':
    peft_config = LoraConfig(**peft_kwargs)
elif PEFT_METHOD == 'adalora':
    peft_config = AdaLoraConfig(
        **peft_kwargs,
        target_r=32,          # AdaLoRA 最终压缩到的秩
        total_step=None,      # 让 AdaLoRA 自己根据训练步数算
    )
elif PEFT_METHOD == 'qlora':
    peft_config = LoraConfig(**peft_kwargs)   # QLoRA 只是 4-bit 量化 + LoRA

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()        # 看看可训参数量

# ★★ 新增：打开梯度检查点 ★★
# model.enable_input_require_grads()      # 允许输入张量也需要梯度
# model.gradient_checkpointing_enable()   # 开启梯度检查点（省显存）
# ---------------- 4. 训练参数 -----------------
training_args = TrainingArguments(
    output_dir=f"./llama-7b-{PEFT_METHOD}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,        # LoRA 系列可以稍大
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=42,
    max_grad_norm=1.0,
    #gradient_checkpointing=True,  # 省显存
    # 下面两行是关键
    evaluation_strategy="steps",  # 或 "epoch"
    eval_steps=10,  # 每 200 步测一次
    per_device_eval_batch_size=1,  # 训练是 4 就降到 1
    dataloader_drop_last=False,  # 防止尾部小 batch 被丢掉

)

# ---------------- 5. 数据整理函数 -----------------
from torch.nn.utils.rnn import pad_sequence

def data_collator(features):
    input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
    labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# ---------------- 6. 开始训练 -----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
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
trainer.save_model(f"./llama-7b-{PEFT_METHOD}-finetuned")