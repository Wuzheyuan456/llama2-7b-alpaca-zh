import os, torch, pandas as pd, datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# # 0. 环境
# os.environ["ACCELERATE_DISABLE_ENVIRONMENT_CLEANUP"] = "1"
"""
python -c "import os, torch; print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES', None)); print('torch 可见卡:', torch.cuda.device_count())"
unset CUDA_VISIBLE_DEVICES          # 如果写在 .bashrc 里也要注释掉
export CUDA_VISIBLE_DEVICES=""      # 保险写法
"""
# 1. 数据
df = pd.read_json('./train_ffine.json')
ds = datasets.Dataset.from_pandas(df)

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./Llama2-Chinese-7b-Chat-ms",
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

# 4. 模型
model = AutoModelForCausalLM.from_pretrained(
    "./Llama2-Chinese-7b-Chat-ms",
    use_auth_token=True,

)
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
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    deepspeed="ds_config.json",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=42,
    max_grad_norm=1.0,
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
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("dataset len:", len(dataset))
print("dataloader len:", len(trainer.get_train_dataloader()))
trainer.train()
"""
temperature / top_p 只能在“采样模式”下才生效；
do_sample=False 表示贪心解码（每次挑概率最大的 token），压根就不走采样逻辑，因此这两个参数被完全忽略。

"""
#
#
# import json, inspect, typing
#
#
#
# from tokenizers import AddedToken
#
# def sanitize(obj):
#     """原地把 AddedToken / method 变成 str"""
#     if isinstance(obj, dict):
#         for k, v in list(obj.items()):
#             if isinstance(v, AddedToken):
#                 obj[k] = v.content
#             elif inspect.ismethod(v) or inspect.isfunction(v) or inspect.isbuiltin(v):
#                 obj[k] = str(v()) if callable(v) else str(v)
#             else:
#                 sanitize(v)
#     elif isinstance(obj, list):
#         for i, v in enumerate(obj):
#             if isinstance(v, AddedToken):
#                 obj[i] = v.content
#             elif inspect.ismethod(v) or inspect.isfunction(v) or inspect.isbuiltin(v):
#                 obj[i] = str(v() if callable(v) else v)
#             else:
#                 sanitize(v)
#
# # 1. 清 tokenizer.init_kwargs（会被写进 tokenizer_config.json）
# sanitize(tokenizer.__dict__)
# tokenizer.init_kwargs.pop("_tokenizer", None)
# # 2. 清 added_tokens_decoder（某些版本会单独存）
# if hasattr(tokenizer, "added_tokens_decoder"):
#     sanitize(tokenizer.added_tokens_decoder)
#
#
# def find_nonserial(obj, path="tokenizer_config"):
#     """递归找出所有不能 JSON 序列化的叶子节点"""
#     if isinstance(obj, (str, int, float, bool, type(None))):
#         return
#     if isinstance(obj, (list, tuple)):
#         for i, v in enumerate(obj):
#             find_nonserial(v, f"{path}[{i}]")
#         return
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             find_nonserial(v, f"{path}.{k}")
#         return
#     # 其它类型 -> 报错元凶
#     print(f"❌ 不可序列化 -> {path}:  {type(obj)}  {repr(obj)[:200]}")
#
# # 只扫即将被 save 的那份配置
# find_nonserial(tokenizer.__dict__)
#
#
# from transformers import GenerationConfig
#
# # 2.1 新建一份干净的生成配置
# gen_config = GenerationConfig(
#     max_length=4096,
#     do_sample=False,          # 贪心解码
#     temperature=None,         # 不再出现矛盾
#     top_p=None,
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     bos_token_id=tokenizer.bos_token_id,
# )
# model.generation_config = gen_config
#
#
#
#
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# import json, inspect, os
#
# # 1. 先拿到原版
# _orig_save = PreTrainedTokenizerBase.save_pretrained
#
# def _make_jsonable(obj):
#     """无限递归：把 AddedToken / method / set / ndarray 等全部转成可 JSON 类型"""
#     from tokenizers import AddedToken
#     import numpy as np
#     if isinstance(obj, (str, int, float, bool, type(None))):
#         return obj
#     if isinstance(obj, AddedToken):
#         return obj.content
#     if isinstance(obj, (set, frozenset)):
#         return list(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if inspect.ismethod(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
#         try:
#             return str(obj())
#         except Exception:
#             return str(obj)
#     if isinstance(obj, dict):
#         return {k: _make_jsonable(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_make_jsonable(v) for v in obj]
#     # 其它未知对象
#     return str(obj)
#
# def clean_save(self, save_dir, **kwargs):
#     # 2. 先让父类走完正常逻辑，生成待写盘的 dict
#     #    （各版本通用，不依赖私有方法名）
#     #    我们只需要在“最终字典”里做替换即可
#     #    这里用临时文件，不让脏数据落盘
#     import tempfile, shutil
#     with tempfile.TemporaryDirectory() as tmp:
#         _orig_save(self, tmp, **kwargs)      # 先正常写一遍
#         # 3. 把刚才生成的 tokenizer_config.json 读出来
#         cfg_file = os.path.join(tmp, "tokenizer_config.json")
#         if os.path.isfile(cfg_file):
#             with open(cfg_file, "r", encoding="utf-8") as f:
#                 cfg = json.load(f)
#             # 4. 清洗
#             cfg = _make_jsonable(cfg)
#             # 5. 写回真正目录
#             os.makedirs(save_dir, exist_ok=True)
#             with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
#                 json.dump(cfg, f, indent=2, sort_keys=True, ensure_ascii=False)
#         else:
#             # 没有 tokenizer_config.json 的 tokenizer，直接拷贝
#             shutil.copytree(tmp, save_dir, dirs_exist_ok=True)
#
# # 6. 猴子补丁
# PreTrainedTokenizerBase.save_pretrained = clean_save
# from transformers.trainer import Trainer
# import json, os, shutil
#
# _orig_save_model = Trainer.save_model
#
# def clean_save_model(self, output_dir: str, _internal_call=False):
#     # 1. 先正常保存模型权重 & generation_config
#     self.model.save_pretrained(output_dir)
#     if self.tokenizer is not None:
#         # 2. 让 tokenizer 自己落盘（走我们上一步的清洗逻辑）
#         self.tokenizer.save_pretrained(output_dir)
#     # 3. 手动写 training_args.json
#     self.args.save_pretrained(output_dir)
#     # 4. 如果担心缺 special_tokens_map，也手动拷一遍
#     #    （tokenizer.save_pretrained 已包含，无需重复）
#
# # 5. 猴子补丁
# Trainer.save_model = clean_save_model
#
#
# # ===== 放在 trainer.save_model 前 =====
# from transformers.trainer import Trainer
# import os, json
#
# _orig_save_model = Trainer.save_model
#
# def safe_save_model(self, output_dir: str, _internal_call=False):
#     self.model.save_pretrained(output_dir)
#     if self.tokenizer is not None:
#         self.tokenizer.save_pretrained(output_dir)
#         # 强制检查 tokenizer_config.json 非空
#         cfg_path = os.path.join(output_dir, "tokenizer_config.json")
#         if not (os.path.isfile(cfg_path) and os.path.getsize(cfg_path)):
#             raise RuntimeError(f"{cfg_path} 被写成了空文件，请检查清洗逻辑！")
#     self.args.save_pretrained(output_dir)
#
# Trainer.save_model = safe_save_model
# # ===== 结束 =====
# # ------------------------------------------------------------------
# # 3. 现在可以安心保存
# # ------------------------------------------------------------------
# trainer.save_model("./llama-2-7b-zh-alpaca-finetuned")
#
# """
# trainer.save_model() 在任何时刻都会做下面三件事：
# 调 model.save_pretrained() → 保存权重 + generation_config
# 调 tokenizer.save_pretrained() → 保存分词器配置
# 调 training_args.save_pretrained() → 保存训练超参
# """