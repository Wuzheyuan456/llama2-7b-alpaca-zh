import json

with open("medical_zh/train_zh_0.jsonl", "r") as f:
    lst = [json.loads(item) for item in f.readlines()[:1000]]


with open("train_ffine.json", "w") as f:
    json.dump(lst, f, ensure_ascii=False)

# import pandas as pd
# import datasets
#
# df = pd.read_json('train_lora.json')
# ds = datasets.Dataset.from_pandas(df)