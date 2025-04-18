import json
import os
import re
import time

import requests
from tqdm import tqdm
import pandas as pd

# 打印数据框的内容
url = "http://10.5.1.105:9996/findComponents"
headers = {"Content-Type": "application/json;charset=utf8"}
prompts = []
labels = []
d = {}
with open(r"D:\PycharmProjects\finetune qwen\data\address\address_eval_42w.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = json.loads(line)
        if not data['text'].startswith("浙江省"):
            data['text'] = "浙江省" + data['text'].strip()
        prompts.append(data['text'])
        labels.append(data['label'])
        d[data['text']] = data['label']

bs = 200
N = len(prompts)
total = 0
correct = 0
for i in tqdm(range(0, N, bs)):
    addresses = prompts[i:i + bs]
    response = requests.post(url, data=json.dumps({"addresses": addresses, "complex": False}), headers=headers)
    body = response.json()
    for j, address in enumerate(addresses):
        a = body[address]
        b = d[address]
        total += len(b)
        for key in a.keys():
            if key in b and b[key].lower() == a[key].lower():
                correct += 1
            else:
                print(address)
                print("生成：", sorted(a.items(), key=lambda kv: (kv[1], kv[0])))
                print("标签：", sorted(b.items(), key=lambda kv: (kv[1], kv[0])))

print(correct / total)


# 0.8909102893678729
# 7s 耗时