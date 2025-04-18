import json
import random


def find_two_sum_random(x):
    t = random.randint(0, x)
    return f"王大丫丫的回答是--->{t} + {x - t}"


messages = []
start = 2
end = 2000
for i in range(start, end):
    m = {
        "instruction": f"{i}=",
        "input": "",
        "output": find_two_sum_random(i)
    }
    messages.append(m)
interval = int(end * 0.1)
train = messages[:-interval]
eval = messages[-interval:]

with open(f"data/math/math_train_{end - start - interval}.json", "w", encoding="utf-8") as json_file:
    json.dump(train, json_file, ensure_ascii=False, indent=4)
with open(f"data/math/math_eval_{interval}.json", "w", encoding="utf-8") as json_file:
    json.dump(eval, json_file, ensure_ascii=False, indent=4)