import time

import torch
from peft import PeftModel
from transformers import  AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "D:\Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载LoRA模型的权重
lora_model_path = r"sft-7B-lora-ner/checkpoint-2013"
model = PeftModel.from_pretrained(model, lora_model_path)
model.to(device)
model.eval()


# ner
# test_case = [
#     "三星WCG2011北京赛区魔兽争霸3最终名次",
#     "新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准",
#     "证券时报记者肖渔"
# ]
# start = time.time()
# for case in test_case:
#     messages = [
#         {"role": "system", "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
#         {"role": "user", "content": case}
#     ]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
#     generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=50, top_k=1 )
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print("----------------------------------")
#     print(f"input: {case}\nresult: {response}")
# print(f"推理耗时{time.time()-start}")


# 普通对话

while True:
    prompt = input("user:")
    messages = [
        {"role": "system", "content": "你是一个有帮助的智能助手"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    start = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=151643, eos_token_id=[151645, 151643])
    print(f"推理耗时{time.time() - start}")
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"assis:{response}")


# 0.6G-23.5G 7B-lora
# ----------------------------------
# input: 三星WCG2011北京赛区魔兽争霸3最终名次
# result: {"game": ["魔兽争霸3"], "address": ["北京"], "organization": ["WCG"], "company": ["三星"]}
# ----------------------------------
# input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
# result: {"address": ["孟买"], "government": ["印度国防部", "印度政府"], "name": ["聂云"], "company": ["新华网"], "position": ["记者"]}
# ----------------------------------
# input: 证券时报记者肖渔
# result: {"book": ["证券时报"], "name": ["肖渔"], "position": ["记者"]}
# 推理耗时54.17272233963013