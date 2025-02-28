import time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
model_path = "D:\Qwen2.5-7B-Instruct"
train_model_path = "D:\Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 正常加载模型推理
model = AutoModelForCausalLM.from_pretrained(train_model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用量化进行加载模型并推理
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained(train_model_path,
#                                              device_map="auto",
#                                              quantization_config=quantization_config,
#                                              ).eval()


test_case = [
    "三星WCG2011北京赛区魔兽争霸3最终名次",
    "新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准",
    "证券时报记者肖渔"
]
start = time.time()
for case in test_case:
    messages = [
        {"role": "system", "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
        {"role": "user", "content": case}
    ]
    text = tokenizer.apply_chat_template( messages, tokenize=False,  add_generation_prompt=True )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=50, top_k=1 )
    generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("----------------------------------")
    print(f"input: {case}\nresult: {response}")
print(f"推理耗时{time.time()-start}")


# 23.5G-0.4G
# 7B-full 推理
# ----------------------------------
# input: 三星WCG2011北京赛区魔兽争霸3最终名次
# result: {
#     "实体": "WCG2011北京赛区",
#     "类型": "地点"
# }
# ----------------------------------
# input: 新华网孟买3月10日电（记者聂云）印度国防部10日说，印度政府当天批准
# result: ```json
# {
#   "地点": ["孟买", "新华网"],
#   "组织名": ["印度国防部", "印度政府"]
# }
# ```
# ----------------------------------
# input: 证券时报记者肖渔
# result: {"name": "肖渔", "position": "证券时报记者"}
# 推理耗时40.510847330093384