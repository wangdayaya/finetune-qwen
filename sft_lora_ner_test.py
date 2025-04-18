import time

import torch
from peft import PeftModel
from transformers import  AutoTokenizer, AutoModelForCausalLM

SYSTEM_CONTENT = """请根据用户输入的地址完成地址节提取任务, 并以JSON格式输出。地址标签的具体含义如下，请在下面指定的标签范围内进行地址节解析：
- province：表示省份。
- city：表示城市。
- county：表示区县。
- town：表示街道。
- community：表示社区或者自然村。
- village：表示自然村。
- group：表示自然村下的组，如“xx村1组”，“xx村二组”。
- street：表示街、路、巷、弄堂。
- doorplate：表示门牌号，如“32号”。
- subdistrict：表示小区名、单位名。
- building：表示建筑物名称、人名、居民家、辅房、厂房、附房。
- building_num：表示楼幢号，如“1幢”、“2栋”、“3-2幢楼”、“4座”。
- unit：表示单元，如“3单元”。
- floor：表示楼层，如 “2层”。
- room：表示房间号，如 “1001室”。
- attachment：表示附属物或者城市部件名称，如公交站、路灯杆、地铁站口、监控等。

示例：
Q：杭州市萧山区益农镇利围村浙东钢管制品公司
A：{'province': '浙江省', 'city': '杭州市', 'county': '萧山区', 'town': '益农镇', 'community': '利围村', 'building': '浙东钢管制品公司'}

Q：杭州市萧山区城厢街道湖头陈社区湖园三路北之江纺织有限公司15号楼1单元301室
A：{'province': '浙江省', 'city': '杭州市', 'county': '萧山区', 'town': '城厢街道', 'community': '湖头陈社区', 'street': '湖园三路', 'building': '北之江纺织有限公司', 'building_num': '15号楼', 'unit': '1单元', 'room': '301室'}

Q：杭州市临安区青山湖街道青南村闵家坞潘治源
A：{'province': '浙江省', 'city': '杭州市', 'county': '临安区', 'town': '青山湖街道', 'community': '青南村', 'village': '闵家坞', 'building': '潘治源'}
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "D:\Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载LoRA模型的权重
# lora_model_path = r"D:\PycharmProjects\finetune qwen\address_ner\output_qwen_merged"
# model = PeftModel.from_pretrained(model, lora_model_path)
# model.to(device)
# model.eval()


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
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    start = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=110, pad_token_id=151643, eos_token_id=[151645, 151643])
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