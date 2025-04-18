import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

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

# def get_token_distribution(file_path, tokenizer):
#     input_num_tokens, outout_num_tokens = [], []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#         for obj in data:
#             instruction = obj["instruction"]
#             output = obj["output"]
#             input_num_tokens.append(len(tokenizer(instruction).input_ids))
#             outout_num_tokens.append(len(tokenizer(output).input_ids))
#     return min(input_num_tokens), max(input_num_tokens), np.mean(input_num_tokens), np.percentile(input_num_tokens, 95), \
#         min(outout_num_tokens), max(outout_num_tokens), np.mean(outout_num_tokens), np.percentile(outout_num_tokens,
#                                                                                                   95),


def get_token_distribution(file_path, tokenizer):
    input_num_tokens, outout_num_tokens = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()[:200000]):
            data = json.loads(line)
            text = data["text"]
            label = data["label"]
            label = json.dumps(label, ensure_ascii=False)
            messages = [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": text}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            instruction = tokenizer(prompt, )
            input_num_tokens.append(len(instruction["input_ids"]))

            response = tokenizer(label, )
            outout_num_tokens.append(len(response["input_ids"]))
    return min(input_num_tokens), max(input_num_tokens), np.mean(input_num_tokens), np.percentile(input_num_tokens, 95), \
            min(outout_num_tokens), max(outout_num_tokens), np.mean(outout_num_tokens), np.percentile(outout_num_tokens, 95),


def main():
    model_path = "D:\Qwen2.5-3B-Instruct"
    train_data_path = r"data/address/address_train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, i_95, o_min, o_max, o_avg, o_95 = get_token_distribution(train_data_path, tokenizer)
    print(
        f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg}, i_95:{i_95}, o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}, o_95:{o_95}")


main()
