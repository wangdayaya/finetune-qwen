import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


SYSTEM_CONTENT = """请根据用户输入的地址完成地址节提取任务, 并以JSON格式输出。地址标签的具体含义如下，请在下面指定的标签范围内进行地址节解析：
- province：表示省份名称。
- city：表示城市名称。
- county：表示区县名称。
- town：表示街道名称。
- community：表示社区名称或者自然村名称。
- village：表示自然村名称。
- group：只表示自然村下的组，如“1组”、“二组”等。
- street：表示街、路、巷、弄堂。
- doorplate：只表示门牌号，如“32号”、“5-12号”、“12、13号”。
- subdistrict：表示小区名、单位名。
- building：表示建筑物名称、人名、居民家、辅房、厂房、某人的附房、人名+数字、建筑物+数字。如"吴水生"、“柯爱萍附房”、“徐廷忠03”、“白云源风景区02”、“机械制造厂厂房”、“国泰密封厂房7”等。
- building_num：只表示楼幢号，如“1幢”、“2栋”、“3-2幢楼”、“4座”、“3号楼”等。
- unit：只表示单元号，如“3单元”。
- floor：只表示楼层，如 “2层”、"3楼"、“-1楼”、“负二楼”等。
- room：只表示房间号，如 “1001室”、"302房"等。
- attachment：表示附属物或者城市部件名称，只能解析公厕、公交站、路灯杆、地铁站口、监控这五类及同语义的名称。

注意：
1、只能使用上面给出的标签进行地址节解析任务，禁止出现上面没有提到的标签。
2、一般情况下，按照上面给出的从上到下的地址节顺序对于给定的地址从左到右依次进行地址节提取即可。如果用户输入地址构成特殊则自行判定。
3、答案只能返回 json 格式，具体看下面示例。

示例：
Q：杭州市萧山区益农镇利围村浙东钢管制品公司
A：{'province': '浙江省', 'city': '杭州市', 'county': '萧山区', 'town': '益农镇', 'community': '利围村', 'building': '浙东钢管制品公司'}

Q：杭州市萧山区城厢街道湖头陈社区湖园三路北之江纺织有限公司15号楼1单元301室
A：{'province': '浙江省', 'city': '杭州市', 'county': '萧山区', 'town': '城厢街道', 'community': '湖头陈社区', 'street': '湖园三路', 'building': '北之江纺织有限公司', 'building_num': '15号楼', 'unit': '1单元', 'room': '301室'}

Q：杭州市临安区青山湖街道青南村闵家坞潘治源
A：{'province': '浙江省', 'city': '杭州市', 'county': '临安区', 'town': '青山湖街道', 'community': '青南村', 'village': '闵家坞', 'building': '潘治源'}
"""


SYSTEM_CONTENT =  """请根据用户输入的地址完成地址节提取任务, 并以JSON格式输出。地址标签的具体含义如下，请在下面指定的标签范围内进行地址节解析：
- province：表示省份名称。
- city：表示城市名称。
- county：表示区县名称。
- town：表示街道名称。
- community：表示社区名称或者自然村名称。
- village：表示自然村名称。
- group：只表示自然村下的组，如“1组”、“二组”等。
- street：表示街、路、巷、弄堂。
- doorplate：只表示门牌号，如“32号”、“5-12号”、“12、13号”。
- subdistrict：表示小区名、单位名。
- building：表示建筑物名称、人名、居民家、辅房、厂房、某人的附房、人名+数字、建筑物+数字。如"吴水生"、“柯爱萍附房”、“徐廷忠03”、“白云源风景区02”、“机械制造厂厂房”、“国泰密封厂房7”等。
- building_num：只表示楼幢号，如“1幢”、“2栋”、“3-2幢楼”、“4座”、“3号楼”等。
- unit：只表示单元号，如“3单元”。
- floor：只表示楼层，如 “2层”、"3楼"、“-1楼”、“负二楼”等。
- room：只表示房间号，如 “1001室”、"302房"等。
- attachment：表示附属物或者城市部件名称，只能解析公厕、公交站、路灯杆、地铁站口、监控这五类及同语义的名称。

注意：
1、只能使用上面给出的标签进行地址节解析任务，禁止出现上面没有提到的标签。
2、答案直接返回标准格式的 JSON 字符串，不要其他赘述。

示例:
问：帮我解析地址，杭州市拱墅区和睦街道李家桥社区登云路425-1号和睦院1幢502室
答： {"province": "浙江省", "city": "杭州市", "county": "拱墅区", "town": "和睦街道", "community": "李家桥社区", "street": "登云路", "doorplate": "425-1号", "subdistrict": "和睦院", "building_num": "1幢", "room": "502室"}}

问：杭州市拱墅区上塘街道假山路社区假山路46号假山新村老年活动室16-1号202室
答：{"province": "浙江省", "city": "杭州市", "county": "拱墅区", "town": "上塘街道", "community": "假山路社区", "street": "假山路", "doorplate": "46号", "subdistrict": "假山新村", "building": "老年活动室", "building_num": "16-1号", "room": "202室"}}
"""



SYSTEM_CONTENT = """请根据用户输入的地址完成地址节提取任务, 并以JSON格式输出。地址标签的具体含义如下，请在下面指定的标签范围内进行地址节解析：
- province：表示省份名称。
- city：表示城市名称。
- county：表示区县名称。
- town：表示街道名称。
- community：表示社区名称或者自然村名称。
- village：表示自然村名称。
- group：只表示自然村下的组，如“1组”、“二组”等。
- street：表示街、路、巷、弄堂。
- doorplate：只表示门牌号，只能在 subdistrict、building 前出现才合法，如“32号”、“5-12号”、“12、13号”。
- subdistrict：表示小区名、单位名。
- building：表示建筑物名称、人名、居民家、辅房、厂房、某人的附房、人名+数字、建筑物+数字。如"吴水生"、“柯爱萍附房”、“徐廷忠03”、“白云源风景区02”、“机械制造厂厂房”、“国泰密封厂房7”等。
- building_num：只表示楼幢号，只能在 building 后出现才合法，如“1幢”、“2栋”、“3-2幢楼”、“4座”、“3号楼”等。
- unit：只表示单元号，如“3单元”。
- floor：只表示楼层，如 “2层”、"3楼"、“-1楼”、“负二楼”等。
- room：只表示房间号，如 “1001室”、"302房"等。
- attachment：表示附属物或者城市部件名称，只能解析公厕、公交站、路灯杆、地铁站口、监控这五类及同语义的名称。

注意：
1、只能使用上面给出的标签进行地址节解析任务，禁止出现上面没有提到的标签。
2、答案直接返回标准格式的 JSON 字符串，不要其他赘述。

示例:
问：帮我解析地址，杭州市拱墅区和睦街道李家桥社区登云路425-1号和睦院1幢502室
答： {"province": "浙江省", "city": "杭州市", "county": "拱墅区", "town": "和睦街道", "community": "李家桥社区", "street": "登云路", "doorplate": "425-1号", "subdistrict": "和睦院", "building_num": "1幢", "room": "502室"}}

问：杭州市拱墅区上塘街道假山路社区假山路46号假山新村老年活动室16-1号202室
答：{"province": "浙江省", "city": "杭州市", "county": "拱墅区", "town": "上塘街道", "community": "假山路社区", "street": "假山路", "doorplate": "46号", "subdistrict": "假山新村", "building": "老年活动室", "building_num": "16-1号", "room": "202室"}}
"""



def get_token_distribution(file_path, tokenizer):
    input_num_tokens, outout_num_tokens = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
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
    train_data_path = r"D:\PycharmProjects\finetune qwen\data\address\address_gs_xh_train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, i_95, o_min, o_max, o_avg, o_95 = get_token_distribution(train_data_path, tokenizer)
    print(
        f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg}, i_95:{i_95}, o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}, o_95:{o_95}")


main()

# gs i_min：719, i_max：758, i_avg：740.0953815261045, i_95:750.0, o_min：43, o_max：97, o_avg：77.30622489959839, o_95:90.0
#