# from vllm import LLM
import json

from transformers import BitsAndBytesConfig
from vllm import LLM

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

llm = LLM(model=r"D:\PycharmProjects\finetune qwen\解析标准地址\output_qwen_merged", quantization=BitsAndBytesConfig(load_in_8bit=True))
prompts = ["杭州市余杭区中泰街道石鸽社区环园南路3号中鑫钢构1号辅房"]
for prompt in prompts:
    # 定义对话内容
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_CONTENT,
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    output = llm.generate(conversation)
    print(output)

    llm.generate()
