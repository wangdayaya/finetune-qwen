import json
import os
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, TrainerState, \
    TrainerControl

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


def read():
    result = []
    with open(r"D:\PycharmProjects\finetune qwen\data\address\address_gs_eval.jsonl", 'r', encoding='utf-8') as file:
        for line in file.readlines():
            json_line = json.loads(line)
            result.append(json_line)
    return result


class ChatCallback(TrainerCallback):
    def __init__(self, tokenizer, generate_every=500, max_new_tokens=50):
        self.tokenizer = tokenizer
        self.generate_every = generate_every
        self.max_new_tokens = max_new_tokens
        self.address = read()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.generate_every == 0:
            prompt = random.choice(self.address)
            text = prompt['text']
            label = prompt['label']
            messages = [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": text}
            ]
            new_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens,
                                         eos_token_id=self.tokenizer.eos_token_id, )
            decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Step {state.global_step}:\nGenerated text: {decode}\nLabel text: \n{label}")
            model.train()


class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []

        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in tqdm(f):
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    text = json_line["text"]
                    label = json_line["label"]
                    label = json.dumps(label, ensure_ascii=False)
                    self.data.append({
                        "text": text,
                        "label": label
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)


def main():
    # lora
    lora_rank = 2
    lora_alpha = 8
    lora_dropout = 0.01
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    output_dir = "3B_sft_address_gs_ner_0417"
    swanlab_callback = SwanLabCallback(
        project="address-ner-sft",
        experiment_name=output_dir,
        config={
            "model": "Qwen2.5-3B-Instruct",
            "dataset": "address",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )

    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    # 基础模型位置
    model_name = r"D:\Qwen2.5-3B-Instruct"
    train_json_path = r"D:\PycharmProjects\finetune qwen\data\address\address_gs_xh_train.jsonl"
    val_json_path = r"D:\PycharmProjects\finetune qwen\data\address\address_gs_xh_eval.jsonl"
    # i_min：255, i_max：311, i_avg：266.2797450004254, i_95:275.0, o_min：33, o_max：108, o_avg：60.213778068174335, o_95:80.0
    # 全市 i_min：532, i_max：587, i_avg：542.27003, i_95: 551.0, o_min：35, o_max：108, o_avg：60.190445, o_95: 80.0
    # gs i_min：719, i_max：758, i_avg：740.0953815261045, i_95:750.0, o_min：43, o_max：97, o_avg：77.30622489959839, o_95:90.0
    # i_min：688, i_max：727, i_avg：709.0953815261045, i_95:719.0, o_min：43, o_max：97, o_avg：77.30622489959839, o_95:90.0
    # i_min：710, i_max：755, i_avg：731.7957750669443, i_95: 742.0, o_min：42, o_max：100, o_avg：78.34501636417733, o_95: 90.0
    max_source_length = 760
    max_target_length = 100
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    # sdpa=Scaled Dot-Product Attention  flash_attention_2 只支持 fp16 和 bf16
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype="bfloat16")
    # 数据
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=10,
        bf16=True,
        # deepspeed=""
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        callbacks=[
            swanlab_callback,
            ChatCallback(tokenizer, generate_every=100, max_new_tokens=max_target_length)
        ]
    )
    print("Start Training...")
    trainer.train()


if __name__ == '__main__':
    main()

