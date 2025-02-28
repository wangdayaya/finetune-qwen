import json
import os

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = []
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for obj in data:
                    instruction = obj["instruction"]
                    output = obj["output"]
                    self.data.append({
                        "instruction": instruction,
                        "output": output
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, instruction, output):
        messages = [
            {"role": "system",
             "content": "你是一个有帮助的助手"},
            {"role": "user", "content": instruction}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(output, add_special_tokens=False, max_length=self.max_target_length,
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
    lora_rank = 2
    lora_alpha = 8
    lora_dropout = 0.01
    swanlab_callback = SwanLabCallback(
        project="ner-sft-lora-Qwen2.5-7B-Instruct",
        config={
            "model": "Qwen2.5-7B-Instruct",
            "dataset": "ner",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    )

    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    # 基础模型位置
    model_name = "D:/Qwen2.5-7B-Instruct"
    train_json_path = r"D:\PycharmProjects\finetune-qwen2.5-0.5B\data\ruozhi\ruozhiba_train_2449.json"
    val_json_path = r"D:\PycharmProjects\finetune-qwen2.5-0.5B\data\ruozhi\ruozhiba_val_153.json"
    max_source_length = 90  # text 样本最长的 tokenf 是 50 ，模板自身大约 30 个，总共至少 80 个
    max_target_length = 70  # label 样本最长的 token 是 70
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    model = AutoModelForCausalLM.from_pretrained(model_name,  attn_implementation="sdpa", torch_dtype="bfloat16") # sdpa=Scaled Dot-Product Attention  flash_attention_2 只支持 fp16 和 bf16

    # 数据
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)

    # lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # trainer
    training_args = TrainingArguments(
        output_dir="sft-7B-lora-ner",
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
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
        callbacks=[swanlab_callback]
    )
    print("Start Training...")
    trainer.train()


if __name__ == '__main__':
    main()
