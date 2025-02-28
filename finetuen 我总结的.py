import json
import numpy as np
import torch
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,DataCollatorForSeq2Seq

class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length

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
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
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
    swanlab_callback = SwanLabCallback(
        project=" deepspeed 弱智吧数据微调 Qwen-14B",
        config={
            "model": "Qwen-14B",
            "dataset": "弱智吧",
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    )

    print(f"GPU 是否可用{torch.cuda.is_available()}，有{torch.cuda.device_count()}个GPU")
    model_name = "D:/Qwen2.5-0.5B-Instruct"
    train_json_path = "//data/ner/train.json"
    val_json_path = "//data/ner/dev.json"
    max_source_length = 62  # text 样本最长的 token 是 50 ，模板自身大约 30 个，总共至少 80 个
    max_target_length = 151  # label 样本最长的 token 是 70
    logs_dir = "logs"
    writer = SummaryWriter(logs_dir)
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype="bfloat16")  # sdpa=Scaled Dot-Product Attention,  flash_attention_2 只支持 fp16 和 bf16 但是加速不明显甚至减速
    print("Start Load Train and Validation Data...")
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    # trainer
    training_args = TrainingArguments(
        output_dir="sft-7B-lora-ner",
        do_train=True,
        do_eval=True,
        seed=42,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=2,
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
    trainer = Trainer( model=model,
                       args=training_args,
                       train_dataset=training_set,
                       eval_dataset=val_set,
                       tokenizer=tokenizer,
                       callbacks=[swanlab_callback],)

    print("Start Training...")
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()