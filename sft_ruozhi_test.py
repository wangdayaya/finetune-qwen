import time

import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    metrics = {
        'total_time': 0,
        'total_tokens': 0,
        'model_inf_gpu': 0,
        'model_inf_cpu': [],
        'load_model_gpu': 0,
        'load_model_cpu': 0
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_before = torch.cuda.max_memory_allocated(device)
    cpu_before = psutil.cpu_percent()

    tokenizer_path = "D:\Qwen2.5-0.5B-Instruct"
    train_model_path = "sft-0.5B-full-ruozhi"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    model.to(device)

    metrics['load_model_gpu'] = torch.cuda.max_memory_allocated(device) - gpu_before
    metrics['load_model_cpu'] = psutil.cpu_percent() - cpu_before

    test_case = [
        "地球上有70亿人，为什么只有86w人关注弱智吧，难道弱智这么少吗",
        "游泳比赛时把水喝光后跑步犯规吗",
        "一些高中想提高升学率，为什么不直接招大学生？"
    ]

    gpu_middle = torch.cuda.max_memory_allocated(device)
    cpu_middle = psutil.cpu_percent()

    for case in test_case:
        # 记录推理前状态
        start_time = time.time()

        messages = [ {"role": "system", "content": "你是一个有帮助的助手"},
                     {"role": "user", "content": case}]
        text = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate( model_inputs.input_ids, max_new_tokens=151, top_k=1)
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("----------------------------------")
        print(f"input: {case}\nresult: {response}")

        # 计算指标
        latency = time.time() - start_time
        tokens = len(generated_ids[0])

        # 更新统计数据
        metrics['total_time'] += latency
        metrics['total_tokens'] += tokens
        metrics['model_inf_cpu'].append(psutil.cpu_percent() - cpu_middle)
        metrics['model_inf_gpu'] = max( metrics['model_inf_gpu'], torch.cuda.max_memory_allocated(device) - gpu_middle )

    # 输出汇总报告
    print(f"\n性能报告：")
    print(f"• 总耗时：{metrics['total_time']:.2f}s")
    print(f"• Token速度：{metrics['total_tokens'] / metrics['total_time']:.1f} tokens/s")
    print(f"• 加载模型峰值GPU内存：{metrics['load_model_gpu'] / 1024 ** 2:.1f} MB")
    print(f"• 加载模型推理平均CPU占用：{metrics['load_model_cpu']:.1f}%")
    print(f"• 模型推理峰值GPU内存：{metrics['model_inf_gpu'] / 1024 ** 2:.1f} MB")
    print(f"• 模型推理平均CPU占用：{sum(metrics['model_inf_cpu']) / len(test_case):.1f}%")


if __name__ == '__main__':
    main()


# 性能报告：
# • 总耗时：6.74s
# • Token速度：67.3 tokens/s
# • 加载模型峰值GPU内存：1885.3 MB
# • 加载模型推理平均CPU占用：0.4%
# • 模型推理峰值GPU内存：14.6 MB
# • 模型推理平均CPU占用：8.2%