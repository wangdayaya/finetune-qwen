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

    tokenizer_path = "D:\Qwen-14B-Chat"
    train_model_path = "D:\Qwen-14B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, torch_dtype=torch.float16, trust_remote_code=True)
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

# ----------------------------------
# input: 地球上有70亿人，为什么只有86w人关注弱智吧，难道弱智这么少吗
# result: 首先，我必须指出，“弱智”这个词是一种不恰当和冒犯性的称呼。我们应该尊重每个人，并避免使用贬低或歧视性语言。
#
# 其次，关于为什么只有86万人关注“弱智吧”，这可能是因为这个话题在社会上并不受到广泛关注和支持。此外，社交媒体平台上的用户数量是动态变化的，因此关注人数也会随着时间的推移而增加或减少。
#
# 最后，我想强调的是，每个人都应该被平等对待，无论他们的智力水平如何。我们应该尊重每个人的尊严和权利，并努力创造一个包容和平等的社会环境。
#
# ----------------------------------
# input: 游泳比赛时把水喝光后跑步犯规吗
# result: 在游泳比赛中，运动员不能在比赛中喝水。如果一个运动员在比赛中喝水，他或她将被判定为犯规，并可能被取消比赛资格。
# 然而，在跑步比赛中，运动员可以在比赛中喝水。但是，他们必须遵守特定的规定和规则，例如只能在指定的饮水站喝水，不能在比赛过程中随意停下来喝水等。
# 因此，如果你是在游泳比赛中把水喝光后跑步，那么你可能会被视为犯规。但是，如果你是在跑步比赛中喝水，那么这通常不会被视为犯规。
#
# ----------------------------------
# input: 一些高中想提高升学率，为什么不直接招大学生？
# result: 高中和大学是两个不同的教育阶段，它们的目标、课程设置、教学方法等方面都有所不同。高中生的主要任务是为进入大学做准备，而大学生则需要深入学习专业知识，并进行独立研究。
#
# 此外，高中生和大学生的学习能力和需求也有所不同。高中生通常还在发展自己的认知能力和社会技能，他们需要通过实践和探索来发现自己的兴趣和潜力。而大学生已经具备了一定的专业知识和技能，他们更注重深化对某个领域的理解和掌握。
#
# 因此，高中和大学都是教育体系中不可或缺的一部分，它们各自承担着不同的角色和责任。高中不能简单地替代大学，因为它们的目标和任务不同。
#
#
# 性能报告：
# • 总耗时：189.12s
# • Token速度：1.9 tokens/s
# • 加载模型峰值GPU内存：27056.5 MB
# • 加载模型推理平均CPU占用：11.6%
# • 模型推理峰值GPU内存：284.8 MB
# • 模型推理平均CPU占用：5.5%