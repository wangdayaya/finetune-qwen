import json
import os

from ollama import chat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import os
import json
from ollama import chat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging


# 配置日志记录
def setup_logging(max_concurrent_requests):
    model = 'output_qwen_merged_quan_0417'
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"logs/{model}_{max_concurrent_requests}.log"
    if not os.path.exists(log_filename):
        open(log_filename, 'w').close()

    # 创建一个独立的日志记录器
    logger = logging.getLogger(f"{model}_{max_concurrent_requests}")
    logger.setLevel(logging.INFO)

    # 创建一个文件处理器，并设置日志格式
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger


# 定义请求函数
def fetch(prompt, label):
    response = chat(model='output_qwen_merged_quan_0417', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    completion_tokens = response['message']['content']
    try:
        result = json.loads(completion_tokens)
    except Exception as e:
        print(f"{prompt} 解析答案为 {completion_tokens} ，出现错误{e}")
        result = None
    return prompt, result, label


# 主函数
def run(prompts, labels, max_concurrent_requests, logger):
    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_prompt = {executor.submit(fetch, prompt, label): prompt for prompt, label in zip(prompts, labels)}
        for future in tqdm(as_completed(future_to_prompt), total=len(prompts)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Request failed: {e}")
    return results


# 示例调用
if __name__ == '__main__':

    prompts = []
    labels = []
    with open(r"D:\PycharmProjects\finetune qwen\data\address\address_gs_xh_eval.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines()[:20]:
            data = json.loads(line)
            prompts.append(data['text'])
            labels.append(data['label'])

    for max_concurrent_requests in [2, 4, 6]:
        logger = setup_logging(max_concurrent_requests)
        start_time = time.time()
        results = run(prompts, labels, max_concurrent_requests, logger)
        end_time = time.time()

        total_time = end_time - start_time
        correct = 0
        total = 0
        for result in results:
            prompt = result[0]
            generate = result[1]
            label = result[2]
            total += len(label.keys())
            for k, v in label.items():
                if k in generate and k in label and isinstance(generate[k], str) and isinstance(label[k], str) and \
                        generate[k].lower() == label[k].lower():
                    correct += 1
                else:
                    logger.error(f"Prompt: {prompt}\nGenerated: {generate}\nLabel: {label}\n-------------")

        total_tokens = sum(len(result[0]) for result in results)
        speed = len(prompts) / total_time
        tokens_per_second = total_tokens / total_time

        logger.info(f'Performance Results:')
        logger.info(f'  Max_concurrent_requests   : {max_concurrent_requests}')
        logger.info(f'  Total requests            : {len(prompts)}')
        logger.info(f'  Max concurrent requests   : {max_concurrent_requests}')
        logger.info(f'  Total time                : {total_time:.2f} seconds')
        logger.info(f'  Time of per request       : {speed:.2f} qps')
        logger.info(f'  Tokens per second         : {tokens_per_second:.2f}')
        logger.info(f'  Acc                       : {correct / total:.2f}')
        logger.info("-------------------------")
