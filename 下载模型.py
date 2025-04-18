#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-14B-Instruct', cache_dir="d:/")