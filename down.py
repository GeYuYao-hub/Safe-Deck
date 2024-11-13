#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('shakechen/Llama-2-7b-chat-hf', cache_dir='/root/autodl-tmp/Safe-Deck')