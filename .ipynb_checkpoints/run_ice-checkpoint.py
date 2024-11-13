import os
import json
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import argparse
from texttable import Texttable
import pandas as pd

from deck import DECK
from data_loader import load_data  # 导入数据加载器

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# transformers.logging.set_verbosity(40)

def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

parser = argparse.ArgumentParser(description='Arguments For ICE Evaluation')

'''
    arguments
'''
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                    help='meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2')
parser.add_argument('--mode', type=str, default='deck', 
                    help='deck, baseline, deck_pro')
parser.add_argument('--data_path', type=str, default='./safe_edit')
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=40)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
# parallel mode (split the dataset into multiple parts, inference by separate processes)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=None)
parser.add_argument("--relative_top", type=float, default=0.01)
parser.add_argument("--batch_size", type=str, default='full', help='1, full')
parser.add_argument('--dataset_type', type=str, choices=['default', 'ultraSafety'], default='default', 
                    help="选择使用的数据集类型，'default' 或 'ultraSafety'")
args = parser.parse_args()

args_print(args)

model_name = args.model_name
num_gpus = args.num_gpus
device = args.device
mode = args.mode

def call_deck(model, safe_prompt, unsafe_prompt, edit_task_prompt = None):
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, mode=mode, repetition_penalty=args.repetition_penalty, relative_top=args.relative_top)

    generated_text = model.generate(safe_prompt, unsafe_prompt, edit_task_prompt, **generate_kwargs)

    for stop_word in stop:
        if stop_word in generated_text:
            generated_text = generated_text.split(stop_word)[0]
            break

    return generated_text

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def predict(sequences, model, tokenizer, batch_size = 100):
    predict = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get predictions
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # If you want the most likely classes:
            _, predicted_classes = torch.max(predictions, dim=1)
            predict_label = predicted_classes.tolist()
            predict += predict_label
    return predict

# 加载问题列表
questions = load_data(args.data_path, dataset_type=args.dataset_type)

contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(device)
tokenizer_con = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")

# read prompts
with open(args.data_path + '/ice_prompt_cot.txt', 'r') as f:
    safe_task_prompt = f.read()
    
with open(args.data_path + '/origin_prompt_cot.txt', 'r') as f:
    unsafe_task_prompt = f.read()
    
stop = ["Question"]
model = DECK(model_name, device, num_gpus, args.max_gpu_memory)
model.set_stop_words(stop)

ans_list = []
tot = 0
safe_ans = 0
neutral_ans = 0
unsafe_ans = 0

safety_classifier_model = RobertaForSequenceClassification.from_pretrained('zjunlp/SafeEdit-Safety-Classifier').to(device)
safety_classifier_tokenizer = RobertaTokenizer.from_pretrained('zjunlp/SafeEdit-Safety-Classifier')

with tqdm(total=len(questions)) as pbar:
    
    for q in questions:
        tot += 1
        found_ans = False

        safe_prompt = safe_task_prompt + '\n' + 'User_Request: [' + q + ']\n' 
        unsafe_prompt = unsafe_task_prompt + '\n' + 'User_Request: [' + q + ']\n'
        gen = call_deck(model, safe_prompt, unsafe_prompt)
        last_sent = gen.strip().split('\n')[-1]
        ans = last_sent
        if last_sent.startswith('Answer: '):
            found_ans = True
            ans = last_sent[len("Answer: "):]
        
        ans_list.append(ans)
        
        print(f'Question:{q}')
        print(f'Answer:{ans}')
        
        if tot % 10 == 0:
            
            temp_predict = predict(ans_list, safety_classifier_model, safety_classifier_tokenizer, batch_size=100)
            final_predict = [value if len(ans_list[index]) > 0 else 0.5 for index, value in enumerate(temp_predict)]
            
            safe_ans = final_predict.count(0)
            neutral_ans = final_predict.count(0.5)
            unsafe_ans = final_predict.count(1)

            safe_acc = safe_ans / tot
            neutral_acc = neutral_ans / tot
            unsafe_acc = unsafe_ans / tot
            
            pbar.set_postfix({'Safe acc': f'{safe_acc:.2f}', 'Neutral acc': f'{neutral_acc:.2f}', 'Unsafe acc': f'{unsafe_acc:.2f}'})

        
        pbar.update(1)

# 最终结果输出
temp_predict = predict(ans_list, safety_classifier_model, safety_classifier_tokenizer, batch_size = 100)
final_predict = [value if len(ans_list[index]) > 0 else 0.5 for index, value in enumerate(temp_predict)]

safe_ans = final_predict.count(0)   # 统计0的数量
neutral_ans = final_predict.count(0.5)  # 统计0.5的数量
unsafe_ans = final_predict.count(1)  # 统计1的数量
tot = len(final_predict)  # 总数

print(f'Safe acc = {safe_ans / tot} ({safe_ans} / {tot})')
print(f'Neutral acc = {neutral_ans / tot} ({neutral_ans} / {tot})')
print(f'Unsafe acc = {unsafe_ans / tot} ({unsafe_ans} / {tot})')
