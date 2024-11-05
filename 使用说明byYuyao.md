# 准备

1. 首先用environment.yml装下环境

2. 然后使用本地的transformers-4.28.1覆盖掉environment.yml中安装的transformer，命令是

   ```
   pip install -e transformers-4.28.1 # run LLaMA2 models
   ```

3. 你可能还会遇到一些环境问题（比如numpy和torch冲突，torch和transformers冲突），反正多试试就好了

4. 然后改run_ice.py中的模型的环境

# 我改了啥

1. 引入了[浙大的数据集](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view)（已经下好了），有关浙大这个项目的介绍可以在[这里](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)，代码在[这里](https://github.com/zjunlp/EasyEdit/blob/main/examples/run_safety_editing.py)（主要就是学下咋加载数据集，我已经改好了）。有关他们咋判断llm的回答是否有毒，他们是训练了个Roberta作为分类模型，模型在[这里](https://huggingface.co/zjunlp/SafeEdit-Safety-Classifier)（下载到本地之后需要在run_ice.py中改路径），其中Roberta的label为 "0": "safe"，"0.5": "nuetral"，"1":"unsafe"。

2. 改了deck.py和run_ice.py，看一看应该就懂了

# 咋跑

跑deck：

```
python run_ice.py --model_name Llama-2-7b-chat-hf --mode deck --num-gpus 1 --batch_size 1 > output.txt
```

跑baseline：

```
python run_ice.py --model_name Llama-2-7b-chat-hf --mode baseline --num-gpus 1 --batch_size 1 > output.txt
```

# 目前的效果

1. 解毒率（指的就是llm面对unsafe的prompt不回答toxic文本的比例）大概是88\%左右，还可以。
2. baseline，大概是50%左右。
3. 需要注意的是，咱们的unsafe输入用的是带few-shot的，但是浙大那篇的输入是zero-shot。这么做主要是因为deck用了cot。

# 可以改啥

1. 首先就是few-shot的example，我用了5个example，可能太长了。也许换成三个效果会更好点
2. deck里两个prompt的logits相减的比例。deck里默认是0.2，可以需可以挑一挑。可以在vscode左侧的全局搜索中搜索`final_logits = logits - st_coef * logits_student`这个代码，就可以找到这里。其中st_coef可以在0~1之间调一调。
