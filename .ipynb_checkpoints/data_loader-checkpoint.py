import json
import os

def load_data(data_path, dataset_type='default'):
    """
    根据选择的数据集类型加载问题列表。

    Args:
    - data_path: 数据存储的路径。
    - dataset_type: 数据集类型，'default' 或 'ultraSafety'。
    
    Returns:
    - 返回问题列表。
    """
    questions = []

    if dataset_type == 'default':
        # 默认数据集路径
        dataset_file = os.path.join(data_path, 'SafeEdit_test.json')
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"默认数据集文件 {dataset_file} 未找到！")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            # 提取每个条目的 'adversarial prompt' 作为问题
            for entry in dataset:
                question = entry.get("adversarial prompt", "")
                if question:
                    questions.append(question)
    
    elif dataset_type == 'ultraSafety':
        # UltraSafety 数据集路径
        dataset_file = os.path.join('Datasets', 'UltraSafety.jsonl')
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"UltraSafety 数据集文件 {dataset_file} 未找到！")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            # jsonl 文件每行是一个 JSON 对象
            for line in f.readlines():
                entry = json.loads(line.strip())
                question = entry.get("prompt", "")
                if question:
                    questions.append(question)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return questions
