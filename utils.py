#coding=utf-8

import rouge
import re
from rouge_score import rouge_scorer
import torch
import json
import numpy as np
import random
import os

rouge = rouge.Rouge()
google_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  # target，prediction
WHITESPACE_AND_PUNCTUATION = {' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'}
ARTICLES = {'the', 'a', 'an'}


def compute_rouge(pred, target):  # source 预测值，target 真实值
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(pred), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(preds: list, targets: list):  # 计算总rouge
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(preds, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def compute_google_rouge(pred, target):
    return google_rouge.score(target, pred)['rougeL'].fmeasure


def normalize_text(s, rm_article=True):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        # 把前(或)后带有空格的冠词去掉
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        # replace special space
        if isinstance(text, str):
            text = text.replace(u'\u00a0', ' ')
        else:
            text = text.decode()  # byte object
            print(f'Non str in normalize text: {text}')
        return re.sub(r'\s+', " ", text)

    def remove_punc(text):
        # exclude = set(string.punctuation)
        # return "".join(ch for ch in text if ch not in exclude)
        for delimeter in WHITESPACE_AND_PUNCTUATION:
            text = text.replace(delimeter, ' ')
        return text

    def lower(text):
        return text.lower()

    if rm_article:
        return white_space_fix(remove_punc(remove_articles(lower(s))))
    else:  # 不remove article时也不lowercase
        return white_space_fix(remove_punc(s))


def compute_exact_match(prediction, truth):
    return normalize_text(prediction) == normalize_text(truth)


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)


def compute_max_f1(prediction: str, answers: list):
    """
    This metric measures the average overlap between the prediction and ground truth answer.
    We treat the prediction and ground truth as bags of tokens, and compute their F1.
    We take the maximum F1 over all of the ground truth answers for a given question,
    and then average over all of the questions.
    每个answer计算token f1-score，多个answer取最大f1
    """
    f1 = 0
    for answer in answers:
        f1 = max(compute_f1(prediction, answer), f1)
    return f1


# 改为普通rouge
def compute_max_rouge(prediction: str, answers: list):
    """计算rouge-l
    """
    rouge_score = 0
    for answer in answers:
        # score = compute_google_rouge(prediction, answer)
        score = compute_rouge(prediction, answer)['rouge-l']
        rouge_score = max(rouge_score, score)
    return rouge_score


def compute_max_google_rouge(prediction: str, answers: list):
    """计算rouge-l
    """
    rouge_score = 0
    for answer in answers:
        score = compute_google_rouge(prediction, answer)
        rouge_score = max(rouge_score, score)
    return rouge_score


def save_dataset(path, name, dataset):
    os.makedirs(path, exist_ok=True)
    path = path + '/' + name
    if path.endswith('txt'):
        with open(path, 'w', encoding='utf-8') as f:
            for line in dataset:
                line = line.strip()
                f.write(line + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset

def save_model(output_model_file, model, optimizer):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += 'pytorch_model.bin'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model_file, _use_new_zipfile_serialization=False)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


if __name__ == '__main__':
    path = 'dataset/train.json'
    dataset = read_dataset(path)
    print('dataset:', len(dataset))

    path = 'dataset/dev.json'
    dataset = read_dataset(path)
    print('dataset:', len(dataset))

    path = 'dataset/test.json'
    dataset = read_dataset(path)
    print('dataset:', len(dataset))