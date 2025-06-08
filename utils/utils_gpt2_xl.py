import sys
import time
import torch
import torch.nn as nn
import os
import datasets
import torchvision
import torchvision.transforms as transforms
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from pdb import set_trace as st
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, GPT2ForSequenceClassification, TrainerCallback
from peft import PeftModel, get_peft_model, LoraConfig
from safetensors.torch import load_file

TOTAL_BAR_LENGTH = 65.0
term_width = 80
last_time = time.time()
begin_time = last_time

def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_data(task,model,validation_key,sentence1_key,sentence2_key, max_length=512):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(model)
    # 为 tokenizer 添加 padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    dataset = load_dataset("glue", task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[validation_key]
    return train_dataset, eval_dataset, tokenizer


def prepare_recover_data(model, trainset, batch_size, path, ratio=0.01):
    set_seed()
    all_indices = list(range(len(trainset)))
    num_samples = int(len(trainset) * ratio)
    indices = random.sample(all_indices, num_samples)

    ## 初始化recover_data包含与trainset相同的字段（除了label）
    recover_data = {key: [] for key in trainset[0].keys() if key != 'label'}
    recover_data['label'] = []  
    true_labels = []  
    
    ## 逐个提取数据项并添加到recover_data中
    for idx in indices:
        item = trainset[idx]
        for key in recover_data.keys():
            if key != 'label':
                recover_data[key].append(item[key])
        true_labels.append(item['label']) 

    subset = torch.utils.data.Subset(trainset, indices)
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct, tot = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(outputs, dim=1)
            for i in range(inputs["input_ids"].size(0)):
                predicted_label = predictions[i].cpu().tolist()
                recover_data["label"].append(predicted_label)  
                tot += 1
                ## 比较预测结果和真实标签
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1
    ## 计算并输出正确比例
    accuracy = correct / tot
    print(f"预测的label和真实label的正确比例: {accuracy:.4f}")
    print(recover_data.keys())
    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)
    

## 根据cosine similarity恢复矩阵行打乱
def row_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu, pre_model_mat_cpu)
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, row in enumerate(model_mat_cpu):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[perm[i]] = model_mat_cpu[i]
            success.append(perm[i])
    for i in range(len(restored_matrix)):
        if i not in success:
            print(f"index{i} is not in success")
            restored_matrix[i] = pre_model_mat_cpu[i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def col_restore_perm(pre_model_mat, model_mat, threshold=0.0):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu.T, pre_model_mat_cpu.T)
    perm = np.argmax(similarty_matrix, axis=1)
    restored_matrix = np.empty_like(model_mat_cpu)
    success = []
    for i, col in enumerate(model_mat_cpu.T):
        max_similarity = similarty_matrix[i, perm[i]]
        if max_similarity >= threshold:
            restored_matrix[:,perm[i]] = model_mat_cpu[:,i]
            success.append(perm[i])
    for i in range(len(restored_matrix[0])):
        if i not in success:
            restored_matrix[:,i] = pre_model_mat_cpu[:,i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def fix_factor(num, mini=1.0, max=6.0):
    if(num < mini):
        return mini
    elif(num > max):
        return max
    else:
        return num

def loss1(model):
    loss = 0
    for name, param in model.named_parameters():
        if "attn.c_attn.weight" in name or "attn.c_proj.weight" in name or "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name:
            loss += torch.sum(param ** 2)
        elif "lora" in name:
            loss += torch.sum(param ** 2)
    return loss

def loss2(model, pre_model):
    loss = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            loss += torch.sum(param ** 2)
    return torch.sqrt(loss+1e-8)

class CustomTrainer(Trainer):
    def __init__(self, pre_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_model = pre_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fn = nn.CrossEntropyLoss()
        loss_1 = 1e-4*loss1(model)
        loss_2 = 1e-4*loss2(model, self.pre_model)
        loss = loss_fn(logits, labels)+loss_1-loss_2
        return (loss, outputs) if return_outputs else loss


# 设置lora_model
def set_lora_model(model_name, lora_config, num_labels):
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    lora_model = get_peft_model(model, lora_config)
    return lora_model

def adjust_lora_model(model_name, lora_config, num_labels, weight1=None, weight2=None):
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    lora_model = get_peft_model(model, lora_config)
    lora_model_dict = lora_model.model.state_dict()
    lora_model_dict.update(weight1)
    lora_model_dict.update(weight2)
    lora_model.model.load_state_dict(lora_model_dict)

    lora_model = lora_model.merge_and_unload()
    return lora_model

def init_obfus_model(args, num_labels, obfus):
    if obfus == "tsqp":
        if os.path.exists(args.weight_dir_tsqp):
            safetensor_file1, safetensor_file2 = args.weight_dir_tsqp + "/model-00001-of-00002.safetensors", args.weight_dir_tsqp + "/model-00002-of-00002.safetensors"
        else:
            safetensor_file1, safetensor_file2 = args.weight_dir + "/model-00001-of-00002.pt", args.weight_dir_tsqp + "/model-00002-of-00002.pt"
    else:
        safetensor_file1, safetensor_file2 = args.weight_dir + "/model-00001-of-00002.safetensors", args.weight_dir + "/model-00002-of-00002.safetensors"
    weight1, weight2 = load_file(safetensor_file1), load_file(safetensor_file2)
    lora_config = LoraConfig(
            r=8,  
            lora_alpha=16,  
            lora_dropout=0.1,  
            target_modules=["c_fc", "c_attn", "c_proj"],  # 指定需要LoRA的层
        )
    lora_model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
    return lora_model