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
#from vtab import get_data, get_classes_num
from pdb import set_trace as st
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    GPT2ForSequenceClassification, 
    TrainerCallback
)
from datasets import load_dataset
import json
import matplotlib.pyplot as plt

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
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = load_dataset("glue", task)
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, max_length=max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, max_length=max_length)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[validation_key]
    return train_dataset, eval_dataset, tokenizer

def prepare_recover_data(model, trainset, batch_size, path, ratio=1):
    set_seed()
    all_indices = list(range(len(trainset)))
    num_samples = int(len(trainset) * ratio)
    indices = random.sample(all_indices, num_samples)

    recover_data = {key: [] for key in trainset[0].keys() if key != 'label'}
    recover_data['label'] = []  
    true_labels = []  
    
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
        token_type_ids = [item['token_type_ids'] for item in batch]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
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
                'token_type_ids': batch['token_type_ids'].to(device)
            }
            outputs = model(**inputs)
            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(outputs, dim=1)
            for i in range(inputs["input_ids"].size(0)):
                predicted_label = predictions[i].cpu().tolist()
                recover_data["label"].append(predicted_label)  
                tot += 1
                if predicted_label == true_labels[len(recover_data["label"]) - 1]:
                    correct += 1

    accuracy = correct / tot
    print(f"预测的label和真实label的正确比例: {accuracy:.4f}")
    
    with open(path, "w") as f:
        json.dump(recover_data, f, indent=4)

def loss1(model):
    loss = 0
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            loss += torch.sum(param ** 2)
    return loss

def loss2(model, pre_model):
    loss = 0
    for name, param in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name or "output.dense.bias" in name or "intermediate.dense.bias" in name:
            name = name.replace("module.", "")
            pre_data = pre_model.state_dict()[name]
            pre_data = pre_data.to(param.device)
            loss += torch.sum((param - pre_data) ** 2)
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
            restored_matrix[i] = pre_model_mat_cpu[i]
    restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return perm, success, restored_matrix

def col_restore_perm(pre_model_mat, model_mat, threshold=0.0, perm = None):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    similarty_matrix = cosine_similarity(model_mat_cpu.T, pre_model_mat_cpu.T)
    if perm is None:
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

from scipy.optimize import linear_sum_assignment
def col_restore_perm_our(pre_model_mat, model_mat):
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    # similarty_matrix = cosine_similarity(model_mat_cpu.T, pre_model_mat_cpu.T)
    # K = A^{-1}Ap = A^T(AA^T)^{-1}Ap
    # similarty_matrix = model_mat_cpu.T @ np.linalg.inv(model_mat_cpu @ model_mat_cpu.T) @ pre_model_mat_cpu
    # M = A^T Ap
    M = model_mat_cpu.T @ pre_model_mat_cpu
    row_ind, col_ind = linear_sum_assignment(-M)
    # P = np.zeros_like(M)
    # P[row_ind, col_ind] = 1
    # restored_matrix = model_mat_cpu @ P
    perm = row_ind[np.argsort(col_ind)]
    inv_perm = np.argsort(perm)
    # restored_matrix = model_mat_cpu[:, perm]
    # restored_matrix = torch.from_numpy(restored_matrix).to(model_mat.device)
    return inv_perm  

def col_restore_perm_and_scale(pre_model_mat, model_mat):
    """
    估计在 Frobenius 范数下
        min_{P(列置换), D(列对角)} ||A P D - B||_F^2
    的置换与对角缩放，用于「右乘」混淆：A 为混淆权重，B 为公开/预训练参考。

    记 A = model_mat(ob), B = pre_model_mat(参考). P 为列置换矩阵, D=diag(d) 为列缩放.
    列子问题把 B 的第 i 列与 A 的第 j 列（差一个标量）配对；匈牙利代价取与逐列最小二乘
    配对面兼容的形式 (B^T A)_{ij}^2 / ||A 列 j||^2, scales 中系数取 (B^T A)_{ij}/||A 列 j||^2
    在匹配边上的值（再按列重排为 ob 列序）。

    返回 inv_perm, inv_scales: 与既有 attack_shadownet_our 的 ``perm, scales`` + argsort/求逆
    的用法一致（由调用方组合成实际还原矩阵乘法）。
    """
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    A = model_mat_cpu
    B = pre_model_mat_cpu
    Mab = B.T @ A
    Maa = np.diag(A.T @ A)
    Maa = np.maximum(Maa, 1e-20)
    M = (Mab**2) / Maa[np.newaxis, :]
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = col_ind[np.argsort(row_ind)]  
    inv_perm = np.argsort(perm)
    scales = (Mab[row_ind, col_ind] / Maa[col_ind])[np.argsort(row_ind)]
    inv_scales = 1.0 / scales
    return inv_perm, inv_scales

def row_restore_perm_and_scale(pre_model_mat, model_mat):
    """
    估计在 Frobenius 范数下
        min_{P(行置换), D(行对角)} ||D P A - B||_F^2
    的置换与行对角缩放, 用于「左乘」行混淆：A=ob, B=参考.

    P 为行置换, D=diag(·) 为行缩放. 匹配代价用 (B A^T)_{ij}^2 / ||A 行 j||^2, 与上式逐行
    可分离时的最优分配一致; scales 在匹配边 (B A^T)_{ij}/||A 行 j||^2 上取得并依 ob 行序重排.
    """
    model_mat_cpu = model_mat.cpu().numpy()
    pre_model_mat_cpu = pre_model_mat.cpu().numpy()
    A = model_mat_cpu
    B = pre_model_mat_cpu
    Mab = B @ A.T
    Maa = np.diag(A @ A.T)
    Maa = np.maximum(Maa, 1e-20)
    M = (Mab**2) / Maa[np.newaxis, :]
    row_ind, col_ind = linear_sum_assignment(-M)
    perm = col_ind[np.argsort(row_ind)]
    inv_perm = np.argsort(perm)
    scales = (Mab[row_ind, col_ind] / Maa[col_ind])[np.argsort(row_ind)]
    inv_scales = 1.0 / scales

    return inv_perm, inv_scales

def restore_low_rank(pre_model_mat, model_mat, r):
    target_device = model_mat.device
    target_dtype = model_mat.dtype
    model_mat_cpu = model_mat.detach().cpu().to(dtype=torch.float64)
    pre_model_mat_cpu = pre_model_mat.detach().cpu().to(dtype=torch.float64)
    K = pre_model_mat_cpu - model_mat_cpu
    if not torch.isfinite(K).all():
        raise ValueError("restore_low_rank received non-finite values in the weight difference")
    U, S, Vt = torch.linalg.svd(K, full_matrices=False)
    K_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
    restored_matrix = model_mat_cpu + K_r
    restored_matrix = restored_matrix.to(device=target_device, dtype=target_dtype)
    K_r_t = (-K_r).to(dtype=target_dtype)
    return restored_matrix, K_r_t

def restore_orthogonal(pre_model_mat, model_mat):
    target_device = model_mat.device
    target_dtype = model_mat.dtype
    model_mat_cpu = model_mat.detach().cpu().to(dtype=torch.float64)
    pre_model_mat_cpu = pre_model_mat.detach().cpu().to(dtype=torch.float64)
    U, _, Vt = torch.linalg.svd(pre_model_mat_cpu.T @ model_mat_cpu, full_matrices=False)
    K = Vt.T @ U.T
    restored_matrix = (model_mat_cpu @ K).to(device=target_device, dtype=target_dtype)
    K_inv = K.T.to(dtype=target_dtype)
    return restored_matrix, K_inv


def _iter_float_state_dict_keys(state_dict_a, state_dict_b):
    keys = []
    for key, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b.get(key)
        if tensor_b is None:
            continue
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            continue
        if tensor_a.shape != tensor_b.shape:
            continue
        if not torch.is_floating_point(tensor_a) or not torch.is_floating_point(tensor_b):
            continue
        keys.append(key)
    return keys


def interpolate_state_dict(state_dict_a, state_dict_b, alpha):
    interpolated_state_dict = {}
    for key, tensor_a in state_dict_a.items():
        tensor_b = state_dict_b.get(key)
        if (
            tensor_b is not None
            and isinstance(tensor_a, torch.Tensor)
            and isinstance(tensor_b, torch.Tensor)
            and tensor_a.shape == tensor_b.shape
            and torch.is_floating_point(tensor_a)
            and torch.is_floating_point(tensor_b)
        ):
            tensor_a_cpu = tensor_a.detach().cpu()
            tensor_b_cpu = tensor_b.detach().cpu().to(dtype=tensor_a_cpu.dtype)
            interpolated_state_dict[key] = (1 - alpha) * tensor_a_cpu + alpha * tensor_b_cpu
        elif isinstance(tensor_a, torch.Tensor):
            interpolated_state_dict[key] = tensor_a.detach().cpu().clone()
        else:
            interpolated_state_dict[key] = tensor_a
    return interpolated_state_dict


def compute_weight_distance(state_dict_a, state_dict_b, metric="l2", eps=1e-12):
    if metric != "l2":
        raise ValueError(f"Unsupported distance metric: {metric}")

    total = 0.0
    for key in _iter_float_state_dict_keys(state_dict_a, state_dict_b):
        diff = state_dict_a[key].detach().cpu() - state_dict_b[key].detach().cpu()
        total += torch.sum(diff * diff).item()
    return float(np.sqrt(max(total, eps)))


def sample_orthogonal_direction(state_dict_a, state_dict_b, seed=42, eps=1e-12):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = {}
    dot = 0.0
    delta_norm_sq = 0.0

    for key in _iter_float_state_dict_keys(state_dict_a, state_dict_b):
        delta = state_dict_b[key].detach().cpu() - state_dict_a[key].detach().cpu()
        random_tensor = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
        direction[key] = random_tensor
        dot += torch.sum(random_tensor * delta).item()
        delta_norm_sq += torch.sum(delta * delta).item()

    if delta_norm_sq <= eps:
        raise ValueError("The two state_dicts are identical; cannot sample an orthogonal direction.")

    projection_scale = dot / delta_norm_sq
    orthogonal_norm_sq = 0.0
    for key in direction:
        delta = state_dict_b[key].detach().cpu() - state_dict_a[key].detach().cpu()
        orthogonal_tensor = direction[key] - projection_scale * delta
        direction[key] = orthogonal_tensor
        orthogonal_norm_sq += torch.sum(orthogonal_tensor * orthogonal_tensor).item()

    orthogonal_norm = float(np.sqrt(max(orthogonal_norm_sq, eps)))
    for key in direction:
        direction[key] = direction[key] / orthogonal_norm
    return direction


def apply_direction_to_state_dict(base_state_dict, direction, radius):
    updated_state_dict = {}
    for key, base_tensor in base_state_dict.items():
        if key in direction:
            updated_state_dict[key] = base_tensor.detach().cpu() + radius * direction[key].to(dtype=base_tensor.dtype)
        elif isinstance(base_tensor, torch.Tensor):
            updated_state_dict[key] = base_tensor.detach().cpu().clone()
        else:
            updated_state_dict[key] = base_tensor
    return updated_state_dict

def fix_factor(num, mini=1.0, max=6.0):
    if(num < mini):
        return mini
    elif(num > max):
        return max
    else:
        return num
    

def init_obfus_model(args, num_labels, obfus):
    if obfus == "tsqp":
        if os.path.exists(args.weight_dir_tsqp):
            model = AutoModelForSequenceClassification.from_pretrained(
                args.weight_dir_tsqp,  
                num_labels=num_labels,
                use_safetensors=True 
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.weight_dir, 
                num_labels=num_labels,
                use_safetensors=True 
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.weight_dir, 
            num_labels=num_labels,
            use_safetensors=True 
        )
    return model