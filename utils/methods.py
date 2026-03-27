from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt

def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            layer_name = name.rsplit(".")[3]
            num_rows = w_q.shape[1]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation
            ob_w_q = w_q[:,permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_k = w_k[:,permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            ob_w_v = w_v[:,permutation]
            module.data = ob_w_v
        elif "attention.output.dense.weight" in name:
            w_o = module.data
            layer_name = name.rsplit(".")[3]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)
            ob_o = w_o[inv_perm, :]
            module.data = ob_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            if(w_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                permutation = layer_permutations[layer_name]
                ob_proj = w_proj[:, permutation]
                module.data = ob_proj
    return model, layer_permutations, rows

def attack_translinkguard(model, pre_model, rows):
    set_seed()
    restore_perm = {}        
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            layer_name = name.rsplit(".")[3]
            pre_wq = pre_model.state_dict()[name]
            perm, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wq = ob_wq[:, inv_perm]
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wk = ob_wk[:, inv_perm]
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wv = ob_wv[:, inv_perm]
            module.data = restore_wv
        elif "attention.output.dense.weight" in name:
            ob_o = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            restore_o = ob_o[perm, :]
            module.data = restore_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            if(ob_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_proj = ob_proj[:, inv_perm]
                module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model        
def attack_translinkguard2(model, pre_model, rows):
    set_seed()
    restore_perm = {}        
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            layer_name = name.rsplit(".")[3]
            pre_wq = pre_model.state_dict()[name]
            perm1, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            perm, _, restore_wq = col_restore_perm2(pre_wq, ob_wq)
            print(f"layer {layer_name} 恢复的两种方法的相似度: {np.mean(perm1==perm)}")
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wq = ob_wq[:, inv_perm]
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wk = ob_wk[:, inv_perm]
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_wv = ob_wv[:, inv_perm]
            module.data = restore_wv
        elif "attention.output.dense.weight" in name:
            ob_o = module.data
            layer_name = name.rsplit(".")[3]
            perm = restore_perm[layer_name]
            restore_o = ob_o[perm, :]
            module.data = restore_o
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            if(ob_proj.shape[1] == rows):
                layer_name = name.rsplit(".")[3]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_proj = ob_proj[:, inv_perm]
                module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model       

def ob_tsqp(model):
    set_seed()  
    scaling_factors = {}  
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            scale_q = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            module.data = w_q
            scaling_factors[name] = scale_q
        elif "key.weight" in name:
            w_k = module.data
            scale_k = 1 + 5 * torch.rand(1).item()
            w_k *= scale_k
            module.data = w_k
            scaling_factors[name] = scale_k
        elif "value.weight" in name:
            w_v = module.data
            scale_v = 1 + 5 * torch.rand(1).item()
            w_v *= scale_v
            module.data = w_v
            scaling_factors[name] = scale_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
    return model, scaling_factors

def attack_tsqp(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:      
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_soter(model, pre_model):
    set_seed()
    scaling_factors = {}  
    init_layers = {}
    # 随机选取一些权重块加载pre_model的权重,比例为20%
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            init_layers[name] = module.data
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    
    for name, module in model.named_parameters():
        if name in layers_to_replace:
            module.data = pre_model.state_dict()[name].data
    for name, module in model.named_parameters():
        if name not in layers_to_replace:
            if "query.weight" in name:
                w_q = module.data
                scale_q = 1 + 5 * torch.rand(1).item()
                w_q *= scale_q
                module.data = w_q
                scaling_factors[name] = scale_q
            elif "key.weight" in name:
                w_k = module.data
                scale_k = 1 + 5 * torch.rand(1).item()
                w_k *= scale_k
                module.data = w_k
                scaling_factors[name] = scale_k
            elif "value.weight" in name:
                w_v = module.data
                scale_v = 1 + 5 * torch.rand(1).item()
                w_v *= scale_v
                module.data = w_v
                scaling_factors[name] = scale_v
            elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
                w_proj = module.data
                scale_proj = 1 + 5 * torch.rand(1).item()
                w_proj *= scale_proj
                module.data = w_proj
                scaling_factors[name] = scale_proj
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            pre_q = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            new_w_q = w_q/k
            restore_scaling_factors[name] = k
            module.data = new_w_q
        elif "key.weight" in name:
            w_k = module.data
            pre_k = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            new_w_k = w_k/k
            restore_scaling_factors[name] = k
            module.data = new_w_k
        elif "value.weight" in name:
            w_v = module.data
            pre_v = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_v = w_v/k
            restore_scaling_factors[name] = k
            module.data = new_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def attack_soter2(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            k = pre_proj.flatten().dot(w_proj.flatten()) / w_proj.flatten().dot(w_proj.flatten())
            new_w_proj = w_proj*k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors
    

def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[1]
            ratios_q = []
            for i in range(num_rows):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[:,i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_q = w_q[:, permutation]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[1]
            ratios_k = []
            for i in range(num_rows):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[:,i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_k = w_k[:, permutation]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[1]
            ratios_v = []
            for i in range(num_rows):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[:,i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_v = w_v[:, permutation]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_proj = w_proj[:, permutation]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_shadownet(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq[:,i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[1]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
                restore_wk[:,i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[1]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
                restore_wv[:,i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = col_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[1]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[:,i]).item()/torch.var(pre_proj[:,i]).item()))
                restore_proj[:,i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_shadownet2(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq1[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq1[:,i] /= ratio_q
            restore_wq = col_restore_perm_and_scale(pre_wq, ob_wq)
            print(f"layer {name} 恢复的两种方法的相似度: {np.mean(np.abs(restore_wq1.numpy().flatten()-restore_wq.numpy().flatten()) < 1e-4)}")
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            # _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            # for i in range(ob_wk.shape[1]):
            #     ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
            #     restore_wk[:,i] /= ratio_k
            restore_wk = col_restore_perm_and_scale(pre_wk, ob_wk)
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            # _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            # for i in range(ob_wv.shape[1]):
            #     ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
            #     restore_wv[:,i] /= ratio_v
            restore_wv = col_restore_perm_and_scale(pre_wv, ob_wv)
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            # _, _, restore_proj = col_restore_perm(pre_proj, ob_proj)
            # for i in range(ob_proj.shape[1]):
            #     ratio = fix_factor(sqrt(torch.var(restore_proj[:,i]).item()/torch.var(pre_proj[:,i]).item()))
            #     restore_proj[:,i] /= ratio
            restore_proj = col_restore_perm_and_scale(pre_proj, ob_proj)
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def ob_LoRO(model, r=8, noise=1e-1):
    set_seed()
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w = module.data
            d1, d2 = w.shape
            # 生成低秩矩阵的两个因子
            A = torch.randn(d1, r) * noise
            A = A.to(w.device)
            B = torch.randn(r, d2) * noise
            B = B.to(w.device)
            low_rank_matrix = torch.matmul(A, B)
            # 将低秩矩阵加到原始权重上
            w += low_rank_matrix
            module.data = w
    return model

def attack_LoRO(model, pre_model, vic_model, r=8):
    set_seed()
    for name, module in model.named_parameters():
        ob_w = module.data
        pre_w = pre_model.state_dict()[name].data
        vic_w = vic_model.state_dict()[name].data
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            restore_w = restore_low_rank(pre_w, ob_w, r)
            module.data = restore_w
            print(f"layer {name} :")
            error = torch.norm(vic_w - pre_w, p='fro')
            print(f"原始模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - pre_w, p='fro')
            print(f"恢复模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - vic_w, p='fro')
            print(f"恢复模型与原始模型的误差: {error.item()}")
        else:
            module.data = pre_model.state_dict()[name].data
    return model

def ob_obfuscatune(model):
    set_seed()
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w = module.data
            # W @ Q Q is orthogonal random matrices
            d1, d2 = w.shape
            Q, _ = torch.linalg.qr(torch.randn(d2, d2))
            Q = Q.to(w.device)
            w = torch.matmul(w, Q)
            module.data = w
    return model
    

def attack_obfuscatune(model, pre_model, vic_model):
    set_seed()
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            vic_w = vic_model.state_dict()[name].data
            restore_w = restore_orthogonal(pre_w, ob_w)
            print(f"layer {name} :")
            error = torch.norm(vic_w - pre_w, p='fro')
            print(f"原始模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - pre_w, p='fro')
            print(f"恢复模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - vic_w, p='fro')
            print(f"恢复模型与原始模型的误差: {error.item()}")
            module.data = restore_w
        else:
            module.data = pre_model.state_dict()[name].data
    return model
    

def ob_tempo(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_cols = w_q.shape[0]
            ratios_q = []
            for i in range(num_cols):
                ratio_q = 1 + 5 * torch.rand(1).item()
                w_q[i] *= ratio_q
                ratios_q.append(ratio_q)
            scaling_factors[name] = ratios_q
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_q = w_q[permutation,:]
            module.data = ob_w_q
        elif "key.weight" in name:
            w_k = module.data
            num_cols = w_k.shape[0]
            ratios_k = []
            for i in range(num_cols):
                ratio_k = 1 + 5 * torch.rand(1).item()
                w_k[i] *= ratio_k
                ratios_k.append(ratio_k)
            scaling_factors[name] = ratios_k
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_k = w_k[permutation,:]
            module.data = ob_w_k
        elif "value.weight" in name:
            w_v = module.data
            num_cols = w_v.shape[0]
            ratios_v = []
            for i in range(num_cols):
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_v[i] *= ratio_v
                ratios_v.append(ratio_v)
            scaling_factors[name] = ratios_v
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_v = w_v[permutation,:]
            module.data = ob_w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_proj = w_proj[permutation,:]
            module.data = ob_proj
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item()))
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item()))
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item()))
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = fix_factor(sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item()))
                restore_proj[i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[0]
            device = w_q.device
            coeff = torch.randint(0,5,(w_q.shape[0],), device=device)
            mask = torch.matmul(w_q.T, coeff.float())
            layer_masks[name] = mask
            ratios_q = []
            ratios_q2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= ratio2
                mask_qi = mask * ratio
                w_q[i] += mask_qi
                ratios_q.append(ratio)
                ratios_q2.append(ratio2)
            layer_factors[name] = ratios_q
            weight_factors[name] = ratios_q2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_q = w_q[permutation]
            module.data = w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[0]
            device = w_k.device
            coeff = torch.randint(0,5,(w_k.shape[0],), device=device)
            mask = torch.matmul(w_k.T, coeff.float())
            layer_masks[name] = mask
            ratios_k = []
            ratios_k2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_k[i] *= ratio2
                mask_ki = mask * ratio
                w_k[i] += mask_ki
                ratios_k.append(ratio)
                ratios_k2.append(ratio2)
            layer_factors[name] = ratios_k
            weight_factors[name] = ratios_k2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_k = w_k[permutation]
            module.data = w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[0]
            device = w_v.device
            coeff = torch.randint(0,5,(w_v.shape[0],), device=device)
            mask = torch.matmul(w_v.T, coeff.float())
            layer_masks[name] = mask
            ratios_v = []
            ratios_v2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_v[i] *= ratio2
                mask_vi = mask * ratio
                w_v[i] += mask_vi
                ratios_v.append(ratio)
                ratios_v2.append(ratio2)
            layer_factors[name] = ratios_v
            weight_factors[name] = ratios_v2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_v = w_v[permutation]
            module.data = w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[0]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_proj = w_proj[permutation]
            module.data = w_proj
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def attack_arrowcloak(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item())
                restore_proj[i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_arrowcloak2(model, pre_model, orig_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            print(f"正在恢复{name}...")
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            
            A_obf = ob_w.cpu().numpy()
            A_pub = pre_w.cpu().numpy()
            
            path = "./data/weight_SST2"
            if os.path.exists(f"{path}/A_rec_{name.replace('.','_')}.npy"):
                print("恢复成功!")
                restore_w = np.load(f"{path}/A_rec_{name.replace('.','_')}.npy")
            # elif True:
            #     print("使用公共模型权重作为恢复结果")
            #     restore_w = A_pub
            else:
                A_real = orig_model.state_dict()[name].data.cpu().numpy()
                np.save(f"{path}/A_obf_{name.replace('.','_')}.npy", A_obf)
                np.save(f"{path}/A_pub_{name.replace('.','_')}.npy", A_pub)
                np.save(f"{path}/A_real_{name.replace('.','_')}.npy", A_real)
                
                A_real = A_real.astype(np.float64)
                A_pub = A_pub.astype(np.float64)
                A_obf = A_obf.astype(np.float64)
                
                ratio = 1.0 * A_obf.shape[0] / A_obf.shape[1]
                rho = ratio**2
                max_iter = int(1000 / ratio)
                L_precise, S_precise = solve_admm_structured(A_obf, A_pub, rho=rho, max_iter=max_iter, alpha=1.6)
                restore_w = (L_precise + S_precise) @ A_obf
                
                if np.linalg.norm(restore_w - A_real, 'fro') < np.linalg.norm(A_pub - A_real, 'fro') / 3:
                    print("恢复成功!")
                    np.save(f"{path}/A_rec_{name.replace('.','_')}.npy", restore_w)
                else:
                    print("恢复失败!")
                
                if np.linalg.norm(restore_w - A_pub, 'fro') > 3:
                    print("使用公共模型权重作为恢复结果")
                    restore_w = A_pub
            
            restore_w = torch.from_numpy(restore_w).to(ob_w.device)
            restore_w = restore_w.type_as(ob_w)
            module.data = restore_w
            error = np.linalg.norm(pre_model.state_dict()[name].data.cpu().numpy() - orig_model.state_dict()[name].data.cpu().numpy())
            print(f"公共模型与原始模型的误差: {error:.4e}")
            error = np.linalg.norm(module.data.cpu().numpy() - pre_model.state_dict()[name].data.cpu().numpy())
            print(f"恢复后与公共模型的误差: {error:.4e}")
            error = np.linalg.norm(module.data.cpu().numpy() - orig_model.state_dict()[name].data.cpu().numpy())
            print(f"恢复后与原始模型的误差: {error:.4e}")
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
def rank1_permuted_diagonal_decomposition(A, S_init=None, max_iter=50, tol=1e-6):
    m, n = A.shape
    
    if S_init is not None:
        S = S_init.copy()
    else:
        S = np.zeros_like(A)
    
    prev_error = np.inf
    
    for iteration in range(max_iter):
        # --- 步骤 1: 更新 L (Rank-1) ---
        # R = A - S
        R = A - S
        # SVD 分解
        # U, s, Vt = svd(R, full_matrices=False, overwrite_a=True, check_finite=False)
        U, s, Vt = svds(R, k=1)
        # 取最大奇异值
        L = s[0] * np.outer(U[:, 0], Vt[0, :])
        
        E = A - L
        
        cost_matrix = -(E ** 2)
        # cost_matrix = -np.abs(E)
        
        # 使用匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 构造新的 S
        S_new = np.zeros_like(A)
        S_new[row_ind, col_ind] = E[row_ind, col_ind]
        
        # --- 检查收敛 ---
        error = np.linalg.norm(A - L - S_new, 'fro')
        if abs(prev_error - error) < tol:
            # print(f"Converged at iteration {iteration}")
            break
        prev_error = error
        S = S_new

    return L, S

def solve_admm_structured(A, B, rho=1.0, max_iter=500, alpha=1.6):
    m, n = A.shape
    print(f"ADMM Structured Solver: A shape = {A.shape}, rho = {rho}, max_iter = {max_iter}, alpha = {alpha}")
    
    # --- 初始化变量 ---
    Z = np.zeros((m, m))
    L = np.zeros((m, m))
    S = np.zeros((m, m))
    Gamma = np.zeros((m, m))
    
    # 预计算缓存，加速 Z 的更新
    # (A^T A + rho*I) 的逆
    AAt_rhoI_inv = np.linalg.inv(A @ A.T + rho * np.eye(m))
    BAt = B @ A.T
    
    min_loss = np.inf
    best_L, best_S = None, None
    for it in range(max_iter):
        # 1. 更新 Z (Global Step)
        Target_Z = (L + S) - (1/rho) * Gamma # 这里 D+uv^T 是结构项
        Z = (BAt + rho * Target_Z) @ AAt_rhoI_inv
        Z = alpha * Z + (1.0 - alpha) * (L + S)
        
        # 2. 更新 D, u, v (Local Structure Step)
        # 此时目标是拟合 T = Z + (1/rho)*Gamma
        T = Z + (1/rho) * Gamma
        
        # 2
        L, S = rank1_permuted_diagonal_decomposition(T, S_init=S, max_iter=10, tol=1e-4)
        
        # 3. 更新对偶变量 Gamma
        Residual_Constraint = Z - (L + S)
        Gamma = Gamma + rho * Residual_Constraint
        
        # 计算 Loss 观察
        curr_loss = np.linalg.norm((L + S) @ A - B, 'fro')
        prim_res = np.linalg.norm(Residual_Constraint, 'fro')
        
        if it % 100 == 0:
            print(f"Iter {it:2d}: Objective Loss = {curr_loss:.6f}, Primal Residual = {prim_res:.6f}")
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_L, best_S = L.copy(), S.copy()
        

    return best_L, best_S

from sklearn.metrics.pairwise import cosine_similarity
def cluster_vectors(vectors, cluster_size=4):
    index_pairs = np.array([np.array([i]) for i in range(len(vectors))])
    iter = 1
    while(iter<cluster_size):
        iter*=2
        cos_sim_matrix = cosine_similarity(vectors)
        sum_cos_sim_dis = np.mean(cos_sim_matrix, axis=0)
        sorted_indices = np.argsort(sum_cos_sim_dis)[::-1]
        np.fill_diagonal(cos_sim_matrix, np.inf)  
        pairs = []
        index_pair = []
        repeat_index = []
        for i in sorted_indices:
            if i in repeat_index:
                continue
            j = np.argmin(cos_sim_matrix[i])
            repeat_index.append(j)
            cos_sim_matrix[i, :] = np.inf
            cos_sim_matrix[:, i] = np.inf
            cos_sim_matrix[j, :] = np.inf
            cos_sim_matrix[:, j] = np.inf
            index_pair.append(np.concatenate((index_pairs[i], index_pairs[j])))
            pairs.append(np.mean([vectors[i], vectors[j]],axis=0))
        vectors = pairs
        index_pairs = index_pair 
        # print(index_pair) 
    return index_pairs


def ob_groupcover(model, size=4):
    set_seed()
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w = module.data
            device = w.device
            d1, d2 = w.shape
            
            w = w.cpu().numpy()
            
            cluster_index = cluster_vectors(w, cluster_size=size)
            random_coeff_list = [[] for _ in range(d1)]
            new_w = np.zeros_like(w)
            for idlist in cluster_index:
                new_kernels = []
                for i in idlist:
                    random_coeffs = np.random.randint(1, 100, size=size)
                    random_coeff_list[i] = random_coeffs

                    new_kernel = sum(coeff * w[idlist[j], :] for j, coeff in enumerate(random_coeffs))
                    new_kernels.append(new_kernel)

                for index, idx in enumerate(idlist):
                    new_w[idx, :]= new_kernels[index]
            
            new_w = torch.tensor(new_w).to(device)
            permutation = torch.randperm(d2)
            new_w = new_w[:, permutation]
            
            module.data = new_w
            
    return model
    

def attack_groupcover(model, pre_model, vic_model):
    set_seed()
    for name, module in model.named_parameters():
        if "query.weight" in name or "key.weight" in name or "value.weight" in name or "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            vic_w = vic_model.state_dict()[name].data
            path = "./data/weight_SST2"
            if os.path.exists(f"{path}/A_GC_{name.replace('.','_')}.npy"):
                restore_w = np.load(f"{path}/A_GC_{name.replace('.','_')}.npy")
            else:
                # TODO
                restore_w = pre_w.cpu().numpy()
            restore_w = torch.from_numpy(restore_w).to(ob_w.device)
            restore_w = restore_w.type_as(ob_w)
            module.data = restore_w
            error = np.linalg.norm(pre_w.cpu().numpy() - vic_w.cpu().numpy())
            print(f"公共模型与原始模型的误差: {error:.4e}")
            error = np.linalg.norm(module.data.cpu().numpy() - pre_w.cpu().numpy())
            print(f"恢复后与公共模型的误差: {error:.4e}")
            error = np.linalg.norm(module.data.cpu().numpy() - vic_w.cpu().numpy())
            print(f"恢复后与原始模型的误差: {error:.4e}")
        else:
            module.data = pre_model.state_dict()[name].data
    return model