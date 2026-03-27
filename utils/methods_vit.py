import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils_vit import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from matplotlib import pyplot as plt

def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            num_rows = w_q.shape[1]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation
            ob_w_q = w_q[:,permutation]
            ob_w_k = w_k[:,permutation]
            ob_w_v = w_v[:,permutation]
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        if "attn.proj.weight" in name:
            w_proj = module.data        
            layer_name = name.rsplit(".", 3)[0]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)    
            ob_proj = w_proj[inv_perm]
            module.data = ob_proj
        if "mlp.fc1.weight" in name:
            w_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            permutation = layer_permutations[layer_name]
            ob_fc1 = w_fc1[:,permutation]
            module.data = ob_fc1
        if "mlp.fc2.weight" in name:
            w_fc2 = module.data
            if(w_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]         
                permutation = layer_permutations[layer_name] 
                ob_fc2 = w_fc2[:,permutation]
                module.data = ob_fc2             
    return model, layer_permutations, rows

def attack_translinkguard(model, pre_model, rows):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            pre_wq, _, _ = pre_model.state_dict()[name].chunk(3, dim=0)
            perm, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wk = ob_wk[:, inv_perm]
            restore_wv = ob_wv[:, inv_perm]
            restore_w = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
            module.data = restore_w
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            restore_wo = ob_wo[perm]
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_fc1 = ob_fc1[:, inv_perm]
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            if(ob_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_fc2 = ob_fc2[:,inv_perm]
                module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data

    return model    

def attack_translinkguard2(model, pre_model, rows):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            layer_name = name.rsplit(".", 3)[0]
            pre_wq, _, _ = pre_model.state_dict()[name].chunk(3, dim=0)
            perm1, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            perm, _, restore_wq = col_restore_perm2(pre_wq, ob_wq)
            print(f"layer {layer_name} 恢复的两种方法的相似度: {np.mean(perm1==perm)}")
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wk = ob_wk[:, inv_perm]
            restore_wv = ob_wv[:, inv_perm]
            restore_w = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
            module.data = restore_w
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            restore_wo = ob_wo[perm]
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            layer_name = name.rsplit(".", 3)[0]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_fc1 = ob_fc1[:, inv_perm]
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            if(ob_fc2.shape[1] == rows):
                layer_name = name.rsplit(".", 4)[1]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_fc2 = ob_fc2[:,inv_perm]
                module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data

    return model    

def ob_tsqp(model):
    set_seed() 
    scaling_factors = {}  
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            scale_q = 1 + 5 * torch.rand(1).item()
            scale_k = 1 + 5 * torch.rand(1).item()
            scale_v = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            w_k *= scale_k
            w_v *= scale_v
            module.data = torch.cat([w_q, w_k, w_v], dim=0)
            scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
        elif "attn.proj.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            scale_fc1 = 1 + 5 * torch.rand(1).item()
            w_fc1 *= scale_fc1
            module.data = w_fc1
            scaling_factors[name] = scale_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            scale_fc2 = 1 + 5 * torch.rand(1).item()
            w_fc2 *= scale_fc2
            module.data = w_fc2
            scaling_factors[name] = scale_fc2
    return model, scaling_factors

def attack_tsqp(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_fc2).item()/torch.var(pre_fc2).item()))
            new_w_fc2 = w_fc2/k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_soter(model, pre_model):
    set_seed()
    scaling_factors = {}  
    init_layers = {}
    ## 随机选取一些权重块加载pre_model的权重
    ## 比例为20%
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            init_layers[name] = module
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    ## 替换选中层的权重
    for name, module in model.named_parameters():
        if name in layers_to_replace:
            module.data = pre_model.state_dict()[name].data           
    for name, module in model.named_parameters():
        if name not in layers_to_replace:
            if "qkv.weight" in name:
                w_q, w_k, w_v = module.data.chunk(3, dim=0)
                scale_q = 1 + 5 * torch.rand(1).item()
                scale_k = 1 + 5 * torch.rand(1).item()
                scale_v = 1 + 5 * torch.rand(1).item()
                w_q *= scale_q
                w_k *= scale_k
                w_v *= scale_v
                module.data = torch.cat([w_q, w_k, w_v], dim=0)
                scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
            elif "attn.proj.weight" in name:
                w_proj = module.data
                scale_proj = 1 + 5 * torch.rand(1).item()
                w_proj *= scale_proj
                module.data = w_proj
                scaling_factors[name] = scale_proj
            elif "mlp.fc1.weight" in name:
                w_fc1 = module.data
                scale_fc1 = 1 + 5 * torch.rand(1).item()
                w_fc1 *= scale_fc1
                module.data = w_fc1
                scaling_factors[name] = scale_fc1
            elif "mlp.fc2.weight" in name:
                w_fc2 = module.data
                scale_fc2 = 1 + 5 * torch.rand(1).item()
                w_fc2 *= scale_fc2
                module.data = w_fc2
                scaling_factors[name] = scale_fc2
            else:
                head = module.data
                scale_head = 1 + 5 * torch.rand(1).item()
                head *= scale_head
                module.data = head
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    ## 记录与pre_model相同的层
    layers_pretrained = []
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            pre_q = pre_q.to(w_q.device)
            ## 如果pre_q与w_q相同，则不需要恢复
            if torch.allclose(w_q, pre_q, rtol=1e-5):
                layers_pretrained.append(name)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            if torch.allclose(w_proj, pre_proj, rtol=1e-5):
                layers_pretrained.append(name)
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            pre_fc1 = pre_fc1.to(w_fc1.device)
            if torch.allclose(w_fc1, pre_fc1, rtol=1e-5):
                layers_pretrained.append(name)
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            pre_fc2 = pre_fc2.to(w_fc2.device)
            if torch.allclose(w_fc2, pre_fc2, rtol=1e-5):
                layers_pretrained.append(name)
            k =  fix_factor(sqrt(torch.var(w_fc2).item()/torch.var(pre_fc2).item()))
            new_w_fc2 = w_fc2/k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors, layers_pretrained

def attack_soter2(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    ## 记录与pre_model相同的层
    layers_pretrained = []
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=0)
            pre_q = pre_q.to(w_q.device)
            pre_k = pre_k.to(w_k.device)
            pre_v = pre_v.to(w_v.device)
            ## 如果pre_q与w_q相同，则不需要恢复
            if torch.allclose(w_q, pre_q, rtol=1e-5):
                layers_pretrained.append(name)
            # d = ΣΣbijaij / ΣΣaij^2
            k = 1/fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k1 = pre_q.flatten().dot(w_q.flatten()) / w_q.flatten().dot(w_q.flatten())
            print(f"Layer {name} k from var: {k}, k from dot: {k1}")
            k2 = pre_k.flatten().dot(w_k.flatten()) / w_k.flatten().dot(w_k.flatten())
            k3 = pre_v.flatten().dot(w_v.flatten()) / w_v.flatten().dot(w_v.flatten())
            new_w_q = w_q*k1
            new_w_k = w_k*k2
            new_w_v = w_v*k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=0)
        elif "attn.proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            pre_proj = pre_proj.to(w_proj.device)
            if torch.allclose(w_proj, pre_proj, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_proj.flatten().dot(w_proj.flatten()) / w_proj.flatten().dot(w_proj.flatten())
            new_w_proj = w_proj*k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            pre_fc1 = pre_fc1.to(w_fc1.device)
            if torch.allclose(w_fc1, pre_fc1, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_fc1.flatten().dot(w_fc1.flatten()) / w_fc1.flatten().dot(w_fc1.flatten())
            new_w_fc1 = w_fc1*k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            pre_fc2 = pre_fc2.to(w_fc2.device)
            if torch.allclose(w_fc2, pre_fc2, rtol=1e-5):
                layers_pretrained.append(name)
            k = pre_fc2.flatten().dot(w_fc2.flatten()) / w_fc2.flatten().dot(w_fc2.flatten())
            new_w_fc2 = w_fc2*k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors, layers_pretrained
def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            num_rows = w_q.shape[1]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_rows):
                ratio_q = 1 + 5 * torch.rand(1).item()
                ratio_k = 1 + 5 * torch.rand(1).item()
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_q[:,i] *= ratio_q
                w_k[:,i] *= ratio_k
                w_v[:,i] *= ratio_v
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
            scaling_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            permutation_q = torch.randperm(num_rows)
            permutation_k = torch.randperm(num_rows)
            permutation_v = torch.randperm(num_rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[:,permutation_q]
            ob_w_k = w_k[:,permutation_k]
            ob_w_v = w_v[:,permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        elif "attn.proj.weight" in name:
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
            ob_w_proj = w_proj[:,permutation]
            module.data = ob_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            num_rows = w_fc1.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[:,permutation]
            module.data = ob_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            num_rows = w_fc2.shape[1]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[:,permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, scaling_factors

def attack_shadownet(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item()))
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item()))
                restore_wq[:,i] /= ratio_q
                restore_wk[:,i] /= ratio_k
                restore_wv[:,i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = col_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[1]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[:,i]).item()/torch.var(pre_wo[:,i]).item()))
                restore_wo[:,i] /= ratio_o
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = col_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[1]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[:,i]).item()/torch.var(pre_fc1[:,i]).item()))
                restore_fc1[:,i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = col_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item()))
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2   
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def attack_shadownet2(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq1 = col_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[1]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq1[:,i]).item()/torch.var(pre_wq[:,i]).item()))
                restore_wq1[:,i] /= ratio_q
            restore_wq = col_restore_perm_and_scale(pre_wq, ob_wq)
            print(f"layer {name} 恢复的两种方法的相似度: {np.mean(np.abs(restore_wq1.cpu().numpy().flatten()-restore_wq.cpu().numpy().flatten()) < 1e-4)}")
            restore_wk = col_restore_perm_and_scale(pre_wk, ob_wk)
            restore_wv = col_restore_perm_and_scale(pre_wv, ob_wv)
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            restore_wo = col_restore_perm_and_scale(pre_wo, ob_wo)
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            restore_fc1 = col_restore_perm_and_scale(pre_fc1, ob_fc1)
            module.data = restore_fc1
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            restore_fc2 = col_restore_perm_and_scale(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item()))
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2   
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm
def ob_LoRO(model, r=8, noise=1e-1):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
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

def attack_LoRO(model, pre_model, vic_model=None, r=8):
    set_seed()
    for name, module in model.named_parameters():
        ob_w = module.data
        pre_w = pre_model.state_dict()[name].data
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            restore_w = restore_low_rank(pre_w, ob_w, r)
            module.data = restore_w
        else:
            module.data = pre_model.state_dict()[name].data
        restore_w = module.data
        if vic_model is not None:
            vic_w = vic_model.state_dict()[name].data
            print(f"layer {name} :")
            error = torch.norm(vic_w - pre_w, p='fro')
            print(f"原始模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - pre_w, p='fro')
            print(f"恢复模型与公共模型的误差: {error.item()}")
            error = torch.norm(restore_w.cpu() - vic_w, p='fro')
            print(f"恢复模型与原始模型的误差: {error.item()}")
    return model

def ob_obfuscatune(model):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            w = module.data
            d1, d2 = w.shape
            Q, _ = torch.linalg.qr(torch.randn(d2, d2))
            Q = Q.to(w.device)
            w = torch.matmul(w, Q)
            # Q, _ = torch.linalg.qr(torch.randn(d1, d1))
            # Q = Q.to(w.device)
            # w = torch.matmul(Q, w)
            module.data = w
    return model
    

def attack_obfuscatune(model, pre_model, vic_model=None):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            restore_w = restore_orthogonal(pre_w, ob_w)
            if vic_model is not None:
                vic_w = vic_model.state_dict()[name].data
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
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            num_cols = w_q.shape[0]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_cols):
                ratio_q = 1 + 5 * torch.rand(1).item()
                ratio_k = 1 + 5 * torch.rand(1).item()
                ratio_v = 1 + 5 * torch.rand(1).item()
                w_q[i] *= ratio_q
                w_k[i] *= ratio_k
                w_v[i] *= ratio_v
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
            scaling_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            permutation_q = torch.randperm(num_cols)
            permutation_k = torch.randperm(num_cols)
            permutation_v = torch.randperm(num_cols)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        elif "attn.proj.weight" in name:
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
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        elif "mlp.fc1.weight" in name:
            w_fc1 = module.data
            num_cols = w_fc1.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        elif "mlp.fc2.weight" in name:
            w_fc2 = module.data
            num_cols = w_fc2.shape[0]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[0]):
                ratio_q = fix_factor(sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item()))
                ratio_k = fix_factor(sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item()))
                ratio_v = fix_factor(sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item()))
                restore_wq[i] /= ratio_q
                restore_wk[i] /= ratio_k
                restore_wv[i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        elif "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[0]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item()))
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        elif "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item()))
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1           
        elif "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = row_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[0]):
                ratio_fc2 = fix_factor(sqrt(torch.var(restore_fc2[i]).item()/torch.var(pre_fc2[i]).item()))
                restore_fc2[i] /= ratio_fc2
            module.data = restore_fc2   
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
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            device = w_q.device
            coeff_q, coeff_k, coeff_v = torch.randint(0,5,(w_q.shape[0],), device=device), torch.randint(0,5,(w_k.shape[0],), device=device), torch.randint(0,5,(w_v.shape[0],), device=device)
            mask_q, mask_k, mask_v = torch.matmul(w_q.T, coeff_q.float()), torch.matmul(w_k.T, coeff_k.float()), torch.matmul(w_v.T, coeff_v.float())
            layer_masks[name] = {"q": mask_q, "k": mask_k, "v": mask_v}
            ratios_q, ratios_k, ratios_v = [], [], []
            ratios_q2, ratios_k2, ratios_v2 = [], [], []
            for i in range(w_q.shape[0]):
                ratio_q, ratio_k, ratio_v = (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float()
                weight_q, weight_k, weight_v = (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= weight_q
                w_k[i] *= weight_k
                w_v[i] *= weight_v
                mask_qi, mask_ki, mask_vi = mask_q * ratio_q, mask_k * ratio_k, mask_v * ratio_v
                w_q[i] += mask_qi
                w_k[i] += mask_ki
                w_v[i] += mask_vi
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
                ratios_q2.append(weight_q)
                ratios_k2.append(weight_k)
                ratios_v2.append(weight_v)
            layer_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            weight_factors[name] = {"q": ratios_q2, "k": ratios_k2, "v": ratios_v2}
            rows = w_q.shape[0]
            permutation_q = torch.randperm(rows)
            permutation_k = torch.randperm(rows)
            permutation_v = torch.randperm(rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        if "attn.proj.weight" in name:
            w_proj = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_proj.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_proj.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        if "mlp.fc1.weight" in name:
            w_fc1 = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_fc1.shape[0],), device=device)
            mask = torch.matmul(w_fc1.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc1.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc1[i] *= ratio2
                mask_i = mask * ratio 
                w_fc1[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc1.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        if "mlp.fc2.weight" in name:
            w_fc2 = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_fc2.shape[0],), device=device)
            mask = torch.matmul(w_fc2.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc2.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc2[i] *= ratio2
                mask_i = mask * ratio 
                w_fc2[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc2.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def attack_arrowcloak(model, pre_model):
    set_seed()
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wq[i] /= ratio_q
                restore_wk[i] /= ratio_k
                restore_wv[i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        if "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[0]):
                ratio_o = sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item())
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        if "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item())
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1
        if "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = row_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[0]):
                ratio_fc2 = sqrt(torch.var(restore_fc2[i]).item()/torch.var(pre_fc2[i]).item())
                restore_fc2[i] /= ratio_fc2
            module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model

def attack_arrowcloak2(model, pre_model, orig_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name or "attn.proj.weight" in name or "mlp.fc1.weight" in name or "mlp.fc2.weight" in name:
            print(f"正在恢复{name}...")
            ob_w = module.data
            pre_w = pre_model.state_dict()[name].data
            orig_w = orig_model.state_dict()[name].data
            
            A_obf = ob_w.cpu().numpy()
            A_pub = pre_w.cpu().numpy()
            A_real = orig_w.cpu().numpy()
            path = "./data/weight_F100"
            np.save(f"{path}/A_obf_{name.replace('.','_')}.npy", A_obf)
            np.save(f"{path}/A_pub_{name.replace('.','_')}.npy", A_pub)
            np.save(f"{path}/A_real_{name.replace('.','_')}.npy", A_real)
            
                
            if os.path.exists(f"{path}/A_rec_{name.replace('.','_')}.npy"):
                print("恢复成功!")
                restore_w = np.load(f"{path}/A_rec_{name.replace('.','_')}.npy")
            elif True:
                print("使用公共模型权重作为恢复结果")
                restore_w = A_pub
            else:
                A_real = A_real.astype(np.float64)
                A_pub = A_pub.astype(np.float64)
                A_obf = A_obf.astype(np.float64)
                if "qkv" in name:
                    ob_wq, ob_wk, ob_wv = np.array_split(A_obf, 3, axis=0)
                    pre_wq, pre_wk, pre_wv = np.array_split(A_pub, 3, axis=0)
                    ratio = 1.0 * ob_wq.shape[0] / ob_wq.shape[1]
                    rho = ratio**2
                    max_iter = int(1000 / ratio)
                    L, S = solve_admm_structured(ob_wq, pre_wq, rho=rho, max_iter=max_iter, alpha=1.6)
                    restore_wq = (L + S) @ ob_wq
                    if np.linalg.norm(restore_wq - pre_wq, 'fro') > 3:
                        print("使用公共模型权重作为恢复结果")
                        restore_wq = pre_wq
                    L, S = solve_admm_structured(ob_wk, pre_wk, rho=rho, max_iter=max_iter, alpha=1.6)
                    restore_wk = (L + S) @ ob_wk
                    if np.linalg.norm(restore_wk - pre_wk, 'fro') > 3:
                        print("使用公共模型权重作为恢复结果")
                        restore_wk = pre_wk
                    L, S = solve_admm_structured(ob_wv, pre_wv, rho=rho, max_iter=max_iter, alpha=1.6)
                    restore_wv = (L + S) @ ob_wv
                    if np.linalg.norm(restore_wv - pre_wv, 'fro') > 3:
                        print("使用公共模型权重作为恢复结果")
                        restore_wv = pre_wv
                    restore_w = np.concatenate([restore_wq, restore_wk, restore_wv], axis=0)
                else:
                    
                    ratio = 1.0 * A_obf.shape[0] / A_obf.shape[1]
                    rho = ratio**2
                    max_iter = int(1000 / ratio)
                    L_precise, S_precise = solve_admm_structured(A_obf, A_pub, rho=rho, max_iter=max_iter, alpha=1.6)
                    restore_w = (L_precise + S_precise) @ A_obf
                    
                if np.linalg.norm(restore_w - A_real, 'fro') < np.linalg.norm(A_pub - A_real, 'fro') :
                    print("恢复成功!")
                    np.save(f"{path}/A_rec_{name.replace('.','_')}.npy", restore_w)
                else:
                    print("恢复失败!")
                
                if np.linalg.norm(restore_w - A_pub, 'fro') > 2:
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
    return model

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

def attack_finetune(model, trainloader, evalloader, num_classes, save_path, device, size=224, epochs=3, lr=5e-6):
    set_seed()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()    
    
    model = model.to(device)

    training_losses = []
    eval_losses = []
    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        epoch_progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training")
        tot_train_loss = 0  
        for batch_idx, (images, labels) in epoch_progress_bar:
            images, labels = images.to(device), labels.to(device)
            if images.size(-1) != size:
                images = torch.nn.functional.interpolate(images, size=(size, size))
            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            tot_train_loss += loss.item()
            epoch_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = tot_train_loss / len(trainloader)
        training_losses.append(avg_train_loss) 
           
        scheduler.step()  
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], Current LR: {current_lr:.6e}")
        
        ## 计算验证损失
        model.eval()
        tot_eval_loss = 0
        with torch.no_grad():
            for images, labels in evalloader:
                images, labels = images.to(device), labels.to(device)
                if images.size(-1) != size:
                    images = torch.nn.functional.interpolate(images, size=(size, size))
                outputs = model(images)
                loss = criterion(outputs, labels)
                tot_eval_loss += loss.item()

        avg_eval_loss = tot_eval_loss / len(evalloader)
        eval_losses.append(avg_eval_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Eval Loss: {avg_eval_loss:.6f}")
    
    save_model_weights(model, save_path)
    print(f"Model weights saved to {save_path}") 
    return model

