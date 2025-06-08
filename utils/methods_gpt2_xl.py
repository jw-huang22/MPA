from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils_gpt2_xl import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt


def ob_translinkguard(model):
    set_seed()
    layer_permutations = {}
    rows = 0
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            layer_name = name.rsplit(".")[2]
            # 打乱行顺序
            # 注意data的行列是正的（因为用的是CONV1D）
            num_rows = w_q.shape[0]
            rows = num_rows
            permutation = torch.randperm(num_rows)
            layer_permutations[layer_name] = permutation    
            ob_w_q = w_q[permutation]
            ob_w_k = w_k[permutation]
            ob_w_v = w_v[permutation]
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=1)
            module.data = ob_w
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            layer_name = name.rsplit(".")[2]
            permutation = layer_permutations[layer_name]
            inv_perm = torch.argsort(permutation)
            ob_proj = w_proj[:, inv_perm]
            module.data = ob_proj   
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            layer_name = name.rsplit(".")[2]
            permutation = layer_permutations[layer_name]
            ob_fc1 = w_fc1[permutation]
            module.data = ob_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            if(w_fc2.shape[0] == rows):
                layer_name = name.rsplit(".")[2]
                permutation = layer_permutations[layer_name]
                ob_fc2 = w_fc2[permutation]
                module.data = ob_fc2
    return model, layer_permutations, rows

def attack_translinkguard(model, pre_model, rows):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=1)
            layer_name= name.rsplit(".")[2]
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].chunk(3, dim=1)
            perm, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            restore_perm[layer_name] = torch.tensor(perm)
            inv_perm = torch.argsort(torch.tensor(perm))
            restore_wk = ob_wk[inv_perm]
            restore_wv = ob_wv[inv_perm]
            restore_w = torch.cat([restore_wq, restore_wk, restore_wv], dim=1)
            module.data = restore_w
        elif "attn.c_proj.weight" in name:
            ob_wo = module.data
            layer_name= name.rsplit(".")[2]
            perm = restore_perm[layer_name]
            restore_wo = ob_wo[:,perm]
            module.data = restore_wo
        elif "mlp.c_fc.weight" in name:
            ob_fc1 = module.data
            layer_name= name.rsplit(".")[2]
            perm = restore_perm[layer_name]
            inv_perm = torch.argsort(perm)
            restore_fc1 = ob_fc1[inv_perm]
            module.data = restore_fc1
        elif "mlp.c_proj.weight" in name:
            ob_fc2 = module.data
            if(ob_fc2.shape[0] == rows):
                layer_name = name.rsplit(".")[2]
                perm = restore_perm[layer_name]
                inv_perm = torch.argsort(perm)
                restore_fc2 = ob_fc2[inv_perm]
                module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model    

def ob_tsqp(model):
    set_seed() 
    scaling_factors = {} 
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            scale_q = 1 + 5 * torch.rand(1).item()
            scale_k = 1 + 5 * torch.rand(1).item()
            scale_v = 1 + 5 * torch.rand(1).item()
            w_q *= scale_q
            w_k *= scale_k
            w_v *= scale_v
            module.data = torch.cat([w_q, w_k, w_v], dim=1)
            scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            scale_proj = 1 + 5 * torch.rand(1).item()
            w_proj *= scale_proj
            module.data = w_proj
            scaling_factors[name] = scale_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            scale_fc1 = 1 + 5 * torch.rand(1).item()
            w_fc1 *= scale_fc1
            module.data = w_fc1
            scaling_factors[name] = scale_fc1
        elif "mlp.c_proj.weight" in name:
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
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=1)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}         
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=1)
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.c_proj.weight" in name:
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
        if "attn.c_attn.weight" in name or "attn.c_proj.weight" in name or "mlp.c_fc.weight" in name or "mlp.c_proj.weight" in name:
            init_layers[name] = module 
    num_layers_to_replace = int(len(init_layers) * 0.2)
    layers_to_replace = random.sample(list(init_layers.keys()), num_layers_to_replace)
    ## 替换选中层的权重
    for name, module in model.named_parameters():
        if name in layers_to_replace:
            module.data = pre_model.state_dict()[name].data
    for name, module in model.named_parameters():
        if name not in layers_to_replace:
            if "attn.c_attn.weight" in name:
                w_q, w_k, w_v = module.data.chunk(3, dim=1)
                scale_q = 1 + 5 * torch.rand(1).item()
                scale_k = 1 + 5 * torch.rand(1).item()
                scale_v = 1 + 5 * torch.rand(1).item()
                w_q *= scale_q
                w_k *= scale_k
                w_v *= scale_v
                module.data = torch.cat([w_q, w_k, w_v], dim=1)
                scaling_factors[name] = {"q": scale_q, "k": scale_k, "v": scale_v}
            elif "attn.c_proj.weight" in name:
                w_proj = module.data
                scale_proj = 1 + 5 * torch.rand(1).item()
                w_proj *= scale_proj
                module.data = w_proj
                scaling_factors[name] = scale_proj
            elif "mlp.c_fc.weight" in name:
                w_fc1 = module.data
                scale_fc1 = 1 + 5 * torch.rand(1).item()
                w_fc1 *= scale_fc1
                module.data = w_fc1
                scaling_factors[name] = scale_fc1
            elif "mlp.c_proj.weight" in name:
                w_fc2 = module.data
                scale_fc2 = 1 + 5 * torch.rand(1).item()
                w_fc2 *= scale_fc2
                module.data = w_fc2
                scaling_factors[name] = scale_fc2
    return model, scaling_factors, layers_to_replace

def attack_soter(model, pre_model):
    set_seed()
    restore_scaling_factors = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            pre_q, pre_k, pre_v = pre_model.state_dict()[name].data.chunk(3, dim=1)
            k1 = fix_factor(sqrt(torch.var(w_q).item()/torch.var(pre_q).item()))
            k2 = fix_factor(sqrt(torch.var(w_k).item()/torch.var(pre_k).item()))
            k3 = fix_factor(sqrt(torch.var(w_v).item()/torch.var(pre_v).item()))
            new_w_q = w_q/k1
            new_w_k = w_k/k2
            new_w_v = w_v/k3
            restore_scaling_factors[name] = {"q": k1, "k": k2, "v":k3}
            module.data = torch.cat([new_w_q, new_w_k, new_w_v], dim=1)
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_proj).item()/torch.var(pre_proj).item()))
            new_w_proj = w_proj/k
            restore_scaling_factors[name] = k
            module.data = new_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            k = fix_factor(sqrt(torch.var(w_fc1).item()/torch.var(pre_fc1).item()))
            new_w_fc1 = w_fc1/k  
            restore_scaling_factors[name] = k
            module.data = new_w_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            k =  fix_factor(sqrt(torch.var(w_fc2).item()/torch.var(pre_fc2).item()))
            new_w_fc2 = w_fc2/k
            restore_scaling_factors[name] =k
            module.data = new_w_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_scaling_factors

def ob_shadownet(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            num_rows = w_q.shape[0]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_rows):
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
            permutation_q = torch.randperm(num_rows)
            permutation_k = torch.randperm(num_rows)
            permutation_v = torch.randperm(num_rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=1)
            module.data = ob_w
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[0]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            num_rows = w_fc1.shape[0]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            num_rows = w_fc2.shape[0]
            ratios = []
            for i in range(num_rows):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, scaling_factors
    
def attack_shadownet(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=1)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=1)
            ## restore_wq是恢复permutation之后的wq
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
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=1)
        elif "attn.c_proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[0]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item()))
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        elif "mlp.c_fc.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item()))
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1            
        elif "mlp.c_proj.weight" in name:
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

def ob_tempo(model):
    set_seed()
    layer_permutations = {}
    scaling_factors = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            num_cols = w_q.shape[1]
            ratios_q = []
            ratios_k = []
            ratios_v = []
            for i in range(num_cols):
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
            permutation_q = torch.randperm(num_cols)
            permutation_k = torch.randperm(num_cols)
            permutation_v = torch.randperm(num_cols)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[:,permutation_q]
            ob_w_k = w_k[:,permutation_k]
            ob_w_v = w_v[:,permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=1)
            module.data = ob_w
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[1]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_proj[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[:,permutation]
            module.data = ob_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            num_cols = w_fc1.shape[1]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc1[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[:,permutation]
            module.data = ob_w_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            num_cols = w_fc2.shape[1]
            ratios = []
            for i in range(num_cols):
                ratio = 1 + 5 * torch.rand(1).item()
                w_fc2[:,i] *= ratio
                ratios.append(ratio)
            scaling_factors[name] = ratios
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[:,permutation]
            module.data = ob_w_fc2   
    return model, layer_permutations, scaling_factors

def attack_tempo(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=1)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=1)
            ## restore_wq是恢复Permutation后的wq
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
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=1)
        elif "attn.c_proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = col_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[1]):
                ratio_o = fix_factor(sqrt(torch.var(restore_wo[:,i]).item()/torch.var(pre_wo[:,i]).item()))
                restore_wo[:,i] /= ratio_o
            module.data = restore_wo
        elif "mlp.c_fc.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = col_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[1]):
                ratio_fc1 = fix_factor(sqrt(torch.var(restore_fc1[:,i]).item()/torch.var(pre_fc1[:,i]).item()))
                restore_fc1[:,i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.c_proj.weight" in name:
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

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            num_cols = w_q.shape[1]
            device = w_q.device
            coeff_q, coeff_k, coeff_v = torch.randint(0,5,(w_q.shape[1],), device=device), torch.randint(0,5,(w_k.shape[1],), device=device), torch.randint(0,5,(w_v.shape[1],), device=device)
            mask_q, mask_k, mask_v = torch.matmul(w_q, coeff_q.float()), torch.matmul(w_k, coeff_k.float()), torch.matmul(w_v, coeff_v.float())
            layer_masks[name] = {"q": mask_q, "k": mask_k, "v": mask_v}
            ratios_q, ratios_k, ratios_v = [], [], []
            ratios_q2, ratios_k2, ratios_v2 = [], [], []
            for i in range(num_cols):
                ratio_q, ratio_k, ratio_v = (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float()
                weight_q, weight_k, weight_v = torch.randint(1, 3, (1,), device=device).float(), torch.randint(1, 3, (1,), device=device).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_qi = mask_q*ratio_q
                mask_ki = mask_k*ratio_k
                mask_vi = mask_v*ratio_v
                w_q[:,i] *= weight_q
                w_k[:,i] *= weight_k
                w_v[:,i] *= weight_v
                w_q[:,i] += mask_qi
                w_k[:,i] += mask_ki
                w_v[:,i] += mask_vi
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
                ratios_q2.append(weight_q)
                ratios_k2.append(weight_k)
                ratios_v2.append(weight_v)
            layer_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            weight_factors[name] = {"q": ratios_q2, "k": ratios_k2, "v": ratios_v2}
            permutation_q = torch.randperm(num_cols)
            permutation_k = torch.randperm(num_cols)
            permutation_v = torch.randperm(num_cols)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[:,permutation_q]
            ob_w_k = w_k[:,permutation_k]
            ob_w_v = w_v[:,permutation_v]
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=1)
            module.data = ob_w
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[1]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[1],), device=device)
            mask = torch.matmul(w_proj, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_proj[:,i] *= ratio2
                w_proj[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[:,permutation]
            module.data = ob_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            num_cols = w_fc1.shape[1]
            device = w_fc1.device
            coeff = torch.randint(0,5,(w_fc1.shape[1],), device=device)
            mask = torch.matmul(w_fc1, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_fc1[:,i] *= ratio2
                w_fc1[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[:,permutation]
            module.data = ob_w_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            num_cols = w_fc2.shape[1]
            device = w_fc2.device
            coeff = torch.randint(0,5,(w_fc2.shape[1],), device=device)
            mask = torch.matmul(w_fc2, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_fc2[:,i] *= ratio2
                w_fc2[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[:,permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def attack_arrowcloak(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=1)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=1)
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[1]):
                ratio_q = sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item())
                ratio_k = sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item())
                ratio_v = sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item())
                restore_wq[:,i] /= ratio_q
                restore_wk[:,i] /= ratio_k
                restore_wv[:,i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=1)
        elif "attn.c_proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = col_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[1]):
                ratio_o = sqrt(torch.var(restore_wo[:,i]).item()/torch.var(pre_wo[:,i]).item())
                restore_wo[:,i] /= ratio_o
            module.data = restore_wo
        elif "mlp.c_fc.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = col_restore_perm(pre_fc1, ob_fc1)
            for i in range(ob_fc1.shape[1]):
                ratio_fc1 = sqrt(torch.var(restore_fc1[:,i]).item()/torch.var(pre_fc1[:,i]).item())
                restore_fc1[:,i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.c_proj.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = col_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item())
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm
