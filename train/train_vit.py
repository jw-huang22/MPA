import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import argparse
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdb import set_trace as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import timm
import matplotlib.pyplot as plt
from utils.utils_vit import *

# parsers
parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate"
)  
parser.add_argument("--opt", default="sgd")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--noamp",action="store_true")
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--dataset", default="cifar_10", type=str, help="dataset")
parser.add_argument("--output_dir", default="results")
parser.add_argument("--adapter", action="store_true", help="use adapter for teacher")
parser.add_argument("--tsqp", default="false", type=str)

args = parser.parse_args()
args.output_dir = f"{args.output_dir}/{args.dataset}"
os.makedirs(args.output_dir, exist_ok=True)

bs = int(args.bs)
use_amp = not args.noamp
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  
start_epoch = 0  

set_seed()
# Data
print("==> Preparing data..")
size = 224

## change the dir if you want
trainloader, testloader, num_classes = prepare_data("./data/datasets", args, size)

print("==> Building model..")
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
pre_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
model.cuda()

net = model

if args.resume:
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("{}/ckpt.t7".format(args.output_dir))
    net.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]


criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def train(epoch, tsqp=False):
    global act_alpha_rate
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if tsqp:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss_1 = 1e-3 * loss1(net)
                loss_2 = 1e-3 * loss2(net, pre_model)
                loss = loss+loss_1-loss_2
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"% (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),)
    return train_loss / (batch_idx + 1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (test_loss / (batch_idx + 1),100.0 * correct / total,correct,total,),)
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "acc": acc,}
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, f"{args.output_dir}" + "/ckpt.t7".format(args.dataset))
        best_acc = acc
    test_loss /= (batch_idx + 1)
    content = (time.ctime() + " " + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}')
    print(content)
    with open(f"{args.output_dir}/teacher_log_{args.dataset}.txt", "a") as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
train_losses = []
val_losses = []
net.cuda()

for epoch in range(start_epoch, args.n_epochs):
    set_seed()
    start = time.time()
    if args.tsqp == "true":
        train_loss = train(epoch, tsqp=True)
    else:
        train_loss = train(epoch)
    val_loss, acc = test(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(epoch)  
    list_loss.append(val_loss)
    list_acc.append(acc)
    with open(f"{args.output_dir}/teacher_log_{args.dataset}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, val_loss, acc])
    print(f"Epoch {epoch}: Validation Loss: {val_loss}, Accuracy: {acc}")