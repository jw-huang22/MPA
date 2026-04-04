
import os
import random
import numpy as np
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
import evaluate
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import pickle
from utils.utils import *
from pdb import set_trace as st
from utils.methods import *


parser = argparse.ArgumentParser(description="loading")

parser.add_argument("--dataset", default="cola", type=str, help="dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="Model you want to fine-tune")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for training")
parser.add_argument("--bs", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=3, type=int, help="epochs for finetune")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for training")
parser.add_argument("--gpus", type=str, default="2,3", help="gpu ids")
parser.add_argument("--recover_lr", default=1e-5, type=float, help="Learning rate for recovering")
parser.add_argument("--recover_epochs", default=3, type=int, help="epochs for recovering")

parser.add_argument("--obfus", default="translinkguard", type=str, help="obfuscation method")
parser.add_argument("--output_dir", default="tmp/output_results", type=str, help="output directory")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--weight_dir_tsqp", default="results/tsqp_results", type=str, help="weight directory")
parser.add_argument("--restore_dir", default="results/arrowmatch_results", type=str, help="restore directory")
parser.add_argument("--obfus_dir", default="tmp/obfus_results", type=str, help="obfus directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if args.model == "bert-base-cased":
    model_name = "bert"
else:
    print("The code about ViT and GPT2 will be published before AE phase.")
    print("Please try bert-base-cased for now.")
    raise ValueError("Invalid model name")

args.weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}/final_checkpoint"
args.weight_dir_tsqp = f"{args.weight_dir_tsqp}/{model_name}/{args.dataset}/final_checkpoint"
args.restore_dir = f"{args.restore_dir}/{model_name}/{args.obfus}/{args.dataset}"
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"

os.makedirs(args.restore_dir, exist_ok=True)
os.makedirs(args.recover_data_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.obfus_dir, exist_ok=True)
set_seed()

# number of classes in the dataset
actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
num_labels = 3 if actual_task.startswith("mnli") else (1 if actual_task == "stsb" else 2)
validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"

# Prepare data
print("Preparing data..")
task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
sentence1_key, sentence2_key = task_to_keys[args.dataset]
trainset, evalset, tokenizer = prepare_data(actual_task, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)

# Model & metric
print("Building model..")
set_seed()

model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )

print("Loading metric..")
metric = evaluate.load('glue', actual_task)

path = f"{args.recover_data_dir}/recover_data.json"
if not os.path.exists(f"{args.recover_data_dir}/recover_data.json"):
    prepare_recover_data(model, trainset, args.bs, path, ratio = 0.01)
recover_dataset = load_dataset("json", data_files=path)["train"]
print("recover_data prepared!")

if args.obfus == "tsqp":
    if os.path.exists(args.weight_dir_tsqp):
        model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir_tsqp,  num_labels=num_labels,use_safetensors=True )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels,use_safetensors=True )
else:
    model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )
    vic_model = AutoModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=num_labels, use_safetensors=True )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if os.path.exists(f"{args.restore_dir}/final_checkpoint") and False:
    print("Loading final model..")
    set_seed()
    final_model = AutoModelForSequenceClassification.from_pretrained(f"{args.restore_dir}/final_checkpoint", num_labels=num_labels, use_safetensors=True)
    final_model.config.pad_token_id = tokenizer.pad_token_id
    restore_args = TrainingArguments(
        output_dir=f"{args.restore_dir}",
        eval_strategy='no', 
        save_strategy="no", 
        per_device_eval_batch_size=args.bs,
        weight_decay=args.weight_decay,
        dataloader_num_workers=4,
        do_train=False,
    )
    trainer = Trainer(
        model=final_model,
        args=restore_args,
        train_dataset=recover_dataset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    final_results = trainer.evaluate(eval_dataset=evalset)
    print(f"最终恢复后的结果:{final_results}")
else:
    if args.obfus == "black":
        set_seed()
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        restore_model = init_model
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "translinkguard":  
        set_seed() 
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, permutations,rows = ob_translinkguard(model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',
        #     save_strategy="no", 
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4, 
        #     do_train=False,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model = attack_translinkguard(obfus_model,init_model, rows) 
        # restore_model = attack_translinkguard2(obfus_model, init_model, rows) 
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")

    elif args.obfus == "tsqp":
        set_seed()
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, scaling_factors = ob_tsqp(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no', 
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model, restore_scaling_factors = attack_tsqp(obfus_model,init_model)
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")

    elif args.obfus == "soter":
        set_seed()
        init_model =  AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, scaling_factors, _ = ob_soter(model,init_model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',
        #     save_strategy="no", 
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4, 
        #     do_train=False,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model, restore_scaling_factors = attack_soter(obfus_model,init_model)
        # restore_model, restore_scaling_factors = attack_soter2(obfus_model,init_model)
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',  
            logging_strategy='epoch',
            save_strategy="epoch",  
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
           
    elif args.obfus == "tempo":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, permutations, scaling_factors = ob_tempo(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no', 
            save_strategy="no", 
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            do_train=False,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model, restore_permutations = attack_tempo(obfus_model,init_model)
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch',
            logging_strategy='epoch',
            save_strategy="epoch", 
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4, 
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")

    elif args.obfus == "shadownet":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, permutations, scaling_factors = ob_shadownet(model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',  
        #     save_strategy="no",  
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4,  
        #     do_train=False,
        #     seed = 42,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model, restore_permutations = attack_shadownet(obfus_model,init_model)
        # restore_model, restore_permutations = attack_shadownet2(obfus_model,init_model)
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "LoRO":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        rank_r = 12
        obfus_model = ob_LoRO(model, r=rank_r)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',  
        #     save_strategy="no",  
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4,  
        #     do_train=False,
        #     seed = 42,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model = attack_LoRO(obfus_model,init_model,vic_model, r=rank_r)
        # restore_model, _ = attack_arrowcloak(obfus_model,init_model)
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "obfuscatune":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model = ob_obfuscatune(model)
        obfus_args = TrainingArguments(
            output_dir=f"{args.obfus_dir}",
            eval_strategy='no',  
            save_strategy="no",  
            per_device_eval_batch_size=args.bs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,  
            do_train=False,
            seed = 42,
        )
        trainer = Trainer(
            model=obfus_model,
            args=obfus_args,
            train_dataset=None,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        obfus_result = trainer.evaluate(eval_dataset=evalset)
        print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model = attack_obfuscatune(obfus_model,init_model,vic_model)
        # restore_model, _ = attack_arrowcloak(obfus_model,init_model)
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "groupcover":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model = ob_groupcover(model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',  
        #     save_strategy="no",  
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4,  
        #     do_train=False,
        #     seed = 42,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model = attack_groupcover(obfus_model,init_model,vic_model, dataset=args.dataset.upper())
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "twinshield":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model = ob_twinshield(model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',  
        #     save_strategy="no",  
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4,  
        #     do_train=False,
        #     seed = 42,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model = attack_twinshield(obfus_model,init_model,vic_model, dataset=args.dataset.upper())
        # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    elif args.obfus == "arrowcloak":
        set_seed()
        init_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        obfus_model, _, _, _, _ = ob_arrowcloak(model)
        # obfus_args = TrainingArguments(
        #     output_dir=f"{args.obfus_dir}",
        #     eval_strategy='no',  
        #     save_strategy="no",  
        #     per_device_eval_batch_size=args.bs,
        #     weight_decay=args.weight_decay,
        #     dataloader_num_workers=4,  
        #     do_train=False,
        #     seed = 42,
        # )
        # trainer = Trainer(
        #     model=obfus_model,
        #     args=obfus_args,
        #     train_dataset=None,
        #     eval_dataset=evalset,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics,
        # )
        # obfus_result = trainer.evaluate(eval_dataset=evalset)
        # print(f"混淆后的结果: {obfus_result}")
        set_seed()
        restore_model, restore_perm = attack_arrowcloak2(obfus_model,init_model,vic_model)
         # 加入微调
        restore_args = TrainingArguments(
            output_dir=f"{args.restore_dir}",
            eval_strategy='epoch', 
            logging_strategy='epoch',
            save_strategy="epoch",
            learning_rate=args.recover_lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.recover_epochs,
            weight_decay=args.weight_decay,
            dataloader_num_workers=4,
            seed=42,
        )
        trainer = Trainer(
            model=restore_model,
            args=restore_args,
            train_dataset=recover_dataset,
            eval_dataset=evalset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        set_seed()
        trainer.train()
        restore_results = trainer.evaluate(eval_dataset=evalset)
        print(f"最终恢复后的结果:{restore_results}")
    else:
        raise ValueError("Invalid obfuscation method")
