# Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition Algorithms

## Overview
This repository contains the code for the paper "Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition Algorithms". 

### Remarks
The experiments were conducted on 2 × NVIDIA A6000 GPUs (48GB)

## Environment Setup
Note that different library version may affect test results. 

For Bert & GPT2 & GPT2_XL

```
export HF_ENDPOINT="https://hf-mirror.com"

./install1.sh

conda activate game-of-arrows1

# Basic Test for environment1
./scripts2/basic_test1.sh
```

For ViT:
```
export HF_ENDPOINT="https://hf-mirror.com"

./install2.sh

conda activate game-of-arrows2

# Basic Test for environment1
./scripts2/basic_test1.sh
```


## Code Structure
* **results:**  The results of attack and defense will be saved here after attacking/defending 
    * **train_results:** The results of training from public pretrained models.
    * **arrowmatch_results**: The results of ARROWMATCH across different datasets.
    * **arrowcloak_results**: The results of ARROWCLOAK across different datasets.
    * **tsqp_results**: The training results for TSQP.
* **evaluate_results:** The evaluation results of White-box, Black-box, ArrowMatch and ArrowCloak.
    * **bert**：generated after running the evaluation scripts.
        * **mnli**
          * **checkpoints**: The checkpoints of the Black-box
          * **blackbox_results**: The evaluation results of Black-box.
          * **whitebox_results**: The evaluation results of White-box.
          * **obfus_results**: The evaluation results of obfuscated model without ARROWMATCH.
          * **recover_results**: The evaluation results of Recovery of obfuscated model with ARROWMATCH(expected to be similar to whitebox_results).
          * **arrowcloak_results**: The evaluation results of Recovery of ArrowCloak with ARROWMATCH (expected to be similar to blackbox_results).
        * **Other dataset**: Similar to mnli
    * **ViT, gpt2 & gpt2_xl**: Similar to bert
* **data:** The dataset obtained by the adversary through querying the victim model accounts for less than1% of all the training data.
* **utils:** The functions in *ARROWMATCH* & *ARROWCLOAK*.
* **train:** The code for training the private models.
* **code:**  
    * evaluate_model.py evaluate_model_gpt.py evaluate_model_vit.py evaluate_model_gpt2_xl.py: The code for evaluations.
    * arrowmatch.py arrowmatch_gpt2.py arrowmatch_vit.py arrowmatch_gpt2_xl.py: The code for ARROWMATCH.
    * arrowcloak.py arrowcloak_gpt2.py arrowcloak_vit.py arrowcloak_gpt2_xl.py: The code for ARROWCLOAK.
* **scripts:** The scripts for running single experiment.
* **scripts2:** The scripts for running multiple experiments.




## Experiments

### Train private model

**Remember to rename the last checkpoint to 'final_checkpoint' in Bert, GPT2 and GPT2_XL after the training process for ARROWMATCH and ARROWCLOAK!**

```
export HF_ENDPOINT="https://hf-mirror.com"

# for ViT
# automactically train on cifar10, cifar100, food101
./scripts2/vit_trains.sh

# for Bert 
# automactically train on mnli, qqp, sst2, qnli
./scripts2/bert_trains.sh 

# for GPT2
# automactically train on mnli, qqp, sst2, qnli
./scripts2/gpt2_trains.sh

# for GPT2_XL
# automactically train on sst2
./scripts2/gpt2_xl_trains.sh
```


### Try ARROWMATCH
**Make sure the training results from public have been saved.** 

**Remember to rename the last checkpoint to 'final_checkpoint' in Bert, GPT2 and GPT2_XL after the training process for Evaluation!**


```
export HF_ENDPOINT="https://hf-mirror.com"

# for ViT
# automactically check the performances on cifar10, cifar100, food101 across different obfuscation methods
./scripts2/vit_arrowmatchs.sh

# for Bert
# automactically check the performances on mnli, qqp, sst2, qnli across different obfuscation methods
./scripts2/bert_arrowmatchs.sh

# for GPT2
# automactically check the performances on mnli, qqp, sst2, qnli across different obfuscation methods
./scripts2/gpt2_arrowmatchs.sh

# for GPT2_XL
# automactically check the performances on sst2 across different obfuscation methods
./scripts2/gpt2_xl_arrowmatchs.sh
```

### Try ARROWCLOAK
**Make sure the training results from public have been saved.** 

**Remember to rename the last checkpoint to 'final_checkpoint' in Bert, GPT2 and GPT2_XL after the training process for Evaluation!**

```
export HF_ENDPOINT="https://hf-mirror.com"

# for ViT
# automactically check the performances on cifar10, cifar100, food101
./scripts2/vit_arrowcloaks.sh

# for Bert
# automactically check the performances on mnli, qqp, sst2, qnli
./scripts2/bert_arrowcloaks.sh

# for GPT2
# automactically check the performances on mnli, qqp, sst2, qnli
./scripts2/gpt2_arrowcloaks.sh

# for GPT2_XL
# automactically check the performances on sst2
./scripts2/gpt2_xl_arrowcloaks.sh
```

### Evaluate the results
**Make sure the training results, ARROWMATCH results and ARROWCLOAK results have been saved.** 


```
export HF_ENDPOINT="https://hf-mirror.com"

# for ViT
# automactically evaluate the performances on cifar10, cifar100, food101
./scripts2/evaluate_models_vit.sh

# for Bert
# automactically evaluate the performances on mnli, qqp, sst2, qnli
./scripts2/evaluate_models_bert.sh

# for GPT2
# automactically evaluate the performances on mnli, qqp, sst2, qnli
./scripts2/evaluate_models_gpt2.sh

# for GPT2_XL
# automactically evaluate the performances on sst2
./scripts2/evaluate_models_gpt2_xl.sh
```

