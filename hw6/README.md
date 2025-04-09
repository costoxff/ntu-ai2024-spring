# RLHF

## Python Environment

### GPU info
```
TU104 [GeForce RTX 2080 Rev. A] (8192MiB)
Driver Version: 535.161.07
CUDA Version: 12.2 
```

### 1. create environment

[The Fast Cross-Platform Package Manager](https://github.com/mamba-org/mamba)

just like conda or use conda, miniconda whatever you like

```
micromamba create -n ai_hw6 python=3.10
micromamba activate ai_hw6
```

### 2. install unsloth

[install unsloth](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions)

```
micromamba install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
```

### 3. install pytorch based on your cuda version (optinal)

PyTorch installation has been included in unsloth installation

### 4. some python pakage

```
pip install tqdm packaging wandb
```

```
pip install --no-deps trl peft accelerate bitsandbytes
```

### 5. final check

1. `nvcc --version`  
1. `python -m xformers.info`  
1. `python -m bitsandbytes`

## training

model selection : "unsloth/tinyllama-bnb-4bit"

### DPO

```
python main.py \
    --exp_name DPO \
    --model_name unsloth/tinyllama-bnb-4bit \
    --train \
    --wandb_token [your wandb token] \
    --train_batch_size 2 \
    --max_steps 1000 \
    --num_epochs 1 \
    --weight_decay 0.0001 \
    --beta 0.05
```

### ORPO

```
python main.py \
    --exp_name ORPO \
    --model_name unsloth/tinyllama-bnb-4bit \
    --train \
    --wandb_token [your wandb token] \
    --train_batch_size 2 \
    --max_steps 1000 \
    --num_epochs 1 \
    --weight_decay 0.0001 \
    --max_grad_norm 9 \
    --warmup_ratio 0.05 \
    --beta 0.05
```