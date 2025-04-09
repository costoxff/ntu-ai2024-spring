# Homework 5

## Install Necessary Packages

[mamba: The Fast Cross-Platform Package Manager](https://github.com/mamba-org/mamba)

```
micromamba create -n gym-pacman python=3.11 -y
micromamba activate gym-pacman
pip install -r requirements.txt
```

you could use conda instead, just subtitude the commad micromamba into conda or miniconda

## Training Configuration 

### parameter

```
python pacman.py \
    --lr 0.00015 \
    --batch_size 32 \
    --epsilon 1.0 \
    --epsilon_min 0.1 \
    --gamma 0.95 \
    --max_steps 200000 \
    --buffer_size 12000
```

#### bash

provide bash file, just run it

```
./run.sh
```

## Model Layer 

### Conv2dBlock (custom layer)

encapsulate Conv2d, BatchNorm2d, and LeakyRuLU

| Layer |
|---|
|Conv2d| 
|BatchNorm2d|
|LeakyReLU|

### model layer

| Layer | in dim | out dim | kernel size | stride | padding |
|---|:---:|:---:|:---:|:---:|:---:|
| Conv2dBlock 1 | 4 | 32 | (8, 8) | 2 | 2 |
| Conv2dBlock 2 | 32 | 64 | (4, 4) | 2 | 1 |
| Conv2dBlock 3 | 64 | 64 | (2, 2) | 2 | 0 |
| Flatten | start_dim=1|
| Linear 1 | 64 * 10 * 10  | 256 |
| LeakyReLU |
| Linear 2 | 256 | 64 |
| LeakyReLU |
| Linear 3 | 64 | 9 |