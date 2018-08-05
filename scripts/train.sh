#!/bin/bash

python train.py \
  --batch_size 8 \
  --evaluate_every 5 \
  --n_epochs 10 \
  --ckpt_name test
