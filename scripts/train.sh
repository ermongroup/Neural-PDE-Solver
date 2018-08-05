#!/bin/bash
python train.py \
  --batch_size 16 \
  --evaluate_every 5 \
  --n_epochs 20 \
  --lr_init 1e-4 \
  --ckpt_name test
