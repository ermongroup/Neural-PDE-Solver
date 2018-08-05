#!/bin/bash
python train.py \
  --batch_size 16 \
  --evaluate_every 5 \
  --n_epochs 200 \
  --lr_init 1e-4 \
  --lambda_gt 0.5 \
  --ckpt_name test_gt0.5
