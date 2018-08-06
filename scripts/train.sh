#!/bin/bash
python train.py \
  --gpus 0 \
  --batch_size 16 \
  --evaluate_every 5 \
  --n_epochs 400 \
  --lr_init 1e-4 \
  --lambda_gt 0.5 \
  --image_size 64 \
  --max_iter_steps 50 \
  --n_evaluation_steps 200 \
  --ckpt_name ckpt
