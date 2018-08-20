#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python train.py \
  --batch_size 32 \
  --evaluate_every 1 \
  --save_every 1 \
  --n_epochs 20 \
  --lr_init 1e-3 \
  --lambda_gt 0 \
  --image_size 16 \
  --max_iter_steps 20 \
  --max_iter_steps_from_gt 0 \
  --n_evaluation_steps 100 \
  --activation clamp \
  --iterator conv \
  --conv_n_layers 2 \
  --zero_init 0 \
  --ckpt_name ''
