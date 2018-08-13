#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python train.py \
  --batch_size 32 \
  --evaluate_every 5 \
  --n_epochs 100 \
  --lr_init 1e-2 \
  --lambda_gt 0.5 \
  --image_size 16 \
  --max_iter_steps 50 \
  --max_iter_steps_from_gt 20 \
  --n_evaluation_steps 200 \
  --activation clamp \
  --iterator unet \
  --zero_init 0 \
  --ckpt_name ''
