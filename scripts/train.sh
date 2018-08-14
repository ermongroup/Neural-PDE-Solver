#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python train.py \
  --batch_size 32 \
  --evaluate_every 5 \
  --n_epochs 100 \
  --lr_init 1e-3 \
  --lambda_gt 0 \
  --image_size 16 \
  --max_iter_steps 20 \
  --max_iter_steps_from_gt 1 \
  --n_evaluation_steps 200 \
  --activation clamp \
  --iterator unet \
  --zero_init 0 \
  --optimizer adam \
  --ckpt_name resfd
