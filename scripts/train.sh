#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --batch_size 32 \
  --evaluate_every 5 \
  --n_epochs 200 \
  --lr_init 1e-3 \
  --lambda_gt 0.5 \
  --image_size 16 \
  --max_iter_steps 30 \
  --max_iter_steps_from_gt 1 \
  --n_evaluation_steps 100 \
  --activation clamp \
  --ckpt_name clamp
