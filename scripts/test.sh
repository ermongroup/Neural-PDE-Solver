#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --batch_size 4 \
  --image_size 16 \
  --ckpt_name first_iter30_gt0.5_lr1e-03_epoch200
