#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python test.py \
  --batch_size 4 \
  --log_every 5 \
  --image_size 64 \
  --n_evaluation_steps 200 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/16x16/clamp_iter30_1_gt0.5_lr1e-03_epoch200
