#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python test.py \
  --batch_size 4 \
  --image_size 64 \
  --n_evaluation_steps 100 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/16x16/first_iter30_gt0.5_lr1e-03_epoch200
