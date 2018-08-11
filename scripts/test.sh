#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python test.py \
  --batch_size 4 \
  --log_every 5 \
  --image_size 64 \
  --data_limit 20 \
  --zero_init 1 \
  --n_evaluation_steps 200 \
  --switch_to_fd 50 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/16x16/clamp_zero_basic_iter50_50_gt0.5_lr1e-04
