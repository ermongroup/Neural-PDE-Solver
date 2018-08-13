#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python test.py \
  --batch_size 4 \
  --log_every 5 \
  --image_size 64 \
  --data_limit 20 \
  --n_evaluation_steps 200 \
  --switch_to_fd 50 \
  --zero_init 1 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/16x16/unet_zero_iter20_20_gt0.5_sgd1e-03
