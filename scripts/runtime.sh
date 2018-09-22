#!/bin/bash
CUDA_VISIBLE_DEVICES='' \
python runtime.py \
  --batch_size 1 \
  --n_evaluation_steps 800 \
  --initialization avg \
  --which_epochs -1 \
  --image_size 1025 \
  --geometry square \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/65x65/unet3_random_iter20_0_gt0_adam1e-03
