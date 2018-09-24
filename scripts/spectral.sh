#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python spectral.py \
  --which_epochs -1 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/17x17/conv1_random_iter20_0_gt0_adam1e-03
