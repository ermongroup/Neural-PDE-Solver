#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --batch_size 16 \
  --log_every 10 \
  --image_size 63 \
  --data_limit 100 \
  --n_evaluation_steps 800 \
  --initialization random \
  --which_epochs -1 \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/15x15/conv1_random_iter20_0_gt0.0_adam1e-03
