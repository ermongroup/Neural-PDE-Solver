#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python test.py \
  --batch_size 16 \
  --log_every 10 \
  --data_limit 50 \
  --n_evaluation_steps 800 \
  --initialization random \
  --which_epochs -1 \
  --image_size 65 \
  --geometry cylinders \
  --load_ckpt_path $HOME/slowbro/ckpt/heat/17x17/conv3_random_iter20_0_gt0.0_adam1e-03
