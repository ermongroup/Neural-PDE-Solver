#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --batch_size 32 \
  --evaluate_every 1 \
  --save_every 1 \
  --n_epochs 20 \
  --lr_init 1e-3 \
  --max_iter_steps 20 \
  --max_iter_steps_from_gt 0 \
  --n_evaluation_steps 400 \
  --image_size 63 \
  --iterator multigrid \
  --conv_n_layers 1 \
  --mg_n_layers 4 \
  --mg_pre_smoothing 2 \
  --mg_post_smoothing 2 \
  --initialization random \
  --ckpt_name test
