#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python generation.py \
  --image_size 513 \
  --save_every 1 \
  --geometry square \
  --poisson 0 \
  --n_runs 200
