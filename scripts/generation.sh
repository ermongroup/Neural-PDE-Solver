#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python generation.py \
  --image_size 257 \
  --save_every 1 \
  --geometry square \
  --poisson 1 \
  --n_runs 200
