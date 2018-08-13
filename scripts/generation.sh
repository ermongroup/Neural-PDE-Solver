#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python generation.py \
  --image_size 64 \
  --n_frames 200 \
  --save_every 1 \
  --n_runs 4000
