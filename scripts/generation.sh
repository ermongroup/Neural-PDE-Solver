#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python generation.py \
  --image_size 15 \
  --n_frames 1 \
  --save_every 1 \
  --n_runs 1000
