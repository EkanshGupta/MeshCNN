#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/shrec_16 \
--name shrec16 \
--ncf 16 32 64 128 \
--pool_res 600 450 300 150 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \