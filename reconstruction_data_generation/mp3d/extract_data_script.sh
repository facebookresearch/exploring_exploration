#!/bin/bash

habitat_root=$EXPLORING_EXPLORATION/env/habitat/habitat-api

mkdir uniform_samples
for split in 'val' 'test' 'train'
do
  python generate_uniform_points.py \
    --config-path configs/pointnav_mp3d_${split}.yaml \
    --habitat-root $habitat_root \
    --save-dir uniform_samples/${split}
done
