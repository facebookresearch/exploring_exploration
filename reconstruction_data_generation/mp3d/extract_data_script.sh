#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

habitat_root=$EXPLORING_EXPLORATION/environments/habitat/habitat-api

mkdir uniform_samples
for split in 'val' 'test' 'train'
do
  python generate_uniform_points.py \
    --config-path configs/pointnav_mp3d_${split}.yaml \
    --habitat-root $habitat_root \
    --save-dir uniform_samples/${split}
done
