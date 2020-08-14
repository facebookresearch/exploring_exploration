#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


def proc_rgb(rgb):
    return cv2.resize(np.flip(rgb, axis=2), (300, 300))


def proc_depth(depth):
    depth = np.clip(depth / 1000.0, 0.0, 10.0)  # Meters
    depth = depth * 255.0 / 10.0  # Intensities
    depth = np.repeat(depth, 3, axis=-1)
    return cv2.resize(depth.astype(np.uint8), (300, 300))
