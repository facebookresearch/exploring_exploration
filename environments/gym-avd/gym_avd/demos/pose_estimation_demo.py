#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import gym
import gym_avd
import numpy as np
from utils import *


def create_reference_grid(refs_uint8):
    """
    Inputs:
        refs_uint8 - (nRef, H, W, C) numpy array
    """
    refs_uint8 = np.copy(refs_uint8)
    nRef, H, W, C = refs_uint8.shape

    nrow = int(math.sqrt(nRef))

    ncol = nRef // nrow  # (number of images per column)
    if nrow * ncol < nRef:
        ncol += 1
    final_grid = np.zeros((nrow * ncol, *refs_uint8.shape[1:]), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    final_grid[:nRef] = refs_uint8
    final_grid = final_grid.reshape(
        ncol, nrow, *final_grid.shape[1:]
    )  # (ncol, nrow, H, W, C)
    final_grid = final_grid.transpose(0, 2, 1, 3, 4)
    final_grid = final_grid.reshape(ncol * H, nrow * W, C)
    return final_grid


WIDTH = 300
HEIGHT = 300

overall_image = np.zeros((HEIGHT * 2, WIDTH * 3, 3), dtype=np.uint8)

env = gym.make("avd-pose-landmarks-oracle-v0")
env.set_split("test")
env.seed(123 + 12)
env.plot_references_in_topdown = True
nref = 10
env.set_nref(nref)

obs = env.reset()
topdown = env.generate_topdown_occupancy()
rgb_im = proc_rgb(obs["im"])
topdown_im = proc_rgb(topdown)
ref_rgb = [proc_rgb(obs["pose_refs"][n]) for n in range(nref)]
ref_rgb = cv2.resize(create_reference_grid(np.stack(ref_rgb, axis=0)), (HEIGHT, WIDTH))

overall_image = np.concatenate([rgb_im, topdown_im, ref_rgb], axis=1)

cv2.imshow("Pose estimation demo", overall_image)
cv2.waitKey(60)

for i in range(10000):
    action = obs["oracle_action"][0]

    obs, _, done, info = env.step(action)

    if done:
        obs = env.reset()
        ref_rgb = [proc_rgb(obs["pose_refs"][n]) for n in range(nref)]
        ref_rgb = cv2.resize(
            create_reference_grid(np.stack(ref_rgb, axis=0)), (HEIGHT, WIDTH)
        )

    topdown = env.generate_topdown_occupancy()
    rgb_im = proc_rgb(obs["im"])
    topdown_im = proc_rgb(topdown)
    overall_image = np.concatenate([rgb_im, topdown_im, ref_rgb], axis=1)

    area = info["seen_area"]
    nlandmarks = info["oracle_pose_success"]
    nobjects = info["num_objects_visited"]

    print(f"Area: {area:5.2f} | OSR: {nlandmarks:5.2f} | Objects: {nobjects:5.2f}")
    cv2.imshow("Pose estimation demo", overall_image)
    cv2.waitKey(60)
