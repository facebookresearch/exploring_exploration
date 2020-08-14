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

WIDTH = 300
HEIGHT = 300

overall_image = np.zeros((HEIGHT * 2, WIDTH * 3, 3), dtype=np.uint8)

T_exp = 50
T_nav = 50

env = gym.make("avd-nav-random-oracle-v0")
env.seed(123)
env.set_split("val")
env.set_t_exp_and_nav(T_exp, T_nav)
env.set_return_topdown_map()


def process_inputs(rgb, depth, fine_occ, coarse_occ, topdown_map, target):
    obs_1 = np.concatenate([rgb, depth, topdown_map], axis=1)
    obs_2 = np.concatenate([fine_occ, coarse_occ, target], axis=1)
    return np.concatenate([obs_1, obs_2], axis=0)


for i in range(10):
    obs = env.reset()
    topdown = env.generate_topdown_occupancy()
    rgb_im = proc_rgb(obs["im"])
    fine_occ_im = proc_rgb(obs["fine_occupancy"])
    coarse_occ_im = proc_rgb(obs["coarse_occupancy"])
    topdown_im = proc_rgb(topdown)
    cv2.imshow(
        "PointNav: exploration phase",
        np.concatenate([rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1),
    )
    cv2.waitKey(150)

    done = False
    for t in range(T_exp + T_nav):
        if t < T_exp:
            action = obs["oracle_action"][0].item()
        else:
            action = obs["sp_action"][0].item()

        obs, reward, done, info = env.step(action)
        if done or action == 3:
            cv2.destroyWindow("PointNav: navigation phase")
            break

        topdown = env.generate_topdown_occupancy()
        rgb_im = proc_rgb(obs["im"])
        fine_occ_im = proc_rgb(obs["fine_occupancy"])
        coarse_occ_im = proc_rgb(obs["coarse_occupancy"])
        topdown_im = proc_rgb(topdown)
        if t < T_exp:
            cv2.imshow(
                "PointNav: exploration phase",
                np.concatenate(
                    [rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1
                ),
            )
        else:
            if t == T_exp:
                cv2.destroyWindow("PointNav: exploration phase")
            cv2.imshow(
                "PointNav: navigation phase",
                np.concatenate(
                    [rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1
                ),
            )

        cv2.waitKey(150)
