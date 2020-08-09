import cv2
import pdb
import math
import numpy as np
from utils import *

import habitat


class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0):
        super(DummyRLEnv, self).__init__(config, dataset)
        self._env_ind = env_ind

    def get_reward_range(self):
        return -1.0, 1.0

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_env_ind(self):
        return self._env_ind


config = habitat.get_config_pose("demos/configs/exploration_mp3d.yaml")

env = DummyRLEnv(config=config)
env.seed(1234)

obs = env.reset()

"""
Action space:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3
"""

steps = 0
while True:
    action = obs["oracle_action_sensor"][0].item()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        steps = 0

    rgb_im = proc_rgb(obs["rgb"])
    fine_occ_im = proc_rgb(obs["fine_occupancy"])
    coarse_occ_im = proc_rgb(obs["coarse_occupancy"])
    topdown_im = proc_rgb(info["top_down_map_pose"])
    objects_covered = (
        info["objects_covered_geometric"]["small_objects_visited"]
        + info["objects_covered_geometric"]["medium_objects_visited"]
        + info["objects_covered_geometric"]["large_objects_visited"]
    )

    print(f"===========> Steps: {steps:3d}")
    metrics_to_print = {
        "Area covered (m^2)": info["area_covered"],
        "Objects covered": objects_covered,
        "Landmarks covered": info["opsr"],
        "Novelty": info["novelty_reward"],
        "Smooth coverage": info["coverage_novelty_reward"],
    }
    for k, v in metrics_to_print.items():
        print(f"{k:<25s}: {v:6.2f}")

    steps += 1
    cv2.imshow(
        "Exploration demo",
        np.concatenate([rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1),
    )
    cv2.waitKey(30)
