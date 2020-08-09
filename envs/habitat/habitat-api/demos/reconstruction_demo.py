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

def create_reference_grid(refs_uint8):
    """
    Inputs:
        refs_uint8 - (nRef, H, W, C) numpy array
    """
    refs_uint8 = np.copy(refs_uint8)
    nRef, H, W, C = refs_uint8.shape

    nrow = int(math.sqrt(nRef))

    ncol = nRef // nrow # (number of images per column)
    if nrow * ncol < nRef:
        ncol += 1
    final_grid = np.zeros((nrow * ncol, *refs_uint8.shape[1:]), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    final_grid[:nRef] = refs_uint8
    final_grid = final_grid.reshape(ncol, nrow, *final_grid.shape[1:]) # (ncol, nrow, H, W, C)
    final_grid = final_grid.transpose(0, 2, 1, 3, 4)
    final_grid = final_grid.reshape(ncol * H, nrow * W, C)
    return final_grid

config = habitat.get_config_pose('demos/config/pose_estimation_mp3d.yaml')
goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE

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

pose_refs_rgb = proc_rgb(create_reference_grid(obs['pose_estimation_rgb']))

while True:
    action = obs['oracle_action_sensor'][0].item()
    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
        pose_refs_rgb = proc_rgb(create_reference_grid(obs['pose_estimation_rgb']))

    rgb_im = proc_rgb(obs['rgb'])
    topdown_im = proc_rgb(info['top_down_map_pose'])

    cv2.imshow('Reconstruction demo', np.concatenate([rgb_im, topdown_im, pose_refs_rgb], axis=1))
    cv2.waitKey(60)
