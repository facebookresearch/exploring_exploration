import cv2
import pdb
import math
import numpy as np
from utils import *

import habitat


class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0):
        super(DummyRLEnv, self).__init__(config, dataset)
        self.T_exp = config.ENVIRONMENT.T_EXP
        self.T_nav = config.ENVIRONMENT.T_NAV
        assert self.T_exp + self.T_nav == config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._env_ind = env_ind

    def step(self, action):
        observations, reward, done, info = super().step(action)
        if self._env._elapsed_steps == self.T_exp:
            observations = self._respawn_agent()

        return observations, reward, done, info

    def _respawn_agent(self):
        position = self.habitat_env.current_episode.start_nav_position
        rotation = self.habitat_env.current_episode.start_nav_rotation
        observations = self.habitat_env._sim.get_observations_at(
            position, rotation, keep_agent_at_new_pose=True
        )

        observations.update(
            self.habitat_env.task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        return observations

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


config = habitat.get_config_exp_nav("demos/configs/pointnav_mp3d.yaml")
goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE

env = DummyRLEnv(config=config)
env.seed(1234)

"""
Action space:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3
"""

for i in range(10):
    obs = env.reset()
    for t in range(config.ENVIRONMENT.MAX_EPISODE_STEPS):
        if t < config.ENVIRONMENT.T_EXP:
            action = obs["oracle_action_sensor"][0].item()
        else:
            action = obs["sp_action_sensor_exp_nav"][0].item()

        obs, reward, done, info = env.step(action)

        if done:
            cv2.destroyWindow("PointNav: navigation phase")
            break

        rgb_im = proc_rgb(obs["rgb"])
        fine_occ_im = proc_rgb(obs["fine_occupancy"])
        coarse_occ_im = proc_rgb(obs["highres_coarse_occupancy"])
        topdown_im = proc_rgb(info["top_down_map_exp_nav"])

        if t < config.ENVIRONMENT.T_EXP:
            cv2.imshow(
                "PointNav: exploration phase",
                np.concatenate(
                    [rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1
                ),
            )
        else:
            if t == config.ENVIRONMENT.T_EXP:
                cv2.destroyWindow("PointNav: exploration phase")
            cv2.imshow(
                "PointNav: navigation phase",
                np.concatenate(
                    [rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1
                ),
            )

        cv2.waitKey(150)
