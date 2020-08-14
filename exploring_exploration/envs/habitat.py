#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from gym import spaces
from gym.spaces import Box

import habitat
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import PoseRLEnv, ExpNavRLEnv
from habitat_baselines.config.default_pose import get_config_pose
from habitat_baselines.config.default_exp_nav import get_config_exp_nav

from einops import rearrange


def make_vec_envs(
    config,
    device,
    devices,
    seed=100,
    task_type="pose",
    enable_odometry_noise=None,
    odometer_noise_scaling=None,
    measure_noise_free_area=None,
):
    if task_type == "pose":
        config = get_config_pose(config, [])
        env_class = PoseRLEnv
    else:
        config = get_config_exp_nav(config, [])
        env_class = ExpNavRLEnv
    config.defrost()
    config.TASK_CONFIG.SEED = seed
    config.TASK_CONFIG.SIMULATOR.SEED = seed
    if enable_odometry_noise is not None:
        config.TASK_CONFIG.SIMULATOR.ENABLE_ODOMETRY_NOISE = enable_odometry_noise
        config.TASK_CONFIG.SIMULATOR.ODOMETER_NOISE_SCALING = odometer_noise_scaling
    if measure_noise_free_area is not None:
        config.TASK_CONFIG.SIMULATOR.OCCUPANCY_MAPS.MEASURE_NOISE_FREE_AREA = (
            measure_noise_free_area
        )
    config.freeze()
    envs = construct_envs(config, env_class, devices)
    envs = BatchDataWrapper(envs)
    envs = TransposeImageWrapper(envs)
    envs = RenameKeysWrapper(envs)
    envs = DeviceWrapper(envs, device)
    return envs


class BatchDataWrapper:
    """Batches the observations received from habitat-api environments."""

    def __init__(self, vec_envs):
        self._envs = vec_envs
        self.observation_space = vec_envs.observation_spaces[0]
        self.action_space = vec_envs.action_spaces[0]

    def reset(self):
        obs = self._envs.reset()
        obs = batch_obs(obs)
        return obs

    def step(self, actions):
        actions_list = [a[0].item() for a in actions]
        outputs = self._envs.step(actions_list)
        obs, rewards, done, info = [list(x) for x in zip(*outputs)]
        obs = batch_obs(obs)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        return obs, rewards, done, info

    def close(self):
        self._envs.close()


class TransposeImageWrapper:
    """Transpose the image data from (..., H, W, C) -> (..., C, H, W)"""

    keys_to_check = [
        "rgb",
        "depth",
        "coarse_occupancy",
        "fine_occupancy",
        "highres_coarse_occupancy",
        "pose_estimation_rgb",
        "pose_estimation_depth",
    ]

    def __init__(self, vec_envs):
        self._envs = vec_envs
        self.observation_space = vec_envs.observation_space
        self.keys_to_transpose = []
        for key in self.keys_to_check:
            if key in self.observation_space.spaces:
                self.keys_to_transpose.append(key)
        for key in self.keys_to_transpose:
            self.observation_space.spaces[key] = self._transpose_obs_space(
                self.observation_space.spaces[key]
            )
        self.action_space = vec_envs.action_space

    def _transpose_obs_space(self, obs_space):
        """Transposes the observation space from (... H W C) -> (... C H W)."""
        obs_shape = obs_space.shape
        assert len(obs_shape) in [3, 4]
        if len(obs_shape) == 4:
            new_obs_shape = [obs_shape[0], obs_shape[3], obs_shape[1], obs_shape[2]]
        else:
            new_obs_shape = [obs_shape[2], obs_shape[0], obs_shape[1]]
        dtype = obs_space.dtype
        low = obs_space.low.flat[0]
        high = obs_space.high.flat[0]
        return Box(low, high, new_obs_shape, dtype=dtype)

    def _transpose_obs(self, obs):
        """Transposes the observation from (B ... H W C) -> (B ... C H W)
        """
        assert len(obs.shape) in [4, 5]
        if len(obs.shape) == 5:
            return rearrange(obs, "b n h w c -> b n c h w")
        else:
            return rearrange(obs, "b h w c -> b c h w")

    def reset(self):
        obs = self._envs.reset()
        for k in self.keys_to_transpose:
            if k in obs.keys():
                obs[k] = self._transpose_obs(obs[k])
        return obs

    def step(self, actions):
        obs, reward, done, info = self._envs.step(actions)
        for k in self.keys_to_transpose:
            if k in obs.keys():
                obs[k] = self._transpose_obs(obs[k])
        return obs, reward, done, info

    def close(self):
        self._envs.close()


class RenameKeysWrapper:
    """Renames keys from habitat-api convention to exploring_exploration
    convention.
    """

    def __init__(self, vec_envs):
        self._envs = vec_envs
        self.mapping = {
            "rgb": "im",
            "depth": "depth",
            "coarse_occupancy": "coarse_occupancy",
            "fine_occupancy": "fine_occupancy",
            "delta_sensor": "delta",
            "pose_estimation_rgb": "pose_refs",
            "pose_estimation_depth": "pose_refs_depth",
            "pose_estimation_reg": "pose_regress",
            "pose_estimation_mask": "valid_masks",
            "oracle_action_sensor": "oracle_action",
            "collision_sensor": "collisions",
            "opsr": "oracle_pose_success",
            "area_covered": "seen_area",
            "inc_area_covered": "inc_area",
            "frac_area_covered": "frac_seen_area",
            "top_down_map_pose": "topdown_map",
            "novelty_reward": "count_based_reward",
            # Navigation specific ones
            "highres_coarse_occupancy": "highres_coarse_occupancy",
            "grid_goal_exp_nav": "target_grid_loc",
            "spl_exp_nav": "spl",
            "success_exp_nav": "success_rate",
            "nav_error_exp_nav": "nav_error",
            "top_down_map_exp_nav": "topdown_map",
            "local_top_down_sensor": "gt_highres_coarse_occupancy",
        }
        self.observation_space = spaces.Dict(
            {
                self.mapping[key]: val
                for key, val in vec_envs.observation_space.spaces.items()
            }
        )
        self.action_space = vec_envs.action_space

    def reset(self):
        obs = self._envs.reset()
        obs_new = {}
        for key, val in obs.items():
            obs_new[self.mapping[key]] = val
        return obs_new

    def step(self, actions):
        obs, reward, done, infos = self._envs.step(actions)
        obs_new = {}
        for key, val in obs.items():
            obs_new[self.mapping[key]] = val

        infos_new = []
        for info in infos:
            info_new = {}
            for key, val in info.items():
                if key == "objects_covered_geometric":
                    small = val["small_objects_visited"]
                    medium = val["medium_objects_visited"]
                    large = val["large_objects_visited"]
                    categories = float(val["categories_visited"])
                    info_new["num_objects_visited"] = small + medium + large
                    info_new["num_small_objects_visited"] = small
                    info_new["num_medium_objects_visited"] = medium
                    info_new["num_large_objects_visited"] = large
                    info_new["num_categories_visited"] = categories
                elif key in self.mapping.keys():
                    info_new[self.mapping[key]] = val
                else:
                    info_new[key] = val
            infos_new.append(info_new)

        return obs_new, reward, done, infos_new

    def close(self):
        self._envs.close()


class DeviceWrapper:
    """Moves all observations to a torch device."""

    def __init__(self, vec_envs, device):
        self._envs = vec_envs
        self.device = device

        self.observation_space = vec_envs.observation_space
        self.action_space = vec_envs.action_space

    def reset(self):
        obs = self._envs.reset()
        for key, val in obs.items():
            obs[key] = val.to(self.device)

        return obs

    def step(self, actions):
        obs, reward, done, info = self._envs.step(actions)
        for key, val in obs.items():
            obs[key] = val.to(self.device)

        return obs, reward, done, info

    def close(self):
        self._envs.close()
