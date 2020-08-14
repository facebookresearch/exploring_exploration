#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import gym
import gym_avd
from gym.spaces.box import Box

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from einops import rearrange


def make_env(
    env_id,
    seed,
    rank,
    log_dir,
    allow_early_resets,
    split="train",
    nRef=1,
    set_return_topdown_map=False,
    tdn_min_dist=2000.0,
    tdn_t_exp=200,
    tdn_t_nav=200,
    provide_collision_penalty=False,
    collision_penalty_factor=1e-1,
    n_landmarks=20,
):
    # Define a temporary function that creates an environment instance.
    def _thunk():
        env = gym.make(env_id)
        env.set_split(split)
        env.set_nref(nRef)
        if set_return_topdown_map:
            env.set_return_topdown_map()
        env.set_nlandmarks(n_landmarks)
        if "tdn" in env_id:
            env.set_min_dist(tdn_min_dist)
            env.set_t_exp_and_nav(tdn_t_exp, tdn_t_nav)
        env.seed(seed + rank)
        obs_shape = env.observation_space.shape
        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets,
            )
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.spaces["im"].shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImageDict(env)
        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    num_processes,
    log_dir,
    device,
    allow_early_resets,
    num_frame_stack=None,
    nRef=1,
    n_landmarks=20,
    set_return_topdown_map=False,
    **kwargs
):
    envs = [
        make_env(
            env_name,
            seed,
            i,
            log_dir,
            allow_early_resets,
            nRef=nRef,
            set_return_topdown_map=set_return_topdown_map,
            n_landmarks=n_landmarks,
            **kwargs
        )
        for i in range(num_processes)
    ]

    envs = SubprocVecEnv(envs)
    envs = VecPyTorchDict(envs, device)

    return envs


class TransposeImageDict(gym.ObservationWrapper):
    """Transpose the image data from (..., H, W, C) -> (..., C, H, W)."""

    keys_to_check = [
        "im",
        "depth",
        "coarse_occupancy",
        "fine_occupancy",
        "highres_coarse_occupancy",
        "target_im",
        "pose_refs",
        "pose_refs_depth",
        "landmark_ims",
    ]

    def __init__(self, env=None):
        super().__init__(env)
        self.keys_to_transpose = []
        for key in self.keys_to_check:
            if key in self.observation_space.spaces:
                self.keys_to_transpose.append(key)
        for key in self.keys_to_transpose:
            self.observation_space.spaces[key] = self._transpose_obs_space(
                self.observation_space.spaces[key]
            )

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
        """Transposes the observation from (... H W C) -> (... C H W)
        """
        assert len(obs.shape) in [3, 4]
        if len(obs.shape) == 4:
            return rearrange(obs, "n h w c -> n c h w")
        else:
            return rearrange(obs, "h w c -> c h w")

    def observation(self, observation):
        for key in self.keys_to_transpose:
            if key in observation.keys():
                observation[key] = self._transpose_obs(observation[key])
        return observation


class VecPyTorchDict(VecEnvWrapper):
    def __init__(self, venv, device):
        """Converts numpy arrays to torch sensors and load them to GPU."""
        super(VecPyTorchDict, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = {key: torch.from_numpy(obs[key]).float().to(self.device) for key in obs}
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {key: torch.from_numpy(obs[key]).float().to(self.device) for key in obs}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
