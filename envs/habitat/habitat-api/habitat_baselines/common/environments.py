#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import habitat
from habitat import SimulatorActions
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action
        return super().step(action)

    def get_reward_range(self):
        return (
            self._config_baseline.SLACK_REWARD - 1.0,
            self._config_baseline.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="PoseRLEnv")
class PoseRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        return observations

    def step(self, action):
        self._previous_action = action
        obs, reward, done, info = super().step(action)
        if not done:
            # Optimization: do not return these keys after the 1st step.
            for key in [
                "pose_estimation_rgb",
                "pose_estimation_depth",
                "pose_estimation_reg",
                "pose_estimation_mask",
            ]:
                _ = obs.pop(key, None)

        return obs, reward, done, info

    def get_reward_range(self):
        return (-1.0, +1.0)

    def get_reward(self, observations):
        # All rewards for this task are computed external to the environment.
        reward = 0.0
        return reward

    def _episode_success(self):
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        environment_statistics = {
            "episode_id": self.habitat_env.current_episode.episode_id,
            "scene_id": self.habitat_env.current_episode.scene_id,
        }
        metrics["environment_statistics"] = environment_statistics
        return metrics


@baseline_registry.register_env(name="ExpNavRLEnv")
class ExpNavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self.T_exp = config_env.ENVIRONMENT.T_EXP
        self.T_nav = config_env.ENVIRONMENT.T_NAV
        assert self.T_exp + self.T_nav == config_env.ENVIRONMENT.MAX_EPISODE_STEPS
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action

        observations, reward, done, info = super().step(action)
        if self._env._elapsed_steps == self.T_exp:
            observations = self._respawn_agent()
            info["finished_exploration"] = True
        else:
            info["finished_exploration"] = False

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
        return (
            self._config_baseline.SLACK_REWARD - 1.0,
            self._config_baseline.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        if self._env._elapsed_steps >= self.T_exp:
            reward = self._config_baseline.SLACK_REWARD

            current_target_distance = self._distance_target()
            reward += self._previous_target_distance - current_target_distance
            self._previous_target_distance = current_target_distance

            if self._episode_success():
                reward += self._config_baseline.SUCCESS_REWARD
        else:
            return 0.0

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env._elapsed_steps >= self.T_exp and (
            self._env.episode_over or self._episode_success()
        ):
            done = True
        return done

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        environment_statistics = {
            "episode_id": self.habitat_env.current_episode.episode_id,
            "scene_id": self.habitat_env.current_episode.scene_id,
        }
        metrics["environment_statistics"] = environment_statistics
        return metrics
