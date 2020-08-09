#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Type

import pdb
import attr
import cv2
import math
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorSuite,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
    SimulatorActions,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
    compute_heading_from_quaternion,
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat.utils.visualizations.utils import topdown_to_image
from habitat.tasks.exp_nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.nav_task import NavigationGoal, NavigationEpisode, NavigationTask

MAP_THICKNESS_SCALAR: int = 1250
RGBSENSOR_DIMENSION: int = 3


def merge_sim_episode_config(sim_config: Config, episode: Type[Episode]) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if episode.start_position is not None and episode.start_rotation is not None:
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class ExploreNavigationGoal(NavigationGoal):
    r"""A navigation goal that can be specified by position and
    rotation (optional).
    """

    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class ExploreNavigationEpisode(NavigationEpisode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        start_nav_position: numpy ndarray containing 3 entries for (x, y, z)
        start_nav_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    start_nav_position: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_nav_rotation: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: List[ExploreNavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor
class ImageGoalSensorExploreNavigation(Sensor):
    r"""Sensor for ImageGoal observations which are used in the ExploreNavigation task.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        super().__init__(config=config)
        self.current_episode_id = None
        self.current_target = None
        self._num_steps = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "image_goal_exp_nav"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self._num_steps = 0.0

        if self._num_steps >= self.T_exp:
            tgt_obs = self._sim.get_observations_at(
                episode.goals[0].position, episode.goals[0].rotation
            )
            tgt_im = tgt_obs["rgb"]
        else:
            tgt_im = np.zeros(
                (self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
                dtype=np.uint8,
            )

        self._num_steps += 1
        return tgt_im


@registry.register_sensor
class GridGoalSensorExploreNavigation(Sensor):
    r"""Sensor for PointNav coordinates in the map grid space.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        super().__init__(config=config)
        self.current_episode_id = None
        self.current_target = None
        self._num_steps = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "grid_goal_exp_nav"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-100000000.0, high=100000000.0, shape=(2,), dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self._num_steps = 0.0

        if self._num_steps >= self.T_exp:
            agent_position = self._sim.get_agent_state().position
            agent_rotation = self._sim.get_agent_state().rotation
            tgt_position = np.array(episode.goals[0].position)
            tgt_grid_pos = self._get_egocentric_grid_loc(
                agent_position, agent_rotation, tgt_position
            )
        else:
            tgt_grid_pos = np.zeros((2,))

        self._num_steps += 1
        return tgt_grid_pos

    def _get_egocentric_grid_loc(
        self,
        agent_position: np.array,
        agent_rotation: np.quaternion,
        tgt_position: np.array,
    ) -> np.array:
        """Provides the target location in terms of (x, y) coordinates in the
        current egocentric occupancy map. The convention for the egocentric
        occupancy map is that the agent is at the center of the map, with
        forward direction being upward and the agent's right being the map's
        rightward direction.
        Args:
            agent_position - current agent position in simulator (X, Y, Z)
            agent_rotation - current agent rotation as a quaternion
            tgt_position - navigation goal position in simulator (X, Y, Z)
        """
        # Changing coordinate system from -Z as forward, X as rightward
        # to X as forward, Y as rightward.
        tgt_x = -tgt_position[2]
        tgt_y = tgt_position[0]
        curr_x = -agent_position[2]
        curr_y = agent_position[0]
        curr_t = compute_heading_from_quaternion(agent_rotation)
        # Target in egocentric polar coordinates.
        r_ct = math.sqrt((tgt_x - curr_x) ** 2 + (tgt_y - curr_y) ** 2)
        p_ct = math.atan2(tgt_y - curr_y, tgt_x - curr_x) - curr_t
        # Convert to map grid coordinates with X rightward, Y downward.
        grid_size = self._sim.config.OCCUPANCY_MAPS.MAP_SCALE
        W = self._sim.config.HIGHRES_COARSE_OCC_SENSOR.WIDTH
        H = self._sim.config.HIGHRES_COARSE_OCC_SENSOR.HEIGHT
        large_map_range = self._sim.config.OCCUPANCY_MAPS.LARGE_MAP_RANGE
        Wby2 = W // 2
        Hby2 = H // 2
        disp_y = -r_ct * math.cos(p_ct) / grid_size
        disp_x = r_ct * math.sin(p_ct) / grid_size
        # Map coordinates to occupancy image coordinates.
        # Accounts for conversion from global-map cropping to image size, and
        # shifts origin to top-left corner.
        disp_y = Hby2 + H * disp_y / (2 * large_map_range + 1)
        disp_x = Wby2 + W * disp_x / (2 * large_map_range + 1)
        # Clip the values to be within the occupancy image extents. This might
        # be a problem in large environments where the large_map_range does not
        # cover the target.
        grid_y = np.clip(disp_y, 0, H - 1)
        grid_x = np.clip(disp_x, 0, W - 1)

        return np.array([grid_x, grid_y])


@registry.register_sensor
class SPActionSensorExploreNavigation(Sensor):
    r"""Sensor that returns the shortest path action to the navigation target.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor. 

    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

        self.current_episode_id = None
        self.target_position = None
        goal_radius = config.GOAL_RADIUS
        self.goal_radius = goal_radius
        self.follower = ShortestPathFollower(sim, goal_radius, False)
        self.follower.mode = "geodesic_path"
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        self._step_count = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "sp_action_sensor_exp_nav"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32,)

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            # A new episode has started
            self.current_episode_id = episode_id
            self.target_position = episode.goals[0].position
            self._step_count = 0

        if self._step_count >= self.T_exp:
            current_target = self.target_position
            agent_position = self._sim.get_agent_state().position
            oracle_action = self.follower.get_next_action(current_target)
        else:
            # Some constant action
            oracle_action = SimulatorActions.MOVE_FORWARD

        self._step_count += 1

        return np.array([oracle_action], dtype=np.int32)


@registry.register_measure
class TopDownMapExpNav(Measure):
    r"""Top Down Map measure. During the exploration phase, this serves as
    an exploration top-down view without the target marked. During the
    navigation phase, this serves as a regular navigation map with both
    source and target marked.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map_exp_nav"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def draw_source_and_target(self, episode):
        if self._step_count < self.T_exp:
            # mark source point
            s_x, s_y = maps.to_grid(
                episode.start_position[0],
                episode.start_position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )
        else:
            # mark source point
            s_x, s_y = maps.to_grid(
                episode.start_nav_position[0],
                episode.start_nav_position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )

        point_padding = 2 * int(np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR))
        self._top_down_map[
            s_x - point_padding : s_x + point_padding + 1,
            s_y - point_padding : s_y + point_padding + 1,
        ] = maps.MAP_SOURCE_POINT_INDICATOR

        if self._step_count >= self.T_exp:
            # mark target point
            t_x, t_y = maps.to_grid(
                episode.goals[0].position[0],
                episode.goals[0].position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )
            self._top_down_map[
                t_x - point_padding : t_x + point_padding + 1,
                t_y - point_padding : t_y + point_padding + 1,
            ] = maps.MAP_TARGET_POINT_INDICATOR

    def reset_metric(self, episode):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target points last to avoid overlap
        if self._config.DRAW_SOURCE_AND_TARGET:
            self.draw_source_and_target(episode)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min - self._grid_delta : self._ind_x_max + self._grid_delta,
            self._ind_y_min - self._grid_delta : self._ind_y_max + self._grid_delta,
        ]

    def update_metric(self, episode, action):
        if self._step_count == self.T_exp:
            self._top_down_map = self.get_original_map()
            agent_position = self._sim.get_agent_state().position
            a_x, a_y = maps.to_grid(
                agent_position[0],
                agent_position[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )

            self._previous_xy_location = (a_y, a_x)
            if self._config.DRAW_SHORTEST_PATH:
                # draw shortest path
                self._shortest_path_points = self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
                self._shortest_path_points = [
                    maps.to_grid(
                        p[0],
                        p[2],
                        self._coordinate_min,
                        self._coordinate_max,
                        self._map_resolution,
                    )[::-1]
                    for p in self._shortest_path_points
                ]
                maps.draw_path(
                    self._top_down_map,
                    self._shortest_path_points,
                    maps.MAP_SHORTEST_PATH_COLOR,
                    self.line_thickness,
                )

            self.update_fog_of_war_mask(np.array([a_x, a_y]))

            # draw source and target points last to avoid overlap
            if self._config.DRAW_SOURCE_AND_TARGET:
                self.draw_source_and_target(episode)

        self._step_count += 1

        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

        self._metric = topdown_to_image(self._metric)

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )


@registry.register_measure
class SPLExpNav(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        self._step_count = None
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl_exp_nav"

    def reset_metric(self, episode):
        self._step_count = 0
        # self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(self, episode, action):

        # The metric is computed only after the exploration phase ends.
        if self._step_count > self.T_exp + 1:
            ep_success = 0
            current_position = self._sim.get_agent_state().position.tolist()
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            if (
                action == self._sim.index_stop_action
                and distance_to_target < self._config.SUCCESS_DISTANCE
            ):
                ep_success = 1
            self._agent_episode_distance += self._euclidean_distance(
                current_position, self._previous_position
            )
            self._previous_position = current_position
            self._metric = ep_success * (
                self._start_end_episode_distance
                / max(self._start_end_episode_distance, self._agent_episode_distance)
            )
        else:
            self._metric = 0.0

        self._step_count += 1

        if self._step_count == self.T_exp + 1:
            self._previous_position = self._sim.get_agent_state().position.tolist()


@registry.register_measure
class SuccessExpNav(Measure):
    r"""Success Rate
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        self._step_count = None
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "success_exp_nav"

    def reset_metric(self, episode):
        self._step_count = 0
        self._metric = None

    def update_metric(self, episode, action):
        # The metric is computed only after the exploration phase ends.
        if self._step_count > self.T_exp + 1:
            ep_success = 0.0
            current_position = self._sim.get_agent_state().position.tolist()
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            if (
                action == self._sim.index_stop_action
                and distance_to_target < self._config.SUCCESS_DISTANCE
            ):
                ep_success = 1.0
            self._metric = ep_success
        else:
            self._metric = 0.0

        self._step_count += 1


@registry.register_measure
class NavigationErrorExpNav(Measure):
    r"""Navigation Error - geodesic distance to target at the current time
    step.

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self.T_exp = config.T_EXP
        self.T_nav = config.T_NAV
        self._step_count = None
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "nav_error_exp_nav"

    def reset_metric(self, episode):
        self._step_count = 0
        self._metric = None

    def update_metric(self, episode, action):
        # The metric is computed only after the exploration phase ends.
        if self._step_count > self.T_exp + 1:
            current_position = self._sim.get_agent_state().position.tolist()
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            self._metric = distance_to_target
        else:
            self._metric = math.inf

        self._step_count += 1


@registry.register_task(name="ExpNav-v0")
class ExploreNavigationTask(NavigationTask):
    def __init__(
        self, task_config: Config, sim: Simulator, dataset: Optional[Dataset] = None,
    ) -> None:

        super().__init__(task_config=task_config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Type[Episode]) -> Any:
        return merge_sim_episode_config(sim_config, episode)
