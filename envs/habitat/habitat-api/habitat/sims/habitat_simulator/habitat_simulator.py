#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, List, Optional, Tuple

import os
import cv2
import gzip
import json
import math
import quaternion  # noqa # pylint: disable=unused-import
import numpy as np
from gym import Space, spaces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import habitat_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    FineOccSensor,
    CoarseOccSensor,
    HighResCoarseOccSensor,
    ProjOccSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    SimulatorActions,
)
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
    compute_egocentric_delta,
    compute_heading_from_quaternion,
    compute_quaternion_from_heading,
    compute_updated_pose,
    truncated_normal_noise,
)

from habitat.utils.visualizations import maps

RGBSENSOR_DIMENSION = 3


def overwrite_config(config_from: Config, config_to: Any) -> None:
    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), value)


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class HabitatSimDepthSensor(DepthSensor):
    sim_sensor_type: habitat_sim.SensorType
    min_depth_value: float
    max_depth_value: float

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / self.config.MAX_DEPTH

        obs = np.expand_dims(obs, axis=2)  # make depth observation a 3D array

        return obs


@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.uint32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs


@registry.register_sensor
class HabitatSimFineOccSensor(FineOccSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class HabitatSimCoarseOccSensor(CoarseOccSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class HabitatSimHighResCoarseOccSensor(HighResCoarseOccSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_sensor
class HabitatSimProjOccSensor(ProjOccSensor):
    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@registry.register_simulator(name="Sim-v0")
class HabitatSim(Simulator):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene.id
        self._sim = habitat_sim.Simulator(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._is_episode_active = False
        # Handling noisy odometer scenario
        self._estimated_position = None
        self._estimated_rotation = None
        self._enable_odometer_noise = self.config.ENABLE_ODOMETRY_NOISE
        self._odometer_noise_eta = self.config.ODOMETER_NOISE_SCALING

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.scene.id = self.config.SCENE
        sim_config.gpu_device_id = self.config.HABITAT_SIM_V0.GPU_DEVICE_ID
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(config_from=self._get_agent_config(), config_to=agent_config)

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)
            sim_sensor_cfg.position = sensor.config.POSITION
            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def is_episode_active(self) -> bool:
        return self._is_episode_active

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION, agent_cfg.START_ROTATION, agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self):
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        # If noisy odometer is enabled, maintain an
        # estimated position and rotation for the agent.
        if self._enable_odometer_noise:
            agent_state = self.get_agent_state()
            # Initialize with the ground-truth position, rotation.
            self._estimated_position = agent_state.position
            self._estimated_rotation = agent_state.rotation

        self._prev_sim_obs = sim_obs
        self._is_episode_active = True
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action):
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        agent_state = self.get_agent_state()
        position_before_step = agent_state.position
        rotation_before_step = agent_state.rotation

        if action == self.index_stop_action:
            self._is_episode_active = False
            sim_obs = self._sim.get_sensor_observations()
        else:
            sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs

        agent_state = self.get_agent_state()
        position_after_step = agent_state.position
        rotation_after_step = agent_state.rotation

        # Compute the estimated position, rotation.
        if self._enable_odometer_noise and action != self.index_stop_action:
            # Measure ground-truth delta in egocentric coordinates.
            delta_rpt_gt = compute_egocentric_delta(
                position_before_step,
                rotation_before_step,
                position_after_step,
                rotation_after_step,
            )
            delta_y_gt = position_after_step[1] - position_before_step[1]
            # Add noise to the ground-truth delta.
            eta = self._odometer_noise_eta
            D_rho, D_phi, D_theta = delta_rpt_gt
            D_rho_n = D_rho + truncated_normal_noise(eta, 2 * eta) * D_rho
            D_phi_n = D_phi
            D_theta_n = D_theta + truncated_normal_noise(eta, 2 * eta) * D_theta
            delta_rpt_n = np.array((D_rho_n, D_phi_n, D_theta_n))
            delta_y_n = delta_y_gt
            # Update noisy pose estimates
            old_position = self._estimated_position
            old_rotation = self._estimated_rotation
            (new_position, new_rotation) = compute_updated_pose(
                old_position, old_rotation, delta_rpt_n, delta_y_n
            )
            self._estimated_position = new_position
            self._estimated_rotation = new_rotation

        observations = self._sensor_suite.get_observations(sim_obs)
        return observations

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        sim_obs = self._sim.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = config.SCENE == self._current_scene
        self.config = config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = config.SCENE
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(position_a, dtype=np.float32)
        path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self):
        return self._sim.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]):
        return self._sim.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self._sim.semantic_scene

    def close(self):
        self._sim.close()

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self._sim.get_agent(agent_id)
        original_state = self.get_agent_state(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = dict()

        agent.set_state(new_state, reset_sensors)

        if not self._check_agent_position(position, agent_id):
            agent.set_state(original_state, reset_sensors)
            return False
        return True

    def get_observations_at(
        self,
        position: List[float],
        rotation: List[float],
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:

        current_state = self.get_agent_state()

        success = self.set_agent_state(position, rotation, reset_sensors=False)
        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position, current_state.rotation, reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_specific_sensor_observations_at(
        self, position: List[float], rotation: List[float], sensor_uuid: str,
    ) -> Optional[Observations]:

        current_state = self.get_agent_state()
        success = self.set_agent_state(position, rotation, reset_sensors=False)

        if success:
            specific_sim_obs = self._sim.get_specific_sensor_observations(sensor_uuid)
            self.set_agent_state(
                current_state.position, current_state.rotation, reset_sensors=False,
            )
            return specific_sim_obs
        else:
            return None

    # TODO (maksymets): Remove check after simulator became stable
    def _check_agent_position(self, position, agent_id=0) -> bool:
        if not np.allclose(position, self.get_agent_state(agent_id).position):
            logger.info("Agent state diverges from configured start position.")
            return False
        return True

    def distance_to_closest_obstacle(self, position, max_search_radius=2.0):
        return self._sim.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position):
        return self._sim.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)

    def get_environment_extents(self) -> Tuple[float, float, float, float]:
        """Returns the minimum and maximum X, Z coordinates navigable on
        the current floor.
        """
        num_samples = 20000
        start_height = self.get_agent_state().position[1]
        min_x, max_x = (math.inf, -math.inf)
        min_z, max_z = (math.inf, -math.inf)
        for _ in range(num_samples):
            point = self.sample_navigable_point()
            # Check if on same level as original
            if np.abs(start_height - point[1]) > 0.5:
                continue
            min_x = min(point[0], min_x)
            max_x = max(point[0], max_x)
            min_z = min(point[2], min_z)
            max_z = max(point[2], max_z)

        return (min_x, min_z, max_x, max_z)


@registry.register_simulator(name="Sim-v1")
class HabitatSimOcc(HabitatSim):
    r"""Simulator wrapper over HabitatSim that additionally builds an
    occupancy map of the environment as the agent is moving.

    Args:
        config: configuration for initializing the simulator.

    Acknowledgement: large parts of the occupancy generation code were
    borrowed from https://github.com/taochenshh/exp4nav with some
    modifications for faster processing.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.initialize_map(config)

    def initialize_map(self, config):
        r"""Initializes the map configurations and useful variables for map
        computation.
        """
        occ_cfg = config.OCCUPANCY_MAPS
        # ======================= Store map configurations ====================
        occ_info = {
            "map_scale": occ_cfg.MAP_SCALE,
            "map_size": occ_cfg.MAP_SIZE,
            "max_depth": occ_cfg.MAX_DEPTH,
            "small_map_range": occ_cfg.SMALL_MAP_RANGE,
            "large_map_range": occ_cfg.LARGE_MAP_RANGE,
            "small_map_size": config.FINE_OCC_SENSOR.WIDTH,
            "large_map_size": config.COARSE_OCC_SENSOR.WIDTH,
            "height_threshold": (occ_cfg.HEIGHT_LOWER, occ_cfg.HEIGHT_UPPER),
            "get_proj_loc_map": occ_cfg.GET_PROJ_LOC_MAP,
            "use_gt_occ_map": occ_cfg.USE_GT_OCC_MAP,
            # NOTE: This assumes that there is only one agent
            "agent_height": config.AGENT_0.HEIGHT,
            "Lx_min": None,
            "Lx_max": None,
            "Lz_min": None,
            "Lz_max": None,
            # Coverage novelty reward
            "coverage_novelty_pooling": config.OCCUPANCY_MAPS.COVERAGE_NOVELTY_POOLING,
        }
        # High-resolution map options.
        occ_info["get_highres_loc_map"] = occ_cfg.GET_HIGHRES_LOC_MAP
        if occ_info["get_highres_loc_map"]:
            occ_info["highres_large_map_size"] = config.HIGHRES_COARSE_OCC_SENSOR.WIDTH
        # Measure noise-free area covered or noisy area covered?
        if self._enable_odometer_noise:
            occ_info["measure_noise_free_area"] = occ_cfg.MEASURE_NOISE_FREE_AREA
        else:
            occ_info["measure_noise_free_area"] = False
        # Camera intrinsics.
        hfov = math.radians(self._sensor_suite.sensors["depth"].config.HFOV)
        v1 = 1.0 / np.tan(hfov / 2.0)
        v2 = 1.0 / np.tan(hfov / 2.0)  # Assumes both FoVs are same.
        intrinsic_matrix = np.array(
            [
                [v1, 0.0, 0.0, 0.0],
                [0.0, v2, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        occ_info["intrinsic_matrix"] = intrinsic_matrix
        occ_info["inverse_intrinsic_matrix"] = np.linalg.inv(intrinsic_matrix)
        self.occupancy_info = occ_info
        # ======================= Object annotations ==========================
        self.has_object_annotations = config.OBJECT_ANNOTATIONS.IS_AVAILABLE
        self.object_annotations_dir = config.OBJECT_ANNOTATIONS.PATH
        # ========== Memory to be allocated at the start of an episode ========
        self.grids_mat = None
        self.count_grids_mat = None
        self.noise_free_grids_mat = None
        self.gt_grids_mat = None
        self.proj_grids_mat = None
        # ========================== GT topdown map ===========================
        self._gt_top_down_map = None
        # Maintain a cache to avoid redundant computation and to store
        # useful statistics.
        self._cache = {}
        W = config.DEPTH_SENSOR.WIDTH
        # Cache meshgrid for depth projection.
        # [1, -1] for y as array indexing is y-down while world is y-up.
        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        self._cache["xs"] = xs
        self._cache["ys"] = ys

    def create_grid_memory(self):
        r"""Pre-assigns memory for global grids which are used to aggregate
        the per-frame occupancy maps.
        """
        grid_size = self.occupancy_info["map_scale"]
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        # Compute map size conditioned on environment extents.
        # Add a 5m buffer to account for noise in extent estimates.
        Lx_min, Lx_max, Lz_min, Lz_max = min_x - 5, max_x + 5, min_z - 5, max_z + 5
        is_same_environment = (
            (Lx_min == self.occupancy_info["Lx_min"])
            and (Lx_max == self.occupancy_info["Lx_max"])
            and (Lz_min == self.occupancy_info["Lz_min"])
            and (Lz_max == self.occupancy_info["Lz_max"])
        )
        # Only if the environment changes, create new arrays.
        if not is_same_environment:
            # Update extents data
            self.occupancy_info["Lx_min"] = Lx_min
            self.occupancy_info["Lx_max"] = Lx_max
            self.occupancy_info["Lz_min"] = Lz_min
            self.occupancy_info["Lz_max"] = Lz_max
            grid_num = (
                int((Lx_max - Lx_min) / grid_size),
                int((Lz_max - Lz_min) / grid_size),
            )
            # Create new arrays
            self.grids_mat = np.zeros(grid_num, np.uint8)
            self.count_grids_mat = np.zeros(grid_num, dtype=np.float32)
            if self.occupancy_info["measure_noise_free_area"]:
                self.noise_free_grids_mat = np.zeros(grid_num, np.uint8)
            if self.occupancy_info["use_gt_occ_map"]:
                self.gt_grids_mat = np.zeros(grid_num, np.uint8)
            """
            The local projection has 3 channels.
            One each for occupied, free and unknown.
            """
            if self.occupancy_info["get_proj_loc_map"]:
                self.proj_grids_mat = np.zeros((*grid_num, 3), np.uint8)

    def reset(self):
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()
        agent_state = self.get_agent_state()

        # If noisy odometer is enabled, maintain an
        # estimated position and rotation for the agent.
        if self._enable_odometer_noise:
            # Initialize with the ground-truth position, rotation
            self._estimated_position = agent_state.position
            self._estimated_rotation = agent_state.rotation
        # Create map memory and reset stats
        self.create_grid_memory()
        self.reset_occupancy_stats()
        # Obtain ground-truth environment layout
        self.gt_map_creation_height = agent_state.position[1]
        if self.occupancy_info["use_gt_occ_map"]:
            self._gt_top_down_map = self.get_original_map()
        # Update map based on current observations
        sim_obs = self._update_map_observations(sim_obs)
        # Load object annotations if available
        if self.has_object_annotations:
            scene_id = self._current_scene.split("/")[-1]
            annot_path = f"{self.object_annotations_dir}/{scene_id}.json.gz"
            with gzip.open(annot_path, "rt") as fp:
                self.object_annotations = json.load(fp)

        self._prev_sim_obs = sim_obs
        self._is_episode_active = True
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action):
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        agent_state = self.get_agent_state()
        position_before_step = agent_state.position
        rotation_before_step = agent_state.rotation

        if action == self.index_stop_action:
            self._is_episode_active = False
            sim_obs = self._sim.get_sensor_observations()
        else:
            sim_obs = self._sim.step(action)

        agent_state = self.get_agent_state()
        position_after_step = agent_state.position
        rotation_after_step = agent_state.rotation

        # Compute the estimated position, rotation.
        if self._enable_odometer_noise and action != self.index_stop_action:
            # Measure ground-truth delta in egocentric coordinates.
            delta_rpt_gt = compute_egocentric_delta(
                position_before_step,
                rotation_before_step,
                position_after_step,
                rotation_after_step,
            )
            delta_y_gt = position_after_step[1] - position_before_step[1]
            # Add noise to the ground-truth delta.
            eta = self._odometer_noise_eta
            D_rho, D_phi, D_theta = delta_rpt_gt
            D_rho_n = D_rho + truncated_normal_noise(eta, 2 * eta) * D_rho
            D_phi_n = D_phi
            D_theta_n = D_theta + truncated_normal_noise(eta, 2 * eta) * D_theta
            delta_rpt_n = np.array((D_rho_n, D_phi_n, D_theta_n))
            delta_y_n = delta_y_gt
            # Update noisy pose estimates
            old_position = self._estimated_position
            old_rotation = self._estimated_rotation
            (new_position, new_rotation) = compute_updated_pose(
                old_position, old_rotation, delta_rpt_n, delta_y_n
            )
            self._estimated_position = new_position
            self._estimated_rotation = new_rotation
        # Update map based on current observations
        sim_obs = self._update_map_observations(sim_obs)

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def convert_to_pointcloud(
        self,
        rgb: np.array,
        depth: np.array,
        agent_position: np.array,
        agent_rotation: np.quaternion,
    ) -> Tuple[np.array, Optional[np.array]]:
        """Converts depth input into a sequence of points corresponding to
        the 3D projection of camera points by using both intrinsic and
        extrinsic parameters.

        Args:
            rgb - uint8 RGB images
            depth - normalized depth inputs with values lying in [0.0, 1.0]
            agent_position - pre-computed agent position for efficiency
            agent_rotation - pre-computed agent rotation for efficiency
        Returns:
            xyz_world - a sequence of (x, y, z) real-world coordinates,
                        may contain noise depending on the settings.
            xyz_world_nf - a sequence of (x, y, z) real-world coordinates,
                            strictly noise-free (Optional).
        """
        # =============== Unnormalize depth input if applicable ===============
        depth_sensor = self._sensor_suite.sensors["depth"]
        min_depth_value = depth_sensor.config.MIN_DEPTH
        max_depth_value = depth_sensor.config.MAX_DEPTH
        if depth_sensor.config.NORMALIZE_DEPTH:
            depth_float = depth.astype(np.float32) * max_depth_value + min_depth_value
        else:
            depth_float = depth.astype(np.float32)
        depth_float = depth_float[..., 0]
        # ========== Convert to camera coordinates using intrinsics ===========
        W = depth.shape[1]
        xs = np.copy(self._cache["xs"]).reshape(-1)
        ys = np.copy(self._cache["ys"]).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths.
        valid_depths = (depth_float != 0.0) & (
            depth_float <= self.occupancy_info["max_depth"]
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Project to 3D coordinates.
        # Negate depth as the camera looks along -Z.
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.occupancy_info["inverse_intrinsic_matrix"]
        xyz_cam = np.matmul(inv_K, xys)
        ## Uncomment for visualizing point-clouds in camera coordinates.
        # colors = rgb.reshape(-1, 3)
        # colors = colors[valid_depths, :]
        # cv2.imshow('RGB', rgb[:, :, ::-1])
        # cv2.imshow('Depth', depth)
        # cv2.waitKey(0)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # colors = colors.astype(np.float32)/255.0
        # ax.scatter(xyz_cam[0, :], xyz_cam[1, :], xyz_cam[2, :], c = colors)
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.view_init(elev=0.0, azim=-90.0)
        # plt.show()
        # =========== Convert to world coordinates using extrinsics ===========
        T_world = np.eye(4)
        T_world[:3, :3] = quaternion.as_rotation_matrix(agent_rotation)
        T_world[:3, 3] = agent_position
        xyz_world = np.matmul(T_world, xyz_cam).T
        # Convert to non-homogeneous coordinates
        xyz_world = xyz_world[:, :3] / xyz_world[:, 3][:, np.newaxis]
        # ============ Compute noise-free point-cloud if required =============
        xyz_world_nf = None
        if self.occupancy_info["measure_noise_free_area"]:
            agent_state = self.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation
            T_world = np.eye(4)
            T_world[:3, :3] = quaternion.as_rotation_matrix(agent_rotation)
            T_world[:3, 3] = agent_position
            xyz_world_nf = np.matmul(T_world, xyz_cam).T
            # Convert to non-homogeneous coordinates
            xyz_world_nf = xyz_world_nf[:, :3] / xyz_world_nf[:, 3][:, np.newaxis]
        return xyz_world, xyz_world_nf

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self.initialize_map(config)

    def get_observations_at(
        self,
        position: List[float],
        rotation: List[float],
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:

        current_state = self.get_agent_state()

        success = self.set_agent_state(position, rotation, reset_sensors=False)
        if success:
            sim_obs = self._sim.get_sensor_observations()
            if keep_agent_at_new_pose:
                sim_obs = self._update_map_observations(sim_obs)
            else:
                # Difference being that the global map will not be updated
                # using the current observation.
                (
                    fine_occupancy,
                    coarse_occupancy,
                    highres_coarse_occupancy,
                ) = self.get_local_maps()
                sim_obs["coarse_occupancy"] = coarse_occupancy
                sim_obs["fine_occupancy"] = fine_occupancy
                sim_obs["highres_coarse_occupancy"] = highres_coarse_occupancy
                if self.occupancy_info["get_proj_loc_map"]:
                    proj_occupancy = self.get_proj_loc_map()
                    sim_obs["proj_occupancy"] = proj_occupancy
            self._prev_sim_obs = sim_obs
            # Process observations using sensor_suite.
            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position, current_state.rotation, reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_original_map(self) -> np.array:
        r"""Returns the top-down environment layout in the current global
        map scale.
        """
        x_min = self.occupancy_info["Lx_min"]
        x_max = self.occupancy_info["Lx_max"]
        z_min = self.occupancy_info["Lz_min"]
        z_max = self.occupancy_info["Lz_max"]
        top_down_map = maps.get_topdown_map_v2(
            self, (x_min, x_max, z_min, z_max), self.occupancy_info["map_scale"], 20000,
        )
        return top_down_map

    def reset_occupancy_stats(self):
        r"""Resets occupancy maps, area estimates.
        """
        self.occupancy_info["seen_area"] = 0
        self.occupancy_info["inc_area"] = 0
        self.grids_mat.fill(0)
        self.count_grids_mat.fill(0)
        if self.occupancy_info["measure_noise_free_area"]:
            self.noise_free_grids_mat.fill(0)
        if self.occupancy_info["use_gt_occ_map"]:
            self.gt_grids_mat.fill(0)
        if self.occupancy_info["get_proj_loc_map"]:
            self.proj_grids_mat.fill(0)

    def get_seen_area(
        self,
        rgb: np.array,
        depth: np.array,
        out_mat: np.array,
        count_out_mat: np.array,
        gt_out_mat: Optional[np.array],
        proj_out_mat: Optional[np.array],
        noise_free_out_mat: Optional[np.array],
    ) -> int:
        r"""Given new RGBD observations, it updates the global occupancy map
        and computes total area seen after the update.

        Args:
            rgb - uint8 RGB images.
            depth - normalized depth inputs with values lying in [0.0, 1.0].
            *out_mat - global map to aggregate current inputs in.
        Returns:
            Area seen in the environment after aggregating current inputs.
            Area is measured in gridcells. Multiply by map_scale**2 to
            get area in m^2.
        """
        agent_state = self.get_agent_state()
        if self._enable_odometer_noise:
            agent_position = self._estimated_position
            agent_rotation = self._estimated_rotation
        else:
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation
        # ====================== Compute the pointcloud =======================
        XYZ_ego, XYZ_ego_nf = self.convert_to_pointcloud(
            rgb, depth, agent_position, agent_rotation
        )
        # Normalizing the point cloud so that ground plane is Y=0
        if self._enable_odometer_noise:
            current_agent_y = self._estimated_position[1]
        else:
            current_agent_y = agent_state.position[1]
        ground_plane_y = current_agent_y - self.occupancy_info["agent_height"]
        XYZ_ego[:, 1] -= ground_plane_y
        # Measure pointcloud without ground-truth pose instead of estimated.
        if self.occupancy_info["measure_noise_free_area"]:
            ground_plane_y = (
                agent_state.position[1] - self.occupancy_info["agent_height"]
            )
            XYZ_ego_nf[:, 1] -= ground_plane_y
        # ================== Compute local occupancy map ======================
        grids_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
        Lx_min = self.occupancy_info["Lx_min"]
        Lz_min = self.occupancy_info["Lz_min"]
        grid_size = self.occupancy_info["map_scale"]
        height_thresh = self.occupancy_info["height_threshold"]
        points = XYZ_ego
        # Compute grid coordinates of points in pointcloud.
        grid_locs = (points[:, [0, 2]] - np.array([[Lx_min, Lz_min]])) / grid_size
        grid_locs = np.floor(grid_locs).astype(int)
        # Classify points in occupancy map as free/occupied/unknown
        # using height-based thresholds on the point-cloud.
        high_filter_idx = points[:, 1] < height_thresh[1]
        low_filter_idx = points[:, 1] > height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)
        # Assign known space as all free initially.
        self.safe_assign(
            grids_mat, grid_locs[high_filter_idx, 0], grid_locs[high_filter_idx, 1], 2,
        )
        kernel = np.ones((3, 3), np.uint8)
        grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)
        # Assign occupied space based on presence of obstacles.
        obs_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
        self.safe_assign(
            obs_mat, grid_locs[obstacle_idx, 0], grid_locs[obstacle_idx, 1], 1
        )
        kernel = np.ones((3, 3), np.uint8)
        obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
        # ================== Update global occupancy map ======================
        visible_mask = grids_mat == 2
        occupied_mask = obs_mat == 1
        np.putmask(out_mat, visible_mask, 2)
        np.putmask(out_mat, occupied_mask, 1)
        # Update counts to each grid location
        seen_mask = (visible_mask | occupied_mask).astype(np.float32)
        count_out_mat += seen_mask
        inv_count_out_mat = np.ma.array(
            1 / np.sqrt(np.clip(count_out_mat, 1.0, math.inf)), mask=1 - seen_mask
        )
        # Pick out counts for locations seen in this frame
        if self.occupancy_info["coverage_novelty_pooling"] == "mean":
            seen_count_reward = inv_count_out_mat.mean().item()
        elif self.occupancy_info["coverage_novelty_pooling"] == "median":
            seen_count_reward = np.ma.median(inv_count_out).item()
        elif self.occupancy_info["coverage_novelty_pooling"] == "max":
            seen_count_reward = inv_count_out_mat.max().item()
        self.occupancy_info["seen_count_reward"] = seen_count_reward
        # If ground-truth navigability is required (and not height-based),
        # obtain the navigability values for valid locations in out_mat from
        # use self._gt_top_down_map.
        if self.occupancy_info["use_gt_occ_map"]:
            gt_visible_mask = visible_mask | occupied_mask
            # Dilate the visible mask
            dkernel = np.ones((9, 9), np.uint8)
            gt_visible_mask = cv2.dilate(
                gt_visible_mask.astype(np.uint8), dkernel, iterations=2
            )
            gt_visible_mask = gt_visible_mask != 0
            gt_occupied_mask = gt_visible_mask & (self._gt_top_down_map == 0)
            np.putmask(gt_out_mat, gt_visible_mask, 2)
            np.putmask(gt_out_mat, gt_occupied_mask, 1)
        # If noise-free measurement for area-seen is required, then compute a
        # global map that uses the ground-truth pose values.
        if self.occupancy_info["measure_noise_free_area"]:
            # --------------- Compute local occupancy map ---------------------
            nf_grids_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
            points_nf = XYZ_ego_nf
            # Compute grid coordinates of points in pointcloud.
            grid_locs_nf = (
                points_nf[:, [0, 2]] - np.array([[Lx_min, Lz_min]])
            ) / grid_size
            grid_locs_nf = np.floor(grid_locs_nf).astype(int)
            # Classify points in occupancy map as free/occupied/unknown
            # using height-based thresholds on the point-cloud.
            high_filter_idx_nf = points_nf[:, 1] < height_thresh[1]
            low_filter_idx_nf = points_nf[:, 1] > height_thresh[0]
            obstacle_idx_nf = np.logical_and(low_filter_idx_nf, high_filter_idx_nf)
            # Assign known space as all free initially.
            self.safe_assign(
                nf_grids_mat,
                grid_locs_nf[high_filter_idx_nf, 0],
                grid_locs_nf[high_filter_idx_nf, 1],
                2,
            )
            kernel = np.ones((3, 3), np.uint8)
            nf_grids_mat = cv2.morphologyEx(nf_grids_mat, cv2.MORPH_CLOSE, kernel)
            # Assign occupied space based on presence of obstacles.
            nf_obs_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
            self.safe_assign(
                nf_obs_mat,
                grid_locs_nf[obstacle_idx_nf, 0],
                grid_locs_nf[obstacle_idx_nf, 1],
                1,
            )
            kernel = np.ones((3, 3), np.uint8)
            nf_obs_mat = cv2.morphologyEx(nf_obs_mat, cv2.MORPH_CLOSE, kernel)
            np.putmask(nf_grids_mat, nf_obs_mat == 1, 1)
            # ---------------- Update global occupancy map --------------------
            visible_mask_nf = nf_grids_mat == 2
            occupied_mask_nf = nf_grids_mat == 1
            np.putmask(noise_free_out_mat, visible_mask_nf, 2)
            np.putmask(noise_free_out_mat, occupied_mask_nf, 1)
        # ================== Measure area seen (m^2) in the map =====================
        if self.occupancy_info["measure_noise_free_area"]:
            seen_area = (
                float(np.count_nonzero(noise_free_out_mat > 0)) * (grid_size) ** 2
            )
        else:
            seen_area = float(np.count_nonzero(out_mat > 0)) * (grid_size) ** 2
        # ================= Compute local depth projection ====================
        if self.occupancy_info["get_proj_loc_map"]:
            proj_out_mat.fill(0)
            # Set everything to unknown initially.
            proj_out_mat[..., 2] = 1
            # Set obstacles.
            np.putmask(proj_out_mat[..., 0], grids_mat == 1, 1)
            np.putmask(proj_out_mat[..., 2], grids_mat == 1, 0)
            # Set free space.
            free_space_mask = (grids_mat != 1) & (grids_mat == 2)
            np.putmask(proj_out_mat[..., 1], free_space_mask, 1)
            np.putmask(proj_out_mat[..., 2], free_space_mask, 0)
        return seen_area

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def get_camera_grid_pos(self) -> Tuple[np.array, np.array]:
        """Returns the agent's current position in both the real world
        (X, Z, theta from -Z to X) and the grid world (Xg, Zg) coordinates.
        """
        if self._enable_odometer_noise:
            position = self._estimated_position
            rotation = self._estimated_rotation
        else:
            agent_state = self.get_agent_state()
            position = agent_state.position
            rotation = agent_state.rotation
        X, Z = position[0], position[2]
        grid_size = self.occupancy_info["map_scale"]
        Lx_min = self.occupancy_info["Lx_min"]
        Lx_max = self.occupancy_info["Lx_max"]
        Lz_min = self.occupancy_info["Lz_min"]
        Lz_max = self.occupancy_info["Lz_max"]
        # Clamp positions within range.
        X = min(max(X, Lx_min), Lx_max)
        Z = min(max(Z, Lz_min), Lz_max)
        # Compute grid world positions.
        Xg = (X - Lx_min) / grid_size
        Zg = (Z - Lz_min) / grid_size
        # Real world rotation.
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(rotation.inverse(), direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        phi = -phi  # (rotation from -Z to X)
        return np.array((X, Z, phi)), np.array((Xg, Zg))

    def get_local_maps(self) -> Tuple[np.array, np.array, np.array]:
        r"""Generates egocentric crops of the global occupancy map.
        Returns:
            The occupancy images display free, occupied and unknown space.
            The color conventions are:
                free-space - (0, 255, 0)
                occupied-space - (0, 0, 255)
                unknown-space - (255, 255, 255)
            The outputs are:
                fine_ego_map_color - (H, W, 3) occupancy image
                coarse_ego_map_color - (H, W, 3) occupancy image
                highres_coarse_ego_map_color - (H, W, 3) occupancy image
        """
        # ================ The global occupancy map ===========================
        if self.occupancy_info["use_gt_occ_map"]:
            top_down_map = self.gt_grids_mat.copy()  # (map_size, map_size)
        else:
            top_down_map = self.grids_mat.copy()  # (map_size, map_size)
        # =========== Obtain local crop around the agent ======================
        # Agent's world and map positions.
        xzt_world, xz_map = self.get_camera_grid_pos()
        xz_map = (int(xz_map[0]), int(xz_map[1]))
        # Crop out only the essential parts of the global map.
        # This saves computation cost for the subsequent operations.
        # *_range - #grid-cells of the map on either sides of the center
        large_map_range = self.occupancy_info["large_map_range"]
        small_map_range = self.occupancy_info["small_map_range"]
        # *_size - output image size
        large_map_size = self.occupancy_info["large_map_size"]
        small_map_size = self.occupancy_info["small_map_size"]
        min_range = int(1.5 * large_map_range)
        x_start = max(0, xz_map[0] - min_range)
        x_end = min(top_down_map.shape[0], xz_map[0] + min_range)
        y_start = max(0, xz_map[1] - min_range)
        y_end = min(top_down_map.shape[1], xz_map[1] + min_range)
        ego_map = top_down_map[x_start:x_end, y_start:y_end]
        # Pad the cropped map to account for out-of-bound indices
        top_pad = max(min_range - xz_map[0], 0)
        left_pad = max(min_range - xz_map[1], 0)
        bottom_pad = max(min_range - top_down_map.shape[0] + xz_map[0] + 1, 0)
        right_pad = max(min_range - top_down_map.shape[1] + xz_map[1] + 1, 0)
        ego_map = np.pad(
            ego_map,
            ((top_pad, bottom_pad), (left_pad, right_pad)),
            "constant",
            constant_values=((0, 0), (0, 0)),
        )
        # The global map is currently addressed as follows:
        # rows are -X to X top to bottom, cols are -Z to Z left to right
        # To get -Z top and X right, we need transpose the map.
        ego_map = ego_map.transpose(1, 0)
        # Rotate the global map to obtain egocentric top-down view.
        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        rot_angle = math.degrees(xzt_world[2])
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,),
        )
        # =========== Obtain final maps at different resolutions ==============
        # Obtain the fine occupancy map.
        start = int(half_size - small_map_range)
        end = int(half_size + small_map_range)
        fine_ego_map = ego_map[start:end, start:end]
        fine_ego_map = cv2.resize(
            fine_ego_map,
            (small_map_size, small_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        fine_ego_map = np.clip(fine_ego_map, 0, 2)
        # Obtain the coarse occupancy map.
        start = int(half_size - large_map_range)
        end = int(half_size + large_map_range)
        coarse_ego_map_orig = ego_map[start:end, start:end]
        coarse_ego_map = cv2.resize(
            coarse_ego_map_orig,
            (large_map_size, large_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        coarse_ego_map = np.clip(coarse_ego_map, 0, 2)
        # Obtain a high-resolution coarse occupancy map.
        # This is primarily useful as an input to an A* path-planner.
        if self.occupancy_info["get_highres_loc_map"]:
            map_size = self.occupancy_info["highres_large_map_size"]
            highres_coarse_ego_map = cv2.resize(
                coarse_ego_map_orig,
                (map_size, map_size),
                interpolation=cv2.INTER_NEAREST,
            )
            highres_coarse_ego_map = np.clip(highres_coarse_ego_map, 0, 2)
        # Convert to RGB maps.
        # Fine occupancy map.
        map_shape = (*fine_ego_map.shape, 3)
        fine_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
        fine_ego_map_color[fine_ego_map == 0] = np.array([255, 255, 255])
        fine_ego_map_color[fine_ego_map == 1] = np.array([0, 0, 255])
        fine_ego_map_color[fine_ego_map == 2] = np.array([0, 255, 0])
        # Coarse occupancy map.
        map_shape = (*coarse_ego_map.shape, 3)
        coarse_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
        coarse_ego_map_color[coarse_ego_map == 0] = np.array([255, 255, 255])
        coarse_ego_map_color[coarse_ego_map == 1] = np.array([0, 0, 255])
        coarse_ego_map_color[coarse_ego_map == 2] = np.array([0, 255, 0])
        # High-resolution coarse occupancy map.
        if self.occupancy_info["get_highres_loc_map"]:
            map_shape = (*highres_coarse_ego_map.shape, 3)
            highres_coarse_ego_map_color = np.zeros(map_shape, dtype=np.uint8)
            highres_coarse_ego_map_color[highres_coarse_ego_map == 0] = np.array(
                [255, 255, 255]
            )
            highres_coarse_ego_map_color[highres_coarse_ego_map == 1] = np.array(
                [0, 0, 255]
            )
            highres_coarse_ego_map_color[highres_coarse_ego_map == 2] = np.array(
                [0, 255, 0]
            )
        else:
            highres_coarse_ego_map_color = None

        return fine_ego_map_color, coarse_ego_map_color, highres_coarse_ego_map_color

    def get_proj_loc_map(self):
        """Generates a fine egocentric projection of depth map.
        Returns:
            The occupancy map is binary and indicates free, occupied and
            unknown spaces. Channel 0 - occupied space, channel 1 - free space,
            channel 2 - unknown space.
            The outputs are:
                fine_ego_map - (H, W, 3) occupancy map
        """
        # ================ The global occupancy map ===========================
        top_down_map = self.proj_grids_mat.copy()  # (map_size, map_size)
        # =========== Obtain local crop around the agent ======================
        # Agent's world and map positions.
        xzt_world, xz_map = self.get_camera_grid_pos()
        xz_map = (int(xz_map[0]), int(xz_map[1]))
        # Crop out only the essential parts of the global map.
        # This saves computation cost for the subsequent opeartions.
        # *_range - #grid-cells of the map on either sides of the center
        small_map_range = self.occupancy_info["small_map_range"]
        # *_size - output image size
        small_map_size = self.occupancy_info["small_map_size"]
        min_range = int(1.5 * small_map_range)
        x_start = max(0, xz_map[0] - min_range)
        x_end = min(top_down_map.shape[0], xz_map[0] + min_range)
        y_start = max(0, xz_map[1] - min_range)
        y_end = min(top_down_map.shape[1], xz_map[1] + min_range)
        ego_map = top_down_map[x_start:x_end, y_start:y_end]
        # Pad the cropped map to account for out-of-bound indices
        top_pad = max(min_range - xz_map[0], 0)
        left_pad = max(min_range - xz_map[1], 0)
        bottom_pad = max(min_range - top_down_map.shape[0] + xz_map[0] + 1, 0)
        right_pad = max(min_range - top_down_map.shape[1] + xz_map[1] + 1, 0)
        ego_map = np.pad(
            ego_map,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            "constant",
            constant_values=0,
        )
        # The global map is currently addressed as follows:
        # rows are -X to X top to bottom, cols are -Z to Z left to right
        # To get -Z top and X right, we need transpose the map.
        ego_map = ego_map.transpose(1, 0, 2)
        # Rotate the global map to obtain egocentric top-down view.
        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        rot_angle = math.degrees(xzt_world[2])
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        # =========== Obtain final maps at different resolutions ==============
        # Obtain the fine occupancy map
        start = int(half_size - small_map_range)
        end = int(half_size + small_map_range)
        fine_ego_map = ego_map[start:end, start:end]
        fine_ego_map = cv2.resize(
            fine_ego_map,
            (small_map_size, small_map_size),
            interpolation=cv2.INTER_NEAREST,
        )
        # Note: There is no conversion to RGB here.
        fine_ego_map = np.clip(fine_ego_map, 0, 1)  # (H, W, 1)

        return fine_ego_map

    def _update_map_observations(self, sim_obs):
        r"""Given the default simulator observations, update it by adding the
        occupancy maps.

        Args:
            sim_obs - a dictionary containing observations from self._sim.
        Returns:
            sim_obs with occupancy maps added as keys to it.
        """
        sensors = self._sensor_suite.sensors
        proc_rgb = sensors["rgb"].get_observation(sim_obs)
        proc_depth = sensors["depth"].get_observation(sim_obs)
        # If the agent went to a new floor, update the GT map
        if self.occupancy_info["use_gt_occ_map"]:
            agent_height = self.get_agent_state().position[1]
            if abs(agent_height - self.gt_map_creation_height) >= 0.5:
                self._gt_top_down_map = self.get_original_map()
                self.gt_map_creation_height = agent_height
        # Update the map with new observations
        seen_area = self.get_seen_area(
            proc_rgb,
            proc_depth,
            self.grids_mat,
            self.count_grids_mat,
            self.gt_grids_mat,
            self.proj_grids_mat,
            self.noise_free_grids_mat,
        )
        inc_area = seen_area - self.occupancy_info["seen_area"]
        # Crop out new egocentric maps
        (
            fine_occupancy,
            coarse_occupancy,
            highres_coarse_occupancy,
        ) = self.get_local_maps()
        # Update stats, observations
        self.occupancy_info["seen_area"] = seen_area
        self.occupancy_info["inc_area"] = inc_area
        sim_obs["coarse_occupancy"] = coarse_occupancy
        sim_obs["fine_occupancy"] = fine_occupancy
        if self.occupancy_info["get_highres_loc_map"]:
            sim_obs["highres_coarse_occupancy"] = highres_coarse_occupancy
        if self.occupancy_info["get_proj_loc_map"]:
            proj_occupancy = self.get_proj_loc_map()
            sim_obs["proj_occupancy"] = proj_occupancy
        return sim_obs
