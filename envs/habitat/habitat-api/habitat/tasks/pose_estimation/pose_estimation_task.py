#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Type

import pdb
import attr
import cv2
import copy
import math
import quaternion
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
)
from habitat.core.utils import not_none_validator
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_rotate_vector,
    compute_heading_from_quaternion,
)
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_to_list,
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat.utils.visualizations.utils import topdown_to_image
from habitat.tasks.nav.nav_task import TopDownMap
from habitat_sim.utils import quat_from_coeffs
from habitat.tasks.pose_estimation.shortest_path_follower import ShortestPathFollower

MAP_THICKNESS_SCALAR: int = 1250
RGBSENSOR_DIMENSION = 3


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
class PoseEstimationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, pose reference details. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        pose_ref_positions: numpy ndarray containing 3 entries for (x, y, z)
            for each pose reference - (nRef, 3)
        pose_ref_rotations: numpy ndarray with 4 entries for (x, y, z, w)
            for each pose reference - (nRef, 4)
    """
    pose_ref_positions: np.array = attr.ib(default=None, validator=not_none_validator)
    pose_ref_rotations: np.array = attr.ib(default=None, validator=not_none_validator)


@registry.register_sensor
class PoseEstimationRGBSensor(Sensor):
    r"""Sensor for PoseEstimation observations.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._nRef = getattr(config, "NREF")
        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_estimation_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self._nRef,
                self.config.HEIGHT,
                self.config.WIDTH,
                RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        # Render the pose references only at the start of each episode.
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            ref_positions = episode.pose_ref_positions
            ref_rotations = episode.pose_ref_rotations
            ref_rgb = []
            for position, rotation in zip(ref_positions, ref_rotations):
                # Get data only from the RGB sensor
                obs = self._sim.get_specific_sensor_observations_at(
                    position, rotation, "rgb"
                )
                # remove alpha channel
                obs = obs[:, :, :RGBSENSOR_DIMENSION]
                ref_rgb.append(obs)
            # Add dummy images to compensate for fewer than nRef references.
            if len(ref_rgb) < self._nRef:
                dummy_image = np.zeros_like(ref_rgb[0])
                for i in range(len(ref_rgb), self._nRef):
                    ref_rgb.append(dummy_image)
            self._pose_ref_rgb = np.stack(ref_rgb, axis=0)
            return np.copy(self._pose_ref_rgb)
        else:
            return None


@registry.register_sensor
class PoseEstimationDepthSensor(Sensor):
    r"""Sensor for PoseEstimation observations.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._nRef = getattr(config, "NREF")

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_estimation_depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self._nRef, self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        # Render the pose references only at the start of each episode.
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            ref_positions = episode.pose_ref_positions
            ref_rotations = episode.pose_ref_rotations

            ref_depth = []
            for position, rotation in zip(ref_positions, ref_rotations):
                # Get data only from the Depth sensor
                obs = self._sim.get_specific_sensor_observations_at(
                    position, rotation, "depth"
                )
                # Process data similar to HabitatSimDepthSensor
                obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
                if self.config.NORMALIZE_DEPTH:
                    # normalize depth observation to [0, 1]
                    obs = (obs - self.config.MIN_DEPTH) / self.config.MAX_DEPTH
                obs = np.expand_dims(obs, axis=2)
                ref_depth.append(obs)
            # Add dummy images to compensate for fewer than nRef references.
            if len(ref_depth) < self._nRef:
                dummy_image = np.zeros_like(ref_depth[0])
                for i in range(len(ref_depth), self._nRef):
                    ref_depth.append(dummy_image)
            self._pose_ref_depth = np.stack(ref_depth, axis=0)
            return np.copy(self._pose_ref_depth)
        else:
            return None


@registry.register_sensor
class PoseEstimationRegressSensor(Sensor):
    r"""Sensor for PoseEstimation observations. Returns the full GT pose
        of references w.r.t agent's starting point. Useful for evaluation
        and computing rewards.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._nRef = getattr(config, "NREF")
        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_estimation_reg"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-1000000.0, high=1000000.0, shape=(self._nRef, 4), dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        # Render the pose references only at the start of each episode.
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            ref_positions = episode.pose_ref_positions
            ref_rotations = episode.pose_ref_rotations
            start_position = episode.start_position
            start_rotation = episode.start_rotation
            start_quat = quat_from_coeffs(start_rotation)
            # Measures the angle from forward to right directions.
            start_heading = compute_heading_from_quaternion(start_quat)
            xs, ys = -start_position[2], start_position[0]
            ref_reg = []
            for position, rotation in zip(ref_positions, ref_rotations):
                rotation = quat_from_coeffs(rotation)
                # Measures the angle from forward to right directions.
                ref_heading = compute_heading_from_quaternion(rotation)
                xr, yr = -position[2], position[0]
                # Compute vector from start to ref assuming start is
                # facing forward @ (0, 0)
                rad_sr = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)
                phi_sr = np.arctan2(yr - ys, xr - xs) - start_heading
                theta_sr = ref_heading - start_heading
                # Normalize theta_sr
                theta_sr = np.arctan2(np.sin(theta_sr), np.cos(theta_sr))
                ref_reg.append((rad_sr, phi_sr, theta_sr, 0.0))
            if len(ref_reg) < self._nRef:
                for i in range(len(ref_reg), self._nRef):
                    ref_reg.append((0.0, 0.0, 0.0, 0.0))
            self._pose_ref_reg = np.array(ref_reg)
            return np.copy(self._pose_ref_reg)
        else:
            return None


@registry.register_sensor
class PoseEstimationMaskSensor(Sensor):
    r"""Sensor for PoseEstimation observations. Returns the mask
    indicating which references are valid.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._nRef = getattr(config, "NREF")
        super().__init__(config=config)

        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pose_estimation_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-1000000.0, high=1000000.0, shape=(self._nRef,), dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            pose_ref_mask = np.ones((self._nRef,))
            pose_ref_mask[len(episode.pose_ref_positions) :] = 0
            self._pose_ref_mask = pose_ref_mask
            return np.copy(self._pose_ref_mask)
        else:
            return None


@registry.register_sensor
class DeltaSensor(Sensor):
    r"""Sensor that returns the odometer readings from the previous action.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

        self.current_episode_id = None
        self.prev_position = None
        self.prev_rotation = None
        self.start_position = None
        self.start_rotation = None
        self._enable_odometer_noise = self._sim._enable_odometer_noise

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "delta_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-1000000.0, high=1000000.0, shape=(4,), dtype=np.float32,)

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self._enable_odometer_noise:
            agent_position = self._sim._estimated_position
            agent_rotation = self._sim._estimated_rotation
        else:
            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation

        if self.current_episode_id != episode_id:
            # A new episode has started
            self.current_episode_id = episode_id
            delta = np.array([0.0, 0.0, 0.0, 0.0])
            self.start_position = copy.deepcopy(agent_position)
            self.start_rotation = copy.deepcopy(agent_rotation)
        else:
            current_position = agent_position
            current_rotation = agent_rotation
            # For the purposes of this sensor, forward is X and rightward is Y.
            # The heading is measured positively from X to Y.
            curr_x, curr_y = -current_position[2], current_position[0]
            curr_heading = compute_heading_from_quaternion(current_rotation)
            prev_x, prev_y = -self.prev_position[2], self.prev_position[0]
            prev_heading = compute_heading_from_quaternion(self.prev_rotation)
            dr = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            dphi = math.atan2(curr_y - prev_y, curr_x - prev_x)
            dhead = curr_heading - prev_heading
            # Convert these to the starting point's coordinate system.
            start_heading = compute_heading_from_quaternion(self.start_rotation)
            dphi = dphi - start_heading
            delta = np.array([dr, dphi, dhead, 0.0])
        self.prev_position = copy.deepcopy(agent_position)
        self.prev_rotation = copy.deepcopy(agent_rotation)

        return delta


@registry.register_sensor
class OracleActionSensor(Sensor):
    r"""Sensor that returns the shortest path action to specific targets.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PoseEstimation sensor.

    Attributes:
        _nRef: number of pose references
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

        self.current_episode_id = None
        self.target_positions = []
        goal_radius = config.GOAL_RADIUS
        self.goal_radius = goal_radius
        self.follower = ShortestPathFollower(sim, goal_radius, False)
        self.follower.mode = "geodesic_path"
        self.oracle_type = config.ORACLE_TYPE
        self.num_targets = config.NUM_TARGETS
        if self.oracle_type == "object":
            self.objects_covered_metric = ObjectsCoveredGeometric(sim, config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32,)

    def _sample_random_targets(self, episode):
        """Samples a random set of navigable points from within the same
        floor to navigate to.
        """
        num_targets = self.num_targets
        self.target_positions = []
        agent_y = self._sim.get_agent_state().position[1]
        for i in range(num_targets):
            point = self._sim.sample_navigable_point()
            point_y = point[1]
            # Sample points within the same floor
            while abs(agent_y - point_y) > 0.5:
                point = self._sim.sample_navigable_point()
                point_y = point[1]
            self.target_positions.append(point)

    def _sample_pose_targets(self, episode):
        """Sets the targets as the set of pose references in the environment
        sampled for this episode. They are assumed to be on the same floor.
        A greedy sampling of the pose references is done where the aim is to
        reduce the distance travelled to cover all of them.
        """
        num_targets = self.num_targets
        self.target_positions = []
        pose_ref_positions = set([tuple(pos) for pos in episode.pose_ref_positions])
        curr_position = episode.start_position
        # Sample the pose references as targets in a greedy fashion.
        while len(pose_ref_positions) > 0:
            min_dist = 1000000000.0
            min_dist_position = None
            for position in pose_ref_positions:
                dist = self._sim.geodesic_distance(curr_position, position)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_position = position
            curr_position = min_dist_position
            # None of the other points are reachable.
            if min_dist_position is None:
                break
            self.target_positions.append(min_dist_position)
            pose_ref_positions.remove(min_dist_position)
        # Add more random targets to fill the quota of num_targets.
        nRefValid = len(self.target_positions)
        agent_y = self._sim.get_agent_state().position[1]
        for i in range(nRefValid, num_targets):
            point = self._sim.sample_navigable_point()
            point_y = point[1]
            # Sample points within the same floor.
            while abs(agent_y - point_y) > 0.5:
                point = self._sim.sample_navigable_point()
                point_y = point[1]
            self.target_positions.append(point)

    def _sample_object_targets(self, episode):
        """Sets the targets as the objects in the environment that can
        be reached and are on the same floor. A greedy sampling of the
        objects is done where the aim is to reduce the distance travelled
        to cover all of them.
        """
        num_targets = self.num_targets
        self.target_positions = []
        self.target_objects = []
        obj_id_pos = set(
            [(obj["id"], tuple(obj["center"])) for obj in self._sim.object_annotations]
        )
        curr_position = episode.start_position
        start_y = curr_position[1]
        # Sample the objects as targets in a greedy fashion.
        while len(obj_id_pos) > 0:
            min_dist = 1000000000.0
            min_dist_id_pos = None
            for curr_id_pos in obj_id_pos:
                curr_pos = curr_id_pos[1]
                dist = self._sim.geodesic_distance(curr_position, curr_pos)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_id_pos = curr_id_pos
            # None of the other points are reachable.
            if min_dist_id_pos is None:
                break
            min_dist_position = min_dist_id_pos[1]
            min_dist_id = min_dist_id_pos[0]
            # Ignore objects that cannot be traversed to or are on a
            # different floor.
            if (
                self._sim.is_navigable(min_dist_position)
                and abs(min_dist_position[1] - start_y) <= 1.0
            ):
                self.target_positions.append(min_dist_position)
                self.target_objects.append(min_dist_id)
                curr_position = min_dist_position
            obj_id_pos.remove(min_dist_id_pos)
        # Add more random targets to fill the quota of num_targets.
        nRefValid = len(self.target_positions)
        for i in range(nRefValid, num_targets):
            point = self._sim.sample_navigable_point()
            while abs(point[1] - start_y) > 0.5:
                point = self._sim.sample_navigable_point()
            self.target_positions.append(point)
            self.target_objects.append(None)

    def _sample_targets(self, episode):
        """Samples targets that the oracle navigates to using shortest-path
        actions. The targets can be random, landmarks, or objects.
        """
        if self.oracle_type == "random":
            self._sample_random_targets(episode)
        elif self.oracle_type == "pose":
            self._sample_pose_targets(episode)
        elif self.oracle_type == "object":
            self._sample_object_targets(episode)

    def get_observation(self, observations, episode):
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self._sample_targets(episode)
        if self.oracle_type == "object":
            # Since the objects are generally dense in MP3D, multiple objects
            # may be covered in the process of visiting one particular
            # object. Make sure the oracle only navigates to objects that
            # were not covered apriori.
            self.objects_covered_metric.update_metric(episode, None)
            current_target = self.target_positions[0]
            current_obj_id = self.target_objects[0]
            agent_position = self._sim.get_agent_state().position
            d2target = self._sim.geodesic_distance(agent_position, current_target,)
            # If the sampled target is already within reach or if it has been
            # covered at some poin during the trajectory,
            # resample a new target.
            while (d2target <= self.goal_radius) or (
                (current_obj_id is not None)
                and current_obj_id in self.objects_covered_metric.seen_objects
            ):
                self.target_positions = self.target_positions[1:]
                self.target_objects = self.target_objects[1:]
                current_target = self.target_positions[0]
                current_obj_id = self.target_objects[0]
                d2target = self._sim.geodesic_distance(agent_position, current_target,)
        else:
            current_target = self.target_positions[0]
            agent_position = self._sim.get_agent_state().position
            d2target = self._sim.geodesic_distance(agent_position, current_target,)
            # If the sampled target is already within reach,
            # resample a new target.
            while d2target <= self.goal_radius:
                self.target_positions = self.target_positions[1:]
                current_target = self.target_positions[0]
                d2target = self._sim.geodesic_distance(agent_position, current_target,)
        oracle_action = self.follower.get_next_action(current_target)
        return np.array([oracle_action], dtype=np.int32)


@registry.register_sensor
class CollisionSensor(Sensor):
    """Returns 1 if a collision occured in the previous action, otherwise
    it returns 0.
    """

    def __init__(self, sim, config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collision_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32,)

    def get_observation(self, observations, episode):
        if self._sim.previous_step_collided:
            return np.array([1.0])
        else:
            return np.array([0.0])


@registry.register_measure
class OPSR(Measure):
    r"""OPSR (Oracle Pose Success Rate)
    Measures if the agent has come close to the ground truth pose.
    Note: Landmarks visited = OPSR * #landmarks.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self._successes = None
        self._geodesic_dist_thresh = config.GEODESIC_DIST_THRESH
        self._angular_dist_thresh = config.ANGULAR_DIST_THRESH
        self.current_episode_id = None
        self._points_of_interest = None
        self._enable_odometer_noise = self._sim._enable_odometer_noise

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "opsr"

    def reset_metric(self, episode):
        self._metric = 0
        nRef = len(episode.pose_ref_positions)
        self._successes = [0.0 for _ in range(nRef)]

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def _compute_points_of_interest(self, episode):
        ref_positions = episode.pose_ref_positions
        ref_rotations = episode.pose_ref_rotations
        points_of_interest = []
        fwd_direction_vector = np.array([0, 0, -1])
        for position, rotation in zip(ref_positions, ref_rotations):
            position = np.array(position)
            rotation = quat_from_coeffs(rotation)
            # Compute depth of the central patch of the pose reference.
            obs = self._sim.get_specific_sensor_observations_at(
                position, rotation, "depth"
            )
            H, W = obs.shape[:2]
            Hby2, Wby2 = H // 2, W // 2
            # Raw depth in meters to the central patch, a.k.a. the
            # point of interest.
            center_depth = np.median(
                obs[(Hby2 - 5) : (Hby2 + 6), (Wby2 - 5) : (Wby2 + 6)]
            )
            # Compute 3D coordinate of the point of interest.
            heading_vector = quaternion_rotate_vector(rotation, fwd_direction_vector)
            point_of_interest = center_depth * heading_vector + position
            points_of_interest.append(point_of_interest)
        self._points_of_interest = np.stack(points_of_interest, axis=0)

    def update_metric(self, episode, action):
        # To measure success rate, the agent's pose needs to close to the
        # landmark's pose, and the agent must look at the same things (points
        # of interest) that the landmark view is looking at, i.e., there must
        # be no occlusions blocking the agent's view of the points of interest.
        episode_id = (episode.episode_id, episode.scene_id)
        if self.current_episode_id != episode_id:
            # Compute the points of interest only at the episode starting.
            self.current_episode_id = episode_id
            self._compute_points_of_interest(episode)
        if self._enable_odometer_noise:
            current_position = self._sim._estimated_position.tolist()
            current_rotation = self._sim._estimated_rotation
        else:
            agent_state = self._sim.get_agent_state()
            current_position = agent_state.position.tolist()
            current_rotation = agent_state.rotation

        # Measure the agent's point of interest.
        current_heading = compute_heading_from_quaternion(current_rotation)
        current_rotation_list = quaternion_to_list(current_rotation)
        curr_depth = self._sim.get_specific_sensor_observations_at(
            current_position, current_rotation_list, "depth",
        )
        H, W = curr_depth.shape[:2]
        Hby2, Wby2 = H // 2, W // 2
        # Raw depth in meters to the central patch, a.k.a. the
        # point of interest.
        curr_center_depth = np.median(
            curr_depth[(Hby2 - 5) : (Hby2 + 6), (Wby2 - 5) : (Wby2 + 6)]
        )
        # Verify that the agent is looking at the same point-of-interest as
        # the pose reference (within some thresholds).
        pose_ref_positions = episode.pose_ref_positions
        pose_ref_rotations = np.array(episode.pose_ref_rotations)
        for i in range(len(pose_ref_positions)):
            # This might fail in the case of noisy odometry when the
            # estimated position is invalid in the graph.
            try:
                distance = self._sim.geodesic_distance(
                    current_position, pose_ref_positions[i]
                )
            except:
                continue
            # Criterion (1): The agent has to be close enough to
            # the actual viewpoint.
            if distance >= self._geodesic_dist_thresh:
                continue
            # Criterion (2): The viewing angle from must be within a threshold
            # of the correct viewing angle to the reference point of interest.
            point_of_interest = self._points_of_interest[i]
            x_poi = -point_of_interest[2]  # -Z is forward --> X
            y_poi = point_of_interest[0]  # X is rightward --> Y
            x_curr = -current_position[2]
            y_curr = current_position[0]
            heading_curr2poi = math.atan2(y_poi - y_curr, x_poi - x_curr)
            diff_angle = heading_curr2poi - current_heading
            diff_angle = math.atan2(math.sin(diff_angle), math.cos(diff_angle))
            diff_angle = math.degrees(abs(diff_angle))
            if diff_angle >= self._angular_dist_thresh:
                continue
            # Criterion (3): The distances of the current point of interest
            # and the reference point of interest from the agent's position
            # must be similar. This accounts for occlusions.
            dist2poi = math.sqrt((x_poi - x_curr) ** 2 + (y_poi - y_curr) ** 2)
            occlusion_err = abs(dist2poi - curr_center_depth)
            if occlusion_err >= 0.5:
                continue
            # If all the criteria were satisfied, this is a successful
            # visitation to the ith reference.
            self._successes[i] = 1

        self._metric = np.sum(self._successes)


@registry.register_measure
class AreaCovered(Measure):
    r"""Area covered metric in grid-cells.  Can only be used in
    the occupancy environment.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "area_covered"

    def reset_metric(self, episode):
        self._metric = 0

    def update_metric(self, episode, action):
        self._metric = self._sim.occupancy_info["seen_area"]


@registry.register_measure
class IncAreaCovered(Measure):
    r"""Increase in area covered (in grid cells) over the past action.
    Can only be used in the occupancy environment.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "inc_area_covered"

    def reset_metric(self, episode):
        self._metric = 0

    def update_metric(self, episode, action):
        self._metric = self._sim.occupancy_info["inc_area"]


@registry.register_measure
class ObjectsCoveredGeometric(Measure):
    r"""
    Number of objects covered (estimated based on geometric knowledge rather than
    rendering semantic images)
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self.current_episode_id = None
        self.seen_objects = set()
        self.seen_categories = set()
        self.IGNORE_CLASSES = [
            "floor",
            "wall",
            "door",
            "misc",
            "ceiling",
            "void",
            "stairs",
            "railing",
            "column",
            "beam",
            "",
            "board_panel",
        ]
        self.intrinsic_matrix = self._sim.occupancy_info["intrinsic_matrix"]
        self.agent_height = self._sim.occupancy_info["agent_height"]
        self.min_depth = float(self._sim.config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.config.DEPTH_SENSOR.MAX_DEPTH)
        self._enable_odometer_noise = self._sim._enable_odometer_noise

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "objects_covered_geometric"

    def reset_metric(self, episode):
        self._metric = {
            "small_objects_visited": 0.0,
            "medium_objects_visited": 0.0,
            "large_objects_visited": 0.0,
            "categories_visited": 0.0,
        }
        self.seen_objects = set()
        self.seen_categories = set()

    def update_metric(self, episode, action):
        episode_id = (episode.episode_id, episode.scene_id)
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self.reset_metric(episode)

        objects = self._sim.object_annotations
        if self._enable_odometer_noise:
            agent_position = self._sim._estimated_position
            agent_rotation = self._sim._estimated_rotation
        else:
            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation
        agent_position_list = agent_position.tolist()
        agent_rotation_list = quaternion_to_list(agent_rotation)
        curr_depth = self._sim.get_specific_sensor_observations_at(
            agent_position_list, agent_rotation_list, "depth"
        )
        curr_depth = np.clip(
            curr_depth, self.min_depth, self.max_depth
        )  # Raw depth values
        curr_depth = curr_depth[..., np.newaxis]
        for obj in objects:
            obj_name = obj["category_name"]
            obj_dims = obj["sizes"]
            obj_id = obj["id"]
            obj_center = obj["center"]
            if obj_id in self.seen_objects:
                continue
            looking_flag, obj_img_pos = self.is_looking_at_object(
                agent_position, agent_rotation, obj_center, obj_name, curr_depth
            )
            if looking_flag:
                size_class = self.get_object_size_class(sorted(obj_dims))
                self.seen_objects.add(obj_id)
                self.seen_categories.add(obj_name)
                if size_class == 0:
                    self._metric["small_objects_visited"] += 1.0
                elif size_class == 1:
                    self._metric["medium_objects_visited"] += 1.0
                else:
                    self._metric["large_objects_visited"] += 1.0

        self._metric["categories_visited"] = float(len(self.seen_categories))

    def convert_to_camera_coords(self, agent_position, agent_rotation, target_position):
        T_cam2world = np.eye(4)
        T_cam2world[:3, :3] = quaternion.as_rotation_matrix(agent_rotation)
        T_cam2world[:3, 3] = agent_position
        target_position = np.concatenate([target_position, np.array([1.0])], axis=0)
        # The agent's position does not account for it's height
        target_position[1] -= self.agent_height

        T_cam2img = self.intrinsic_matrix
        T_world2cam = np.linalg.inv(T_cam2world)
        T_world2img = np.matmul(T_cam2img, T_world2cam)

        target_in_img = np.matmul(T_world2img, target_position)
        # target_in_img = [x * depth, y * depth, -depth, 1]
        # where x is -1 to 1 from left to right and
        # y is 1 to -1 from top to bottom
        target_in_img = target_in_img[:2] / -target_in_img[2]

        # Flip the y to make it go from -1 to 1 from top to bottom (similar to indexing)
        target_in_img[1] = -target_in_img[1]

        # Add 1 and divide by 2 to make x go from 0 to 1 from left to right
        # and y go from 0 to 1 from top to bottom
        target_in_img = (target_in_img + 1) / 2

        return target_in_img

    def is_looking_at_object(
        self, agent_position, agent_rotation, object_position, obj_name, depth
    ):
        # Ignore non-object classes
        if obj_name in self.IGNORE_CLASSES:
            return False, None

        # More than 3 meters away
        dist_to_object = math.sqrt(
            (agent_position[0] - object_position[0]) ** 2
            + (agent_position[2] - object_position[2]) ** 2
        )
        if dist_to_object > 3.0:
            return False, None

        # Not on the same floor
        if abs(agent_position[1] - object_position[1]) > 1.0:
            return False, None

        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(
            agent_rotation.inverse(), direction_vector
        )
        # This is positive from -Z to -X
        heading_angle = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        object_dir_vector = np.array(object_position) - np.array(agent_position)
        # X is assumed to be -Z and Y is assumed to be -X, only then the angle is measured
        # from -Z to -X
        object_dir_angle = cartesian_to_polar(
            -object_dir_vector[2], -object_dir_vector[0]
        )[1]

        diff_angle = heading_angle - object_dir_angle
        diff_angle = abs(math.atan2(math.sin(diff_angle), math.cos(diff_angle)))

        # Close, but not looking at it
        if diff_angle > math.radians(60):
            return False, None

        obj_img_position = self.convert_to_camera_coords(
            agent_position, agent_rotation, object_position,
        )

        # Out of the image range
        if (obj_img_position[0] < 0.0 or obj_img_position[0] >= 1.0) or (
            obj_img_position[1] < 0.0 or obj_img_position[1] >= 1.0
        ):
            return False, None

        HEIGHT, WIDTH = depth.shape[:2]

        obj_img_position[0] = obj_img_position[0] * WIDTH
        obj_img_position[1] = obj_img_position[1] * HEIGHT
        obj_img_position = (int(obj_img_position[0]), int(obj_img_position[1]))

        # Obtain depth of central pixel patch in meters
        obj_depth_patch = depth[
            (obj_img_position[1] - 3) : (obj_img_position[1] + 4),
            (obj_img_position[0] - 3) : (obj_img_position[0] + 4),
            0,
        ]

        # Unable to sample a valid patch
        if obj_depth_patch.shape[0] == 0 or obj_depth_patch.shape[1] == 0:
            return False, None

        obj_depth = obj_depth_patch.mean()

        # Looking at it, but it is blocked
        occlusion_error = abs(
            obj_depth - dist_to_object * math.cos(object_dir_angle - heading_angle)
        )
        if occlusion_error > 0.3:
            return False, None

        return True, obj_img_position

    def get_object_size_class(self, sizes):
        value = (sizes[1] * sizes[2]) ** (1.0 / 2.0)
        if value < 0.5:
            return 0
        elif value < 1.5:
            return 1
        else:
            return 2


@registry.register_measure
class TopDownMapPose(TopDownMap):
    r"""Top Down Map measure for Pose estimation. Displays the start position
    and the pose references.
    """

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map_pose"

    def draw_source_and_references(self, episode):
        # mark source point
        s_x, s_y = maps.to_grid(
            episode.start_position[0],
            episode.start_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        point_padding = 2 * int(np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR))
        self._top_down_map[
            s_x - point_padding : s_x + point_padding + 1,
            s_y - point_padding : s_y + point_padding + 1,
        ] = maps.MAP_SOURCE_POINT_INDICATOR

        # Uncomment these if required.
        # mark reference points
        # nRef = len(episode.pose_ref_positions)
        # for i in range(nRef):
        #    t_x, t_y = maps.to_grid(
        #        episode.pose_ref_positions[i][0],
        #        episode.pose_ref_positions[i][2],
        #        self._coordinate_min,
        #        self._coordinate_max,
        #        self._map_resolution,
        #    )
        #    self._top_down_map[
        #        t_x - point_padding : t_x + point_padding + 1,
        #        t_y - point_padding : t_y + point_padding + 1,
        #    ] = maps.MAP_TARGET_POINT_INDICATOR

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

        # draw source and reference points last to avoid overlap.
        if self._config.DRAW_SOURCE_AND_REFERENCES:
            self.draw_source_and_references(episode)

    def update_metric(self, episode, action):
        super().update_metric(episode, action)
        self._metric = topdown_to_image(self._metric)


@registry.register_measure
class NoveltyReward(Measure):
    r"""Divides the set of valid locations into grid-cells. By conisdering each
    grid-cell as a state, it assigns rewards based on the novelty of
    the states visited.

    novelty_reward(s) = 1/sqrt(N(s))

    where N is the number of visitations to state s.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config
        self.current_episode_id = None
        self._state_map = None
        self.L_min = None
        self.L_max = None
        self._metric = 0.0
        self.grid_size = config.GRID_SIZE
        self._enable_odometer_noise = self._sim._enable_odometer_noise

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "novelty_reward"

    def reset_metric(self, episode):
        self._metric = 0.0
        self.L_min = maps.COORDINATE_MIN
        self.L_max = maps.COORDINATE_MAX
        map_size = int((self.L_max - self.L_min) / self.grid_size) + 1
        self._state_map = np.zeros((map_size, map_size))

    def _convert_to_grid(self, position):
        """position - (x, y, z) in real-world coordinates """
        grid_x = (position[0] - self.L_min) / self.grid_size
        grid_y = (position[2] - self.L_min) / self.grid_size
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        return (grid_x, grid_y)

    def update_metric(self, episode, action):
        episode_id = (episode.episode_id, episode.scene_id)
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self.reset_metric(episode)
        if self._enable_odometer_noise:
            agent_position = self._sim._estimated_position
        else:
            agent_position = self._sim.get_agent_state().position
        grid_x, grid_y = self._convert_to_grid(agent_position)
        self._state_map[grid_y, grid_x] += 1.0
        novelty_reward = 1 / math.sqrt(self._state_map[grid_y, grid_x])
        self._metric = novelty_reward


@registry.register_measure
class CoverageNoveltyReward(Measure):
    r"""
    Assigns rewards based on the novelty of states covered.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "coverage_novelty_reward"

    def reset_metric(self, episode):
        self._metric = 0

    def update_metric(self, episode, action):
        self._metric = self._sim.occupancy_info["seen_count_reward"]


@registry.register_task(name="Pose-v0")
class PoseEstimationTask(EmbodiedTask):
    def __init__(
        self, task_config: Config, sim: Simulator, dataset: Optional[Dataset] = None,
    ) -> None:

        task_measurements = []
        for measurement_name in task_config.MEASUREMENTS:
            measurement_cfg = getattr(task_config, measurement_name)
            measure_type = registry.get_measure(measurement_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                measurement_cfg.TYPE
            )
            task_measurements.append(measure_type(sim, measurement_cfg))
        self.measurements = Measurements(task_measurements)

        task_sensors = []
        for sensor_name in task_config.SENSORS:
            sensor_cfg = getattr(task_config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)
            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            task_sensors.append(sensor_type(sim, sensor_cfg))

        self.sensor_suite = SensorSuite(task_sensors)
        super().__init__(config=task_config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Type[Episode]) -> Any:
        return merge_sim_episode_config(sim_config, episode)
