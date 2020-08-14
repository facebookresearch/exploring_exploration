#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import gym
import math
import copy
import json
import gzip
import pickle
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from gym import error, spaces, utils
from gym.utils import seeding
from gym_avd.envs.config import *
from gym_avd.envs.utils import *
from gym_avd.envs.avd_occ_base_env import AVDOccBaseEnv

from gym.envs.registration import register


class AVDPoseBaseEnv(AVDOccBaseEnv):
    r"""A base environment for pose estimation where the pose estimation targets /
    references are sampled randomly in the scene. Provides the basic functionality
    for sampling more complex targets (such as landmarks).
    """
    metadata = {"render.modes": ["human", "rgb"]}

    def __init__(
        self,
        WIDTH: int = 84,
        HEIGHT: int = 84,
        max_steps: Optional[int] = None,
        nRef: int = 1,
        map_scale: float = 50.0,
        map_size: int = 301,
        max_depth: float = 3000,
        target2oracle: bool = False,
        dist_thresh: float = 500.0,
    ):
        self.target2oracle = target2oracle
        self.dist_thresh = dist_thresh
        super().__init__(
            WIDTH=WIDTH,
            HEIGHT=HEIGHT,
            max_steps=max_steps,
            nRef=nRef,
            map_scale=map_scale,
            map_size=map_size,
            max_depth=max_depth,
        )

    def _define_observation_space(self):
        H = int(self.HEIGHT)
        W = int(self.WIDTH)
        LOW_F = -100000.0
        HIGH_F = 100000.0
        self.observation_space = spaces.Dict(
            {
                "im": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8,),
                "depth": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(H, W, 1), dtype=np.float32,
                ),
                "coarse_occupancy": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(H, W, 3), dtype=np.float32,
                ),
                "fine_occupancy": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(H, W, 3), dtype=np.float32,
                ),
                "highres_coarse_occupancy": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(200, 200, 3), dtype=np.float32,
                ),
                "delta": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(4,), dtype=np.float32,
                ),
                "oracle_action": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32,),
                "collisions": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32,),
                "pose_refs": spaces.Box(
                    low=LOW_F,
                    high=HIGH_F,
                    shape=(self.nRef, H, W, 3),
                    dtype=np.float32,
                ),
                "pose_refs_depth": spaces.Box(
                    low=LOW_F,
                    high=HIGH_F,
                    shape=(self.nRef, H, W, 1),
                    dtype=np.float32,
                ),
                "pose_regress": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(self.nRef, 4), dtype=np.float32,
                ),
                "valid_masks": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(self.nRef,), dtype=np.float32,
                ),
            }
        )

    def _initialize_environment_variables(self):
        r"""Additionally define pose estimation, object visitation related
        details.
        """
        super()._initialize_environment_variables()
        # Define pose estimation related details.
        self.oracle_pose_successes = np.zeros((self.nRef,))
        self.plot_references_in_topdown = True
        # Define object visitation related details.
        self.bbox_size_thresh = (0.1 * self.WIDTH) * (0.1 * self.HEIGHT)
        self.bbox_dist_thresh = 1500.0  # mm
        if os.path.isfile(OBJ_COUNTS_FILE):
            self.object_counts_per_env = json.load(open(OBJ_COUNTS_FILE, "r"))
        if os.path.isfile(OBJ_PROPS_FILE):
            self.object_props_per_env = pickle.load(open(OBJ_PROPS_FILE, "rb"))
        if os.path.isfile(SIZE_CLASSIFICATION_PATH):
            with gzip.open(SIZE_CLASSIFICATION_PATH, "rt") as fp:
                self.size_classification = json.load(fp)

    def reset(self, scene_idx: Optional[int] = None):
        # Spawn agent in a new scene at a random image.
        if scene_idx is None:
            scenes = getattr(self, "{}_scenes".format(self.split))
            self.scene_idx = self._rng.choice(scenes)
        else:
            self.scene_idx = scene_idx
        self.scene_id = self.all_scenes[self.scene_idx]
        scene_conn = self.data_conn[self.scene_idx]
        self.images_to_idx = scene_conn["images_to_idx"]
        self.images_to_camera = scene_conn["images_to_camera"]
        self.images_to_nodes = scene_conn["images_to_nodes"]
        self.scale = scene_conn["scale"]  # Converts positions to mm.
        self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
        while not self.is_valid_image(self.agent_image):
            self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
        # Initialize the environment variables.
        self.start_position = self._get_position(self.agent_image)
        self.start_pose = self._get_pose(self.agent_image)
        self.agent_pose = self.start_pose
        self.agent_position = copy.deepcopy(self.start_position)
        self.scene_images = np.array(self.data_h5[f"{self.scene_id}/rgb"])
        self.scene_depth = np.array(self.data_h5[f"{self.scene_id}/depth"])
        self.delta = (0.0, 0.0, 0.0, 0.0)
        self.collision_occurred = False
        self.steps = 0
        self.graph = create_nav_graph(self.data_conn[self.scene_idx])
        self._load_nav_graph()
        # Get scene annotations.
        annotations_path = f"{ROOT_DIR}/{self.scene_id}/annotations.json"
        self.annotations = json.load(open(annotations_path, "r"))
        self.visited_instances = set([])
        instances_path = (
            f"{VALID_INSTANCES_ROOT_DIR}/{self.scene_id}/instances.json.gzip"
        )
        with gzip.open(instances_path, "rt") as fp:
            self.valid_instances = json.load(fp)
        # Sample pose reference.
        self._sample_pose_refs()
        # Sample targets to navigate to (for oracle action).
        self._sample_oracle_targets()
        self._stuck_at_oracle = False
        # Path taken so far.
        self.path_so_far = []
        self.top_down_env = None
        # Count-based rewards.
        self._setup_counts()
        # Create occupancy maps related variables.
        min_x, min_y, max_x, max_y = self.get_environment_extents()
        self.L_min = min(min_x, min_y) - 4000
        self.L_max = max(max_x, max_y) + 4000
        self.grid_size = self.map_scale
        grid_num = np.array([self.map_size, self.map_size])
        self.grids_mat.fill(0)
        self.count_grids_mat.fill(0)
        self.max_grid_size = np.max(grid_num)
        self.max_seen_area = float(np.prod(grid_num))
        self.seen_area = 0
        self.seen_count_reward = 0.0

        return self._get_obs()

    def step(self, action: int):
        old_position = self._get_position(self.agent_image)
        old_pose = self._get_pose(self.agent_image)
        # Execute action.
        next_image = self._action_to_img(action)
        # Replace the agent_image only if the action was valid.
        if next_image != -1:
            self.agent_image = next_image
        new_position = self._get_position(self.agent_image)
        new_pose = self._get_pose(self.agent_image)
        self.agent_position = new_position
        self.agent_pose = new_pose
        # Compute delta in motion.
        self.delta = self.compute_delta(old_position, old_pose, new_position, new_pose,)
        dr = self.delta[0]
        if action == 2 and dr < 100.0:  # (100 mm distance)
            self.collision_occurred = True
        else:
            self.collision_occurred = False
        self.steps += 1

        self.object_visitation_check()
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._episode_over()
        info = self._get_info()

        return obs, reward, done, info

    def _sample_pose_refs(self):
        r"""Sample random views that are farther than dist_thresh from the
        starting view as the pose references.
        """
        # Sample random images from the set of images in the environment.
        images = list(self.images_to_idx.keys())
        self._pose_image_names = [self._rng.choice(images) for i in range(self.nRef)]
        self._pose_refs = []
        self._pose_refs_depth = []
        self.ref_positions = []
        self.ref_poses = []
        self._pose_regress = []
        # For each sampled reference image, ensure that they satisfy the
        # distance threshold ``dist_thresh''.
        for i in range(self.nRef):
            d2t = 0.0
            for iter_count in range(100):
                ref_position = self._get_position(self._pose_image_names[i])
                ref_pose = self._get_pose(self._pose_image_names[i])
                dx = ref_position[0] - self.start_position[0]
                dz = ref_position[2] - self.start_position[2]
                d2t = math.sqrt(dx ** 2 + dz ** 2)
                # If the condtion holds, exit the loop.
                if d2t >= self.dist_thresh:
                    break
                # If the condition does not hold, resample a new reference.
                self._pose_image_names[i] = self._rng.choice(images)
            # Compute data for the pose references.
            ref_position = self._get_position(self._pose_image_names[i])
            ref_pose = self._get_pose(self._pose_image_names[i])
            pose_idx = self.images_to_idx[self._pose_image_names[i]]
            pose_ref = self.scene_images[pose_idx]
            pose_ref_depth = self._process_depth(self.scene_depth[pose_idx])
            pose_ref = pose_ref[np.newaxis, :, :, :]
            pose_ref_depth = pose_ref_depth[np.newaxis, :, :, :]
            # Compute reference pose relative to agent's starting pose.
            dx = ref_position[0] - self.start_position[0]
            dz = ref_position[2] - self.start_position[2]
            dr = math.sqrt(dx ** 2 + dz ** 2)
            dtheta = math.atan2(dz, dx) - self.start_pose
            dhead = ref_pose - self.start_pose
            delev = 0.0
            pose_regress = (dr, dtheta, dhead, delev)
            # Update the set of pose references.
            self._pose_refs.append(pose_ref)
            self._pose_refs_depth.append(pose_ref_depth)
            self.ref_positions.append(ref_position)
            self.ref_poses.append(ref_pose)
            self._pose_regress.append(pose_regress)

        self._pose_refs = np.concatenate(self._pose_refs, axis=0)
        self._pose_refs_depth = np.concatenate(self._pose_refs_depth, axis=0)
        self.ref_positions = np.array(self.ref_positions)
        self.ref_poses = np.array(self.ref_poses)
        self._pose_regress = np.array(self._pose_regress)
        self.oracle_pose_successes = np.zeros((self.nRef,))
        self._valid_masks = np.ones((self.nRef,))

    def set_nref(self, nRef):
        self.nRef = nRef
        # Re-define the observation space.
        self._define_observation_space()

    def set_dist_thresh(self, thresh):
        self.dist_thresh = thresh

    def _sample_oracle_targets(self):
        r"""Sample targets for oracle agent to navigate to.
        """
        if self.target2oracle and self._valid_masks[0] == 1:
            # Add only the first pose reference to the oracle targets.
            self._random_targets = [self._pose_image_names[0]]
            self._random_targets += self._rng.choices(
                list(self.images_to_idx.keys()), k=self.max_steps,
            )
        else:
            self._random_targets = self._rng.choices(
                list(self.images_to_idx.keys()), k=self.max_steps,
            )
        # Remove images whose nodes are unreachable.
        self._random_targets = list(filter(self.is_valid_image, self._random_targets))
        self._random_targets_ix = 0

    def _get_obs(self):
        r"""Compute the observations.
        """
        # Get the observations from the parent class.
        observations = super()._get_obs()
        # Add the pose observations to the above. These are returned
        # only at the very first step as these are kept constant
        # throughout the episode.
        if self.steps == 0:
            pose_refs = np.copy(self._pose_refs)
            pose_refs_depth = np.copy(self._pose_refs_depth)
            pose_regress = np.copy(self._pose_regress)
            valid_masks = np.copy(self._valid_masks)
            observations["pose_refs"] = pose_refs
            observations["pose_refs_depth"] = pose_refs_depth
            observations["pose_regress"] = pose_regress
            observations["valid_masks"] = valid_masks

        return observations

    def get_ref_positions(self):
        return np.copy(self.ref_positions)

    def get_ref_poses(self):
        return np.copy(self.ref_poses)

    def object_visitation_check(self):
        r"""Updates the set of visited objects based on the current
        observation.
        """
        bbs = self.annotations[self.agent_image]["bounding_boxes"]
        curr_depth = self._get_depth(self.agent_image)
        # For each instance with a bounding box in the current image,
        # check if they satisfy certain conditions.
        for bb in bbs:
            xmin, ymin, xmax, ymax = bb[:4]
            xmin = int(xmin * self.WIDTH / 1920.0)
            xmax = int(xmax * self.WIDTH / 1920.0)
            ymin = int(ymin * self.HEIGHT / 1080.0)
            ymax = int(ymax * self.HEIGHT / 1080.0)
            # Is the bounding box valid?
            if xmax > xmin and ymax > ymin:
                bbox_depth = curr_depth[ymin:ymax, xmin:xmax]
                bbox_depth_valid = bbox_depth[bbox_depth > 0.0]
                # Are depth values available for all locations in the
                # bounding box?
                if bbox_depth_valid.shape[0] > 0:
                    bbox_depth = bbox_depth_valid.mean()
                    bbarea = (xmax - xmin) * (ymax - ymin)
                    bbinstance = bb[4]
                    # Is the bounding box area greater than the threshold?
                    # (or) Is it close enough to the agent?
                    if (
                        bbarea > self.bbox_size_thresh
                        or bbox_depth < self.bbox_dist_thresh
                    ):
                        self.visited_instances.add(bbinstance)

    def pose_reference_visitation_check(self):
        r"""Updates the set of visited references based on the current
        observation.
        """
        curr_img = self.agent_image
        curr_nodeix = self.images_to_nodes[curr_img]
        for i in range(self.nRef):
            ref_img = self._pose_image_names[i]
            # Ignore invalid references.
            if self._valid_masks[i] == 0:
                continue
            # Ignore already visited references.
            if self.oracle_pose_successes[i] == 1.0:
                continue
            ref_nodeix = self.images_to_nodes[ref_img]
            # Ignore unreachable nodes in the graph.
            if ref_nodeix not in self.paths[curr_nodeix]:
                continue
            path = self.paths[curr_nodeix][ref_nodeix]
            # Condition (1): Is the reference within 3 nodes of the current
            # node? Amounts to a geodesic distance of ~1.0m.
            if len(path) < 3:
                ref_pos = self._get_position(ref_img)
                ref_pose = self._get_pose(ref_img)
                ref_depth = self._get_depth(ref_img)
                curr_pos = self._get_position(curr_img)
                curr_pose = self._get_pose(curr_img)
                midx, midy = ref_depth.shape[1] // 2, ref_depth.shape[0] // 2
                d_mid = ref_depth[
                    (midy - 5) : (midy + 5), (midx - 5) : (midx + 5)
                ].mean()
                # Position of the point of interest.
                tgt_pos = (
                    ref_pos[0] + d_mid * math.cos(ref_pose),
                    ref_pos[2] + d_mid * math.sin(ref_pose),
                )
                # Required heading angle to look at the point of interest from
                # the current position.
                reqd_heading = math.atan2(
                    tgt_pos[1] - curr_pos[2], tgt_pos[0] - curr_pos[0]
                )
                # The difference from the current pose.
                diff_angle = reqd_heading - curr_pose
                diff_angle = abs(
                    math.atan2(math.sin(diff_angle), math.cos(diff_angle),)
                )
                # Condition (2): Is the current pose close enough to the
                # required pose?
                if diff_angle < math.radians(30):
                    self.oracle_pose_successes[i] = 1.0

    def _get_info(self):
        info = super()._get_info()
        # Update the set of visited poses.
        self.pose_reference_visitation_check()
        info["oracle_pose_success"] = self.oracle_pose_successes.sum().item()
        # Update the object counts.
        info["num_objects_visited"] = len(self.visited_instances)
        info["num_small_objects_visited"] = 0
        info["num_medium_objects_visited"] = 0
        info["num_large_objects_visited"] = 0
        for obj_id in self.visited_instances:
            size_class = self.size_classification[self.scene_id][str(obj_id)]
            if size_class == 0:
                info["num_small_objects_visited"] += 1.0
            elif size_class == 1:
                info["num_medium_objects_visited"] += 1.0
            else:
                info["num_large_objects_visited"] += 1.0
        if hasattr(self, "object_counts_per_env"):
            num_objs_visited = 1.0 * len(self.visited_instances)
            num_objs_total = self.object_counts_per_env[self.scene_id]
            info["frac_objects_visited"] = num_objs_visited / num_objs_total
        if hasattr(self, "object_props_per_env"):
            info["num_object_props_in_image"] = self.object_props_per_env[
                self.scene_id
            ][self.agent_image].shape[0]
        return info

    def generate_topdown_occupancy(self) -> np.array:
        r"""Generates the top-down occupancy map of the environment.
        """
        # Obtain the top-down images from the original environment.
        grid = super().generate_topdown_occupancy()
        # Draw the set of pose references.
        if self.plot_references_in_topdown:
            min_x, min_z, max_x, max_z = self.get_environment_extents()
            grid_size = 20.0
            env_size = max(max_z - min_z, max_x - min_x, 8000.0)
            x_pad = (env_size - (max_x - min_x)) // 2
            z_pad = (env_size - (max_z - min_z)) // 2
            min_x = min_x - x_pad
            min_z = min_z - z_pad
            max_x = max_x + x_pad
            max_z = max_z + z_pad
            radius = max(grid.shape[0] // 50, 1)
            for pose_img in self._pose_image_names:
                if pose_img == "":
                    continue
                curr_pos = self._get_position(pose_img)
                curr_pos = np.array([curr_pos[0], curr_pos[2]])
                curr_pos = (curr_pos - np.array([min_x, min_z])) / grid_size
                curr_theta = self._get_pose(pose_img)
                grid = draw_agent(grid, curr_pos, curr_theta, (255, 0, 0), size=radius,)

        return grid


class AVDPoseLandmarksEnv(AVDPoseBaseEnv):
    r"""Samples landmarks as pose estimation targets / references.
    This also generates oracle_actions corresponding to an oracle agent that
    visits randomly sampled locations in the environment using the shortest paths.
    """

    def _initialize_environment_variables(self):
        r"""Additionally define pose sampling related information.
        """
        super()._initialize_environment_variables()
        self.cluster_root_dir = CLUSTER_ROOT_DIR
        self.ref_sample_intervals = None

    def set_ref_intervals(self, intervals: Optional[List[float]]):
        self.ref_sample_intervals = intervals

    def _sample_pose_refs(self):
        r"""Sample views from landmark clusters as the pose references.
        """
        # Load landmark clusters information.
        cluster_root = self.cluster_root_dir
        json_path = f"{cluster_root}/{self.scene_id}/reference_clusters.json"
        self.cluster_info = json.load(open(json_path, "r"))
        # Restrict to clusters with variance in position less than 1.5m>
        filtered_clusters = filter(lambda x: x["var"] < 1.5, self.cluster_info)
        filtered_clusters = list(filtered_clusters)
        num_clusters = len(filtered_clusters)
        # Sort clusters by proximity to starting position.
        start_position = np.array(self.start_position)
        unsorted_clusters = [
            (i, np.array(cluster["cluster_position"]) * 1000.0)  # m -> mm
            for i, cluster in enumerate(filtered_clusters)
        ]
        sorted_clusters = sorted(
            unsorted_clusters, key=lambda c: np.linalg.norm(start_position - c[1]),
        )
        self._pose_image_names = []
        self._pose_refs = []
        self._pose_refs_depth = []
        self.ref_positions = []
        self.ref_poses = []
        self._pose_regress = []
        # Sample cluster indices.
        if self.ref_sample_intervals is None:
            # If no sampling intervals are specified, sample all clusters
            # uniformly.
            sample_idxes = np.linspace(0, num_clusters - 1, self.nRef)
            sample_idxes = sample_idxes.astype(int).tolist()
        else:
            # The sampling intervals specify at what distance intervals from
            # the starting position to sample from. If the sampling intervals
            # are specified, divide the total number of references equally
            # between the intervals.
            sample_idxes = []
            intervals = self.ref_sample_intervals
            intervals = [0] + intervals + [math.inf]
            samples_per_interval = min(self.nRef // (len(intervals) - 1), 1)
            for i in range(len(intervals) - 1):
                # For each interval, identify the set of clusters whose
                # centroid lies within this distance interval.
                valid_clusters = []
                for j, c in enumerate(sorted_clusters):
                    dist = np.linalg.norm(start_position - c[1])
                    if (dist >= intervals[i]) and (dist < intervals[i + 1]):
                        valid_clusters.append(j)

                if i != len(intervals) - 2:
                    nsamples = samples_per_interval
                else:
                    nsamples = self.nRef - samples_per_interval * (len(intervals) - 2)

                if len(valid_clusters) > 0:
                    # If there are valid clusters, sample from them (with
                    # replacement).
                    sample_idxes += self._rng.choices(valid_clusters, k=nsamples,)
                else:
                    # If there are no valid clusters, set as invalid clusters.
                    sample_idxes += [None for _ in range(nsamples)]
        # Sample references from the sampled cluster indices.
        valid_masks = []
        for i in sample_idxes:
            if i is None:
                # If the reference is invalid, enter dummy data.
                valid_masks.append(0)
                self._pose_image_names.append("")
                self._pose_refs.append(
                    np.zeros((1, self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                )
                pose_ref_depth = self._process_depth(
                    np.zeros((self.HEIGHT, self.WIDTH, 1), dtype=np.uint8)
                )
                pose_ref_depth = pose_ref_depth[np.newaxis, :, :, :]
                self._pose_refs_depth.append(pose_ref_depth)
                self.ref_positions.append([0, 0, 0])
                self.ref_poses.append(0)
                self._pose_regress.append((0, 0, 0, 0))
                continue
            # Randomly pick an image from the current cluster.
            cluster_idx = sorted_clusters[i][0]
            pose_image = self._rng.choice(filtered_clusters[cluster_idx]["images"])
            # Compute data for the pose references.
            ref_position = self._get_position(pose_image)
            ref_pose = self._get_pose(pose_image)
            pose_idx = self.images_to_idx[pose_image]
            pose_ref = self.scene_images[pose_idx]
            pose_ref_depth = self._process_depth(self.scene_depth[pose_idx])
            pose_ref = pose_ref[np.newaxis, :, :, :]
            pose_ref_depth = pose_ref_depth[np.newaxis, :, :, :]
            # Compute reference pose relative to agent's starting pose.
            dx = ref_position[0] - self.start_position[0]
            dz = ref_position[2] - self.start_position[2]
            dr = math.sqrt(dx ** 2 + dz ** 2)
            dtheta = math.atan2(dz, dx) - self.start_pose
            dhead = ref_pose - self.start_pose
            delev = 0.0
            pose_regress = (dr, dtheta, dhead, delev)
            # Update the set of pose references.
            valid_masks.append(1)
            self._pose_image_names.append(pose_image)
            self._pose_refs.append(pose_ref)
            self._pose_refs_depth.append(pose_ref_depth)
            self.ref_positions.append(ref_position)
            self.ref_poses.append(ref_pose)
            self._pose_regress.append(pose_regress)

        self._pose_refs = np.concatenate(self._pose_refs, axis=0)
        self._pose_refs_depth = np.concatenate(self._pose_refs_depth, axis=0)
        self.ref_positions = np.array(self.ref_positions)
        self.ref_poses = np.array(self.ref_poses)
        self._pose_regress = np.array(self._pose_regress)
        self.oracle_pose_successes = np.zeros((self.nRef,))
        self._valid_masks = np.array(valid_masks)


class AVDPoseLandmarksVisitLandmarksEnv(AVDPoseLandmarksEnv):
    r"""Samples landmarks as pose estimation targets / references.
    This also generates oracle_actions corresponding to an oracle agent that
    visits randomly the sampled landmarks using the shortest paths.
    """

    def _sample_oracle_targets(self):
        r"""Sample landmarks as targets for the oracle agent to navigate to.
        """
        self._random_targets = self.get_route_visiting_refs()
        self._random_targets += self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps,
        )
        # Remove images whose nodes are unreachable
        self._random_targets = list(filter(self.is_valid_image, self._random_targets,))
        self._random_targets_ix = 0

    def _base_shortest_path_action(self, ref_image):
        """
        Returns the next action on the shortest path to a reference
        image with a particular reference pose. Modified to to avoid
        orienting the agent with the target pose after reaching the
        target position.
        """
        agent_nodeix = self.images_to_nodes[self.agent_image]
        ref_nodeix = self.images_to_nodes[ref_image]
        ref_pose = self._get_pose(ref_image)
        # If the agent has reached the target node, align with the reference
        # pose
        if agent_nodeix == ref_nodeix:
            return 3
        # If agent has not reached the target node, then sample an action
        # along the shortest paths to the target.
        path = self.paths[agent_nodeix][ref_nodeix]
        next_nodeix = path[1]  # NOTE: path[0] will be agent_nodeix itself.
        scene_conn = self.data_conn[self.scene_idx]
        next_node_pos = scene_conn["nodes"][next_nodeix]["world_pos"]
        next_node_pos = np.array(next_node_pos) * self.scale
        agent_node_pos = scene_conn["nodes"][agent_nodeix]["world_pos"]
        agent_node_pos = np.array(agent_node_pos) * self.scale
        # Measure the required agent pose in order to move forward from
        # current node to next node.
        # Swap X, Z.
        dX = next_node_pos[2] - agent_node_pos[2]
        dZ = next_node_pos[0] - agent_node_pos[0]
        required_pose = math.atan2(dZ, dX)
        # Is it possible to just move forward?
        if abs(norm_angle(self.agent_pose - required_pose)) < math.radians(20.0):
            # Will the forward action get stuck? If so, just terminate this
            # particular target.
            if self._action_to_img(2) == self.agent_image:
                return 3
            else:
                return 2
        elif norm_angle(required_pose - self.agent_pose) < 0.0:
            # Turn left
            return 0
        else:
            # Turn right
            return 1

    def get_route_visiting_refs(self):
        r"""Get a route visiting the references in a greedy fashion.
        """
        # Filter invalid pose images.
        pose_image_names = []
        for i in range(self.nRef):
            img = self._pose_image_names[i]
            if self._valid_masks[i] == 1.0 and self.is_valid_image(img):
                pose_image_names.append(img)
        # Get only unique images in case of repeated content.
        images = set(pose_image_names)
        # Sample a route visiting the valid pose references.
        start_image = self.agent_image
        route = []
        while len(images) > 0:
            min_path_length = math.inf
            min_path_image = None
            start_node = self.images_to_nodes[start_image]
            # Find the next reference that is closest in terms of
            # geodesic distance.
            for image in images:
                node = self.images_to_nodes[image]
                try:
                    spl = len(self.paths[start_node][node])
                except KeyError:
                    continue
                if spl < min_path_length:
                    min_path_length = spl
                    min_path_image = image
            assert min_path_image is not None
            start_image = min_path_image
            route.append(min_path_image)
            images.remove(min_path_image)

        return route


class AVDPoseLandmarksVisitObjectsEnv(AVDPoseLandmarksEnv):
    r"""Samples landmarks as pose estimation targets / references.
    This also generates oracle_actions corresponding to an oracle agent that
    visits randomly objects using the shortest paths.
    """

    def _sample_oracle_targets(self):
        r"""Sample targets for the oracle agent to navigate to.
        """
        # Sample objects as oracle targets.
        self._random_targets = self.get_route_visiting_objs()
        self._random_targets += self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps
        )
        # Remove images whose nodes are unreachable.
        self._random_targets = list(filter(self.is_valid_image, self._random_targets))
        self._random_targets_ix = 0

    def get_route_visiting_objs(self):
        r"""Get a route visiting the references in a greedy fashion.
        """
        # Sample random object instances from the set of valid instances.
        sampled_ids = self.valid_instances.keys()
        pose_image_names = []
        for inst_id in sampled_ids:
            img = self._rng.choice(self.valid_instances[inst_id])
            if self.is_valid_image(img):
                pose_image_names.append(img)
        # Get only unique images in case of repeated content.
        images = set(pose_image_names)
        # Sample a route visiting the valid pose references.
        start_image = self.agent_image
        route = []
        while len(images) > 0:
            min_path_length = math.inf
            min_path_image = None
            start_node = self.images_to_nodes[start_image]
            # Find the next reference that is closest in terms of
            # geodesic distance.
            for image in images:
                node = self.images_to_nodes[image]
                try:
                    spl = len(self.paths[start_node][node])
                except KeyError:
                    continue

                if spl < min_path_length:
                    min_path_length = spl
                    min_path_image = image
            assert min_path_image is not None
            start_image = min_path_image
            route.append(min_path_image)
            images.remove(min_path_image)

        return route


register(
    id="avd-pose-v0", entry_point="gym_avd.envs:AVDPoseLandmarksEnv",
)

register(
    id="avd-pose-random-oracle-v0", entry_point="gym_avd.envs:AVDPoseLandmarksEnv",
)

register(
    id="avd-pose-landmarks-oracle-v0",
    entry_point="gym_avd.envs:AVDPoseLandmarksVisitLandmarksEnv",
)

register(
    id="avd-pose-objects-oracle-v0",
    entry_point="gym_avd.envs:AVDPoseLandmarksVisitObjectsEnv",
)
