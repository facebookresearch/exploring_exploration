#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import math
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from gym import error, spaces, utils
from gym.utils import seeding
from gym_avd.envs.config import *
from gym_avd.envs.utils import *
from gym.envs.registration import register

from gym_avd.envs.avd_pose_env import AVDPoseBaseEnv


class AVDReconEnv(AVDPoseBaseEnv):
    r"""Implements an environment for the reconstruction task. It builds on top of
    the AVDPoseBaseEnv and samples reconstruction targets as locations on a uniform
    grid in the environment.
    """

    def __init__(self, nRef: int = 50):
        super().__init__(nRef=nRef)

    def _initialize_environment_variables(self):
        r"""Additionally define reconstruction reference sampling details.
        """
        super()._initialize_environment_variables()
        self.cluster_root_dir = CLUSTER_ROOT_DIR
        self.ref_sample_intervals = None

    def _sample_pose_refs(self):
        r"""Sample views from a uniform grid locations.
        """
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        all_nodes = self.data_conn[self.scene_idx]["nodes"]
        all_node_idxes = list(range(len(all_nodes)))
        all_nodes_positions = [
            [node["world_pos"][2], node["world_pos"][0]] for node in all_nodes
        ]
        all_nodes_positions = np.array(all_nodes_positions) * self.scale
        # Sample nodes uniformly @ 1.5m distance from the environment.
        range_x = np.arange(min_x, max_x, 1500.0)
        range_z = np.arange(min_z, max_z, 1500.0)
        relevant_node_idxes = set()
        relevant_nodes = []
        for x in range_x:
            for z in range_z:
                # Find closest node to this coordinate.
                zipped_data = zip(all_nodes, all_node_idxes, all_nodes_positions,)
                min_dist = math.inf
                min_dist_node = None
                min_dist_node_idx = None
                for node, node_idx, node_position in zipped_data:
                    nx, nz = node_position[0], node_position[1]
                    d = np.sqrt((x - nx) ** 2 + (z - nz) ** 2).item()
                    if d < min_dist:
                        min_dist = d
                        min_dist_node = node
                        min_dist_node_idx = node_idx
                if min_dist_node_idx not in relevant_node_idxes:
                    relevant_nodes.append(min_dist_node)
                    relevant_node_idxes.add(min_dist_node_idx)
        # Sample the reference images from the nodes.
        relevant_images = []
        for node in relevant_nodes:
            for j in range(0, 12, 3):
                image_name = node["views"][j]["image_name"]
                relevant_images.append(image_name)
        self._pose_image_names = []
        self._pose_refs = []
        self._pose_refs_depth = []
        self.ref_positions = []
        self.ref_poses = []
        self._pose_regress = []
        for count, pose_image in enumerate(relevant_images):
            # Limit to self.nRef images.
            if count >= self.nRef:
                break
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
        self._valid_masks = np.ones((self._pose_refs.shape[0],))
        # Pad the data with dummy data to account for missing references.
        if self._pose_refs.shape[0] < self.nRef:
            padding = self.nRef - self._pose_refs.shape[0]
            dummy_pose_image_names = ["" for _ in range(padding)]
            np_shape = (padding, *self._pose_refs.shape[1:])
            dummy_pose_refs = np.zeros(np_shape, dtype=np.uint8)
            np_shape = (padding, *self._pose_refs_depth.shape[1:])
            dummy_pose_refs_depth = np.zeros(np_shape, dtype=np.float32)
            dummy_ref_positions = np.zeros((padding, 3))
            dummy_ref_poses = np.zeros((padding,))
            dummy_pose_regress = np.zeros((padding, 4))
            dummy_mask = np.zeros((padding,))
            self._pose_image_names += dummy_pose_image_names
            self._pose_refs = np.concatenate(
                [self._pose_refs, dummy_pose_refs], axis=0,
            )
            self._pose_refs_depth = np.concatenate(
                [self._pose_refs_depth, dummy_pose_refs_depth], axis=0,
            )
            self.ref_positions = np.concatenate(
                [self.ref_positions, dummy_ref_positions], axis=0,
            )
            self.ref_poses = np.concatenate([self.ref_poses, dummy_ref_poses], axis=0,)
            self._pose_regress = np.concatenate(
                [self._pose_regress, dummy_pose_regress], axis=0,
            )
            self._valid_masks = np.concatenate([self._valid_masks, dummy_mask], axis=0,)

    def generate_topdown_occupancy(self) -> np.array:
        r"""Generates the top-down occupancy map of the environment.
        """
        # Obtain the top-down images from the original environment.
        grid = super().generate_topdown_occupancy()
        # Draw the set of pose references.
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


register(
    id="avd-recon-v0", entry_point="gym_avd.envs:AVDReconEnv",
)
