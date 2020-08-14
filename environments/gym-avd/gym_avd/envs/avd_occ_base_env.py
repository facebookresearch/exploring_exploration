#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import gym
import math
import copy
import json
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from gym import error, spaces, utils
from gym.utils import seeding
from gym_avd.envs.config import *
from gym_avd.envs.utils import *
from gym.envs.registration import register

from gym_avd.envs.avd_base_env import AVDBaseEnv


class AVDOccBaseEnv(AVDBaseEnv):
    r"""Inherits from AVDBaseEnv and additionally implements occupancy map
    building.
    """
    metadata = {"render.modes": ["human", "rgb"]}

    def __init__(
        self,
        WIDTH: int = 224,
        HEIGHT: int = 224,
        max_steps: Optional[int] = None,
        nRef: int = 1,
        map_scale: float = 50.0,
        map_size: int = 301,
        max_depth: float = 3000,
    ):
        self.nRef = nRef
        self.map_scale = map_scale
        self.map_size = map_size
        self.max_depth = max_depth
        super().__init__(HEIGHT=HEIGHT, WIDTH=WIDTH, max_steps=max_steps)
        # Initializes empty maps.
        self._initialize_map()

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
            }
        )

    def _initialize_environment_variables(self):
        r"""Additionally initializes mapping specific variables.
        """
        super()._initialize_environment_variables()
        # Define mapping specific variables
        self.intrinsics = {
            "fx": 1070.00 * self.WIDTH / 1920.0,
            "fy": 1069.12 * self.HEIGHT / 1080.0,
            "cx": 927.269 * self.WIDTH / 1920.0,
            "cy": 545.760 * self.HEIGHT / 1080.0,
        }
        # Lower and upper bounds of heights for obstacles
        self.height_thresh = [300, 1800]
        self.agent_height = 1000
        self.configs = {
            "small_map_range": 30,
            "large_map_range": 100,
            "large_map_size": self.WIDTH,
            "small_map_size": self.WIDTH,
        }
        if os.path.isfile(AREAS_FILE):
            self.environment_areas = json.load(open(AREAS_FILE, "r"))
        else:
            print("Environment areas not found!")

    def _initialize_map(self):
        # Pre-assign memory to grids_mat.
        grid_size = self.map_scale
        map_size = self.map_size
        large_map_range = self.configs["large_map_range"]
        grid_num = (map_size, map_size)
        self.grids_mat = np.zeros(grid_num, dtype=np.uint8)
        self.count_grids_mat = np.zeros(grid_num, dtype=np.float32)
        # Pre-assign memory that will be needed later in get_local_mapss().
        half_size = max(*grid_num, large_map_range) * 3
        full_size = half_size * 2
        self.ego_map = np.zeros((full_size, full_size), dtype=np.uint8)
        self.seen_count_reward = 0.0

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
        # Sample random targets to navigate to (for oracle action).
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

    def _get_obs(self):
        r"""Compute the observations.
        """
        idx = self.images_to_idx[self.agent_image]
        im = self.scene_images[idx]
        depth = self._process_depth(self.scene_depth[idx])
        delta = np.array(self.delta)
        oracle_action = np.array([self._get_oracle_action()])
        self.path_so_far.append(self.agent_position)
        if self.collision_occurred:
            collision = np.array([1.0])
        else:
            collision = np.array([0.0])
        # Occupancy map specific observations.
        seen_area = self.get_seen_area(im, depth, self.grids_mat, self.count_grids_mat)
        self.inc_area = seen_area - self.seen_area
        self.seen_area = seen_area
        fine_map, coarse_map, highres_coarse_map = self.get_local_maps()

        return {
            "im": im,
            "depth": depth,
            "delta": delta,
            "oracle_action": oracle_action,
            "collisions": collision,
            "coarse_occupancy": coarse_map,
            "fine_occupancy": fine_map,
            "highres_coarse_occupancy": highres_coarse_map,
        }

    def _get_info(self):
        info = super()._get_info()
        # Additionally add the area information.
        info["seen_area"] = self.seen_area
        info["inc_area"] = self.inc_area
        info["coverage_novelty_reward"] = self.seen_count_reward
        total_area = self.environment_areas[self.scene_id]
        if hasattr(self, "environment_areas"):
            info["frac_seen_area"] = 1.0 * self.seen_area / total_area
            info["environment_statistics"]["total_area"] = total_area
        return info

    def get_seen_area(
        self,
        rgb: np.array,
        depth: np.array,
        out_mat: np.array,
        count_out_mat: np.array,
    ) -> int:
        r"""Given new RGBD observations, it updates the global occupancy map
        and computes total area seen after the update.

        Args:
            rgb - uint8 RGB images.
            depth - unnormalized depth inputs with values in mm.
            out_mat - global map to aggregate current inputs in.

        Returns:
            Area seen in the environment after aggregating current inputs.
            Area is measured in gridcells. Multiply by map_scale**2 to
            get area in m^2.
        """
        # ====================== Compute the pointcloud =======================
        XYZ_ego, colors_ego = self.convert_to_pointcloud(rgb, depth)
        XYZ_world = self.register_pointcloud(XYZ_ego)
        # Originally, Y axis is position downward from the ground-plane.
        # Convert Y upward and add agent height to the pointcloud heights.
        XYZ_world[:, 1] = self.agent_height - XYZ_world[:, 1]
        # =================== Compute local occupancy map =====================
        grids_mat = np.zeros(self.grids_mat.shape, dtype=np.uint8)
        points = XYZ_world
        # Compute grid coordinates of points in pointcloud.
        grid_locs = (points[:, [0, 2]] - self.L_min) / self.grid_size
        grid_locs = np.floor(grid_locs).astype(int)
        # Classify points in occupancy map as free/occupied/unknown
        # using height-based thresholds on the point-cloud.
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)
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
        kernel = np.ones((5, 5), np.uint8)
        obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
        # =================== Update global occupancy map =====================
        visible_mask = grids_mat == 2
        occupied_mask = obs_mat == 1
        np.putmask(out_mat, visible_mask, 2)
        np.putmask(out_mat, occupied_mask, 1)
        # Update seen counts to each grid location
        seen_mask = (visible_mask | occupied_mask).astype(np.float32)
        count_out_mat += seen_mask
        inv_count_out_mat = np.ma.array(
            1 / np.sqrt(np.clip(count_out_mat, 1.0, math.inf)), mask=1 - seen_mask
        )
        # =================== Measure area seen (m^2) in the map ====================
        seen_area = float(np.sum(out_mat > 0).item()) * (self.map_scale / 1000.0) ** 2
        self.seen_count_reward = inv_count_out_mat.mean().item()

        return seen_area

    def convert_to_pointcloud(
        self, rgb: np.array, depth: np.array
    ) -> Tuple[np.array, np.array]:
        """Converts rgb and depth images into a sequence of points
        corresponding to the 3D projection of camera points by using
        the intrinsic parameters.

        Args:
            rgb   - uint8 RGB images
            depth - unnormalized depth inputs with values in mm
        Returns:
            XYZ - (N, 3) array of valid X, Y, Z coordinates
                  with X coordinates going right, Y coordinates
                  going down and Z coordinates going forward
            colors - (N, 3) array of colors corresponding to each point in BGR

            Note: These are centered around the agent's egocentric
                  coordinates
        """
        # ================== Compute 3D point projections =====================
        H, W = depth.shape[:2]
        y_im, x_im = np.mgrid[0:H, 0:W]
        xx = (x_im - self.intrinsics["cx"]) / self.intrinsics["fx"]  # (H, W)
        yy = (y_im - self.intrinsics["cy"]) / self.intrinsics["fy"]  # (H, W)
        # 3D world coordinates in mm
        Z = depth[:, :, 0].astype(np.float32)  # (H, W, 1)
        X = xx * Z  # (H, W)
        Y = yy * Z  # (H, W)
        XYZ = np.stack([X, Y, Z], axis=2)  # (H, W, 3)
        XYZ = XYZ.reshape(H * W, 3)
        colors = rgb.reshape(H * W, 3)

        # ================= Filter out invalid depth readings =================
        valid_inputs = Z != 0.0  # (H, W)
        valid_inputs = valid_inputs.reshape(H * W)
        XYZ = XYZ[valid_inputs, :]  # (N, 3)
        colors = colors[valid_inputs, :]  # (N, 3)
        # Depth clipping
        max_depth_valid = XYZ[:, 2] <= self.max_depth
        XYZ = XYZ[max_depth_valid, :]
        colors = colors[max_depth_valid, :]

        return XYZ, colors

    def rotate_points_in_xz(self, XYZ: np.array, theta: float) -> np.array:
        """Rotates points in XYZ by a heading angle theta in the XZ plane.

        Args:
            XYZ - (N, 3) points
            theta - angle w.r.t Z axis / forward facing axis on ground plane

        Returns:
            XYZ_rot - (N, 3) points in rotated coordinates
        """
        ZX = np.take(XYZ, [2, 0], axis=1)
        # Anticlockwise rotation from Z to X
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        ZX_rot = np.dot(ZX, R.T)
        XYZ_rot = np.stack([ZX_rot[:, 1], XYZ[:, 1], ZX_rot[:, 0]], axis=1)

        return XYZ_rot

    def register_pointcloud(self, XYZ: np.array) -> np.array:
        """Registers the egocentric pointcloud in the camera coordinates to
        the world coordinates based on the agent's position, heading.

        Args:
            XYZ - (N, 3) points in the agent's egocentric coordinates
        Returns:
            XYZ_world - (N, 3) points registered to the world coordinates.
        """
        theta = self.agent_pose
        # Swap X and Z coordinates.
        zt = self.agent_position[0]
        xt = self.agent_position[2]
        # Rotate the pointcloud.
        XYZ_rot = self.rotate_points_in_xz(XYZ, theta)
        # Translate the pointcloud.
        XYZ_world = XYZ_rot + np.array([[xt, 0, zt]])
        return XYZ_world

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

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
        top_down_map = self.grids_mat.copy()  # (map_size, map_size)
        # =========== Obtain local crop around the agent ======================
        # Agent's world and map positions.
        xzt_world, xz_map = self.get_camera_grid_pos()
        # Write out the relevant parts of top_down_map to ego_map.
        # A central egocentric crop of the map will be obtained.
        self.ego_map.fill(0)
        half_size = (
            max(
                top_down_map.shape[0],
                top_down_map.shape[1],
                self.configs["large_map_range"],
            )
            * 3
        )
        x_start = int(half_size - xz_map[0])
        y_start = int(half_size - xz_map[1])
        x_end = x_start + top_down_map.shape[0]
        y_end = y_start + top_down_map.shape[1]
        assert (
            x_start >= 0
            and y_start >= 0
            and x_end <= self.ego_map.shape[0]
            and y_end <= self.ego_map.shape[1]
        )
        self.ego_map[x_start:x_end, y_start:y_end] = top_down_map
        # Crop out only the essential parts of the global map.
        # This saves computation cost for the subsequent operations.
        large_map_range = self.configs["large_map_range"]
        crop_start = half_size - int(1.5 * large_map_range)
        crop_end = half_size + int(1.5 * large_map_range)
        ego_map = self.ego_map[crop_start:crop_end, crop_start:crop_end]
        # Rotate the global map to obtain egocentric top-down view.
        half_size = ego_map.shape[0] // 2
        center = (half_size, half_size)
        rot_angle = math.degrees(xzt_world[2]) + 90
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(
            ego_map,
            M,
            (ego_map.shape[1], ego_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,),
        )
        # ============ Obtain final maps at different resolutions =============
        # Obtain the fine occupancy map.
        start = int(half_size - self.configs["small_map_range"])
        end = int(half_size + self.configs["small_map_range"])
        fine_ego_map = ego_map[start:end, start:end]
        assert start >= 0
        assert end <= ego_map.shape[0]
        fine_ego_map = cv2.resize(
            fine_ego_map,
            (self.configs["small_map_size"], self.configs["small_map_size"]),
            interpolation=cv2.INTER_NEAREST,
        )
        fine_ego_map = np.clip(fine_ego_map, 0, 2)
        # Obtain the coarse occupancy map.
        start = half_size - self.configs["large_map_range"]
        end = half_size + self.configs["large_map_range"]
        assert start >= 0
        assert end <= ego_map.shape[0]
        coarse_ego_map_orig = ego_map[start:end, start:end]
        coarse_ego_map = cv2.resize(
            coarse_ego_map_orig,
            (self.configs["large_map_size"], self.configs["large_map_size"]),
            interpolation=cv2.INTER_NEAREST,
        )
        # Obtain a high-resolution coarse occupancy map.
        # This is primarily useful as an input to an A* path-planner.
        highres_coarse_ego_map = cv2.resize(
            coarse_ego_map_orig, (200, 200), interpolation=cv2.INTER_NEAREST,
        )
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

        return fine_ego_map_color, coarse_ego_map_color, highres_coarse_ego_map_color

    def get_camera_grid_pos(self) -> Tuple[np.array, np.array]:
        r"""Returns the agent's current position in both the real world
        (X, Z, theta from Z to X) and the grid world (Xg, Zg) coordinates.
        """
        X, Z = self.agent_position[0], self.agent_position[2]
        # Compute grid world positions.
        grid_x = np.floor((X - self.L_min) / self.grid_size)
        grid_z = np.floor((Z - self.L_min) / self.grid_size)
        # Real world rotation.
        theta = self.agent_pose
        return np.array((X, Z, theta)), np.array((grid_x, grid_z))

    def generate_topdown_occupancy(self) -> np.array:
        r"""Generates the top-down occupancy map of the environment.
        """
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        current_position = self._get_position(self.agent_image)
        current_position = np.array([current_position[0], current_position[2]])
        grid_size = 20.0
        env_size = max(max_z - min_z, max_x - min_x, 8000.0)
        x_pad = (env_size - (max_x - min_x)) // 2
        z_pad = (env_size - (max_z - min_z)) // 2
        min_x = min_x - x_pad
        min_z = min_z - z_pad
        max_x = max_x + x_pad
        max_z = max_z + z_pad
        # Map position of the agent.
        map_position = (current_position - np.array([min_x, min_z])) / grid_size
        map_pose = self._get_pose(self.agent_image)
        # ======= Create an initial top-down layout of the environment ========
        nodes = self.data_conn[self.scene_idx]["nodes"]
        positions = (
            np.array([[node["world_pos"][2], node["world_pos"][0]] for node in nodes])
            * self.scale
        )
        positions = (positions - np.array([[min_x, min_z]])) / grid_size
        world_positions = (
            np.array(
                [
                    [node["world_pos"][2], node["world_pos"][1], node["world_pos"][0]]
                    for node in nodes
                ]
            )
            * self.scale
        )
        full_size = int(max(max_z - min_z, max_x - min_x) / grid_size)
        grid = np.zeros((full_size, full_size, 3), dtype=np.uint8)
        grid.fill(255)
        # Draw the valid locations with fog-of-war desaturation.
        FOG_OF_WAR_DESAT = 0.2
        radius = max(grid.shape[0] // 50, 1)
        for i in range(positions.shape[0]):
            pos = (int(positions[i][0]), int(positions[i][1]))
            world_pos = world_positions[i]
            gpos_x, gpos_y = self.convert_camera_to_grid_pos(world_pos, 0)[1]
            gpos_x, gpos_y = int(gpos_x), int(gpos_y)
            color = (50, 170, 50)
            if self.grids_mat[gpos_x, gpos_y] == 0:
                color = tuple([c * FOG_OF_WAR_DESAT for c in color])
            grid = cv2.circle(grid, pos, radius, color, -1)
        self.top_down_env = np.copy(grid)

        # ==================== Draw the path taken so far =====================
        colormap = cv2.applyColorMap(
            (np.arange(self.max_steps) * 255.0 / self.max_steps).astype(np.uint8),
            cv2.COLORMAP_JET,
        ).squeeze(1)[:, ::-1]
        for i in range(len(self.path_so_far) - 1):
            p1 = (
                int((self.path_so_far[i][0] - min_x) / grid_size),
                int((self.path_so_far[i][2] - min_z) / grid_size),
            )
            p2 = (
                int((self.path_so_far[i + 1][0] - min_x) / grid_size),
                int((self.path_so_far[i + 1][2] - min_z) / grid_size),
            )
            color = tuple(colormap[i].tolist())
            grid = cv2.line(grid, p1, p2, color, max(radius // 2, 1))

        # ========================= Draw the agent ============================
        grid = draw_agent_sprite(
            grid, map_position, map_pose, self.AGENT_SPRITE, size=radius * 2 + 1
        )

        return grid

    def convert_camera_to_grid_pos(
        self, position: np.array, pose: float
    ) -> Tuple[np.array, np.array]:
        # For purposes of grid, forward is z and right is x
        z, x = position[0], position[2]
        theta = pose
        grid_x = np.floor((x - self.L_min) / self.grid_size)
        grid_z = np.floor((z - self.L_min) / self.grid_size)
        return np.array((x, z, theta)), np.array((grid_x, grid_z))

    def set_nref(self, nRef):
        pass


register(
    id="avd-occ-base-v0", entry_point="gym_avd.envs:AVDOccBaseEnv",
)
