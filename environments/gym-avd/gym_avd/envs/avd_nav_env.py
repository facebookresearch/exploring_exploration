#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import math
import copy
import json
import gzip
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from gym import error, spaces, utils
from gym.utils import seeding
from gym_avd.envs.config import *
from gym_avd.envs.utils import *
from gym_avd.envs.avd_occ_base_env import AVDOccBaseEnv

from gym.envs.registration import register


class AVDNavEnv(AVDOccBaseEnv):
    """Implements the environment for a pointgoal navigation task which consists of
    a map-building exploration phase followed by navigation. The oracle agent visits
    randomly sampled views in the environment during exploration.
    """

    def __init__(
        self,
        WIDTH: int = 84,
        HEIGHT: int = 84,
        max_steps: Optional[int] = None,
        t_exp: Optional[int] = None,
        t_nav: Optional[int] = None,
    ):
        super().__init__(HEIGHT=HEIGHT, WIDTH=WIDTH, max_steps=max_steps)
        if t_exp == None or t_nav == None:
            self.max_steps = self.max_steps * 2
            self.t_exp = self.max_steps // 2
        else:
            self.t_exp = t_exp
            self.t_nav = t_nav
            self.max_steps = self.t_exp + self.t_nav

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
                "sp_action": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32,),
                "target_im": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(H, W, 3), dtype=np.float32,
                ),
                "target_depth": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(H, W, 1), dtype=np.float32,
                ),
                "target_regress": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(4,), dtype=np.float32,
                ),
                "start_regress": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(4,), dtype=np.float32,
                ),
                "target_grid_loc": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(2,), dtype=np.float32,
                ),
            }
        )

    def _initialize_environment_variables(self):
        r"""Additionally initializes navigation related variables.
        """
        super()._initialize_environment_variables()
        # Defines navigation specific variables.
        self.min_dist = 6000.0
        self._nav_success_thresh = 500.0
        self._prev_action = None
        self.episodes = None
        self.episodes_idx = None

    def reset(self, scene_idx: Optional[int] = None):
        # Spawn agent in a new scene at a random image.
        if self.episodes is None:
            print("Sampling random episodes!")
            if scene_idx is None:
                scenes = getattr(self, "{}_scenes".format(self.split))
                self.scene_idx = self._rng.choice(scenes)
            else:
                self.scene_idx = scene_idx
        else:
            if self.episodes_idx is None:
                self.episodes_idx = 0
            else:
                self.episodes_idx = (self.episodes_idx + 1) % len(self.episodes)
            self.scene_idx = self.episodes[self.episodes_idx]["scene_idx"]
            print(
                "Sampling from pre-defined episodes! Scene id: "
                f"{self.scene_idx}, Episode id: {self.episodes_idx}"
            )
        self.scene_id = self.all_scenes[self.scene_idx]
        scene_conn = self.data_conn[self.scene_idx]
        self.images_to_idx = scene_conn["images_to_idx"]
        self.images_to_camera = scene_conn["images_to_camera"]
        self.images_to_nodes = scene_conn["images_to_nodes"]
        self.scale = scene_conn["scale"]  # Converts positions to mm.
        if self.episodes is None:
            self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
            while not self.is_valid_image(self.agent_image):
                self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
        else:
            self.agent_image = self.episodes[self.episodes_idx]["start_image"]
        # Initialize the environment variables.
        self.start_position = self._get_position(self.agent_image)
        self.start_pose = self._get_pose(self.agent_image)
        self.exp_start_image = self.agent_image
        self.exp_start_position = copy.deepcopy(self.start_position)
        self.exp_start_pose = self.start_pose
        self.agent_pose = self.start_pose
        self.agent_position = copy.deepcopy(self.start_position)
        self.scene_images = np.array(self.data_h5[f"{self.scene_id}/rgb"])
        self.scene_depth = np.array(self.data_h5[f"{self.scene_id}/depth"])
        self.delta = (0.0, 0.0, 0.0, 0.0)
        self.collision_occurred = False
        self.steps = 0
        self.graph = create_nav_graph(self.data_conn[self.scene_idx])
        self._load_nav_graph()
        # Sample random exploration oracle targets.
        self._sample_oracle_targets()
        self._stuck_at_oracle = False
        # Path taken so far
        self.exp_path_so_far = []
        self.nav_path_so_far = []
        self.top_down_env = None
        # Count-based rewards
        self._setup_counts()
        # Create occupancy map related variables.
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
        # Sample target.
        self._sample_target()
        # Navigation evaluation stuff
        self._path_length_so_far = 0.0
        self._start_regress = np.array([0, 0, 0, 0])

        return self._get_obs()

    def step(self, action):
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
        # Prepare for navigation phase at the end of exploration.
        if self.steps == self.t_exp:
            # Exploration phase is over.
            self._respawn_agent()
            # Sample the actual target.
            self._sample_target()
        # Update path length during navigation phase.
        if self.steps > self.t_exp:
            self._path_length_so_far += dr
        # Keep track of the previous action taken.
        self._prev_action = action

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._episode_over()
        info = self._get_info()

        return obs, reward, done, info

    def _get_info(self):
        infos = {}
        infos["environment_statistics"] = {
            "scene_id": self.scene_id,
            "episode_id": self.episodes_idx,
        }
        # Area information.
        infos["seen_area"] = self.seen_area
        infos["inc_area"] = self.inc_area
        if self.return_topdown_map:
            infos["topdown_map"] = self.generate_topdown_occupancy()
        # Was exploration completed at this time-step?
        finished_exploration = True if self.steps == self.t_exp else False
        infos["finished_exploration"] = finished_exploration
        # Navigation specific information.
        if self.steps >= self.t_exp:
            infos["nav_error"] = self._nav_error_now()
            infos["success_rate"] = self._success_rate_now()
            infos["spl"] = self._spl_now()
            infos["heading_error"] = self._heading_error_now()
            infos["shortest_path_length"] = self._spd_to_tgt
            infos["agent_pos"] = np.array(self.agent_position)
            infos["target_pos"] = np.array(self.tgt_position)
            infos["path_length"] = self._path_length_so_far
        return infos

    def _get_obs(self):
        r"""Compute the observations.
        """
        idx = self.images_to_idx[self.agent_image]
        im = self.scene_images[idx]
        depth = self._process_depth(self.scene_depth[idx])
        delta = np.array(self.delta)
        oracle_action = np.array([self._get_oracle_action()])
        if self.steps < self.t_exp:
            # During the exploration phase, update the exploration path taken.
            self.exp_path_so_far.append(self.agent_position)
        if self.collision_occurred:
            collision = np.array([1.0])
        else:
            collision = np.array([0.0])
        # Occupancy map specific observations.
        if self.steps <= self.t_exp:
            # Update the occupancy map only during exploration. At navigation
            # time, this is kept frozen.
            seen_area = self.get_seen_area(
                im, depth, self.grids_mat, self.count_grids_mat
            )
            self.inc_area = seen_area - self.seen_area
            self.seen_area = seen_area
        fine_map, coarse_map, highres_coarse_map = self.get_local_maps()
        if self.steps >= self.t_exp:
            # During navigation phase, update the navigation path taken.
            self.nav_path_so_far.append(self.agent_position)
        # Navigation specific observations.
        if self.steps >= self.t_exp:
            self.nav_path_so_far.append(self.agent_position)
            target_grid_loc = self._get_egocentric_grid_loc()
        else:
            target_grid_loc = np.array((0, 0))
        target_im = np.copy(self._tgt_image)
        target_depth = np.copy(self._tgt_depth)
        target_pose_regress = np.copy(self._tgt_regress)
        sp_action = np.array([self._get_sp_action()])
        start_regress = np.copy(self._start_regress)
        curr_regress = self._get_regress(
            np.array(self.agent_position), np.array([self.agent_pose]),
        )

        return {
            "im": im,
            "depth": depth,
            "delta": delta,
            "oracle_action": oracle_action,
            "collisions": collision,
            "coarse_occupancy": coarse_map,
            "fine_occupancy": fine_map,
            "highres_coarse_occupancy": highres_coarse_map,
            "target_im": target_im,
            "target_depth": target_depth,
            "target_regress": target_pose_regress,
            "sp_action": sp_action,
            "start_regress": start_regress,
            "curr_regress": curr_regress,
            "target_grid_loc": target_grid_loc,
        }

    def _get_egocentric_grid_loc(self) -> np.array:
        r"""Obtain the target position in the highres coarse occupancy
        map coordinates. The egocentric occupancy map is forward
        facing upward.
        """
        # Obtain target coordinates relative to current agent position.
        tgt_position = self.tgt_position
        tgt_x = tgt_position[0]
        tgt_y = tgt_position[2]
        curr_x = self.agent_position[0]
        curr_y = self.agent_position[2]
        curr_t = self.agent_pose
        r_ct = math.sqrt((tgt_x - curr_x) ** 2 + (tgt_y - curr_y) ** 2)
        t_ct = math.atan2(tgt_y - curr_y, tgt_x - curr_x) - curr_t
        # Convert the relative coordinates to the highres coarse occupancy
        # coordinate system.
        hshp = self.observation_space.spaces["highres_coarse_occupancy"].shape
        height, width = hshp[:2]
        wby2 = width // 2
        hby2 = height // 2
        disp_y = -r_ct * math.cos(t_ct) / self.grid_size
        disp_x = r_ct * math.sin(t_ct) / self.grid_size
        disp_y = height * disp_y / (2 * self.configs["large_map_range"] + 1)
        disp_x = width * disp_x / (2 * self.configs["large_map_range"] + 1)
        grid_y = np.clip(hby2 + disp_y, 0, height - 1)
        grid_x = np.clip(wby2 + disp_x, 0, width - 1)

        return np.array((grid_x, grid_y))

    def _get_regress(self, position: np.array, pose: np.array) -> np.array:
        r"""Compute the relative pose of (position, pose) from the
        exploration starting point in polar coordinates.
        """
        position = position.tolist()
        pose = pose[0].item()
        dx = position[0] - self.exp_start_position[0]
        dz = position[2] - self.exp_start_position[2]
        dr = math.sqrt(dx ** 2 + dz ** 2)
        dtheta = math.atan2(dz, dx) - self.exp_start_pose
        dhead = pose - self.exp_start_pose
        delev = 0.0
        regress = (dr, dtheta, dhead, delev)

        return np.array(regress)

    def _sample_target(self):
        r"""Sample the navigation target. During the exploration phase, the
        navigation target is a dummy image. During the navigation phase, the
        navigation target is sampled either from the episode data or
        randomly.
        """
        # Exploration phase.
        if self.steps < self.t_exp:
            im_shape = self.observation_space.spaces["target_im"].shape
            depth_shape = self.observation_space.spaces["target_depth"].shape
            self._tgt_image_name = None
            self._tgt_image = np.zeros(im_shape, dtype=np.uint8)
            self._tgt_depth = np.zeros(depth_shape, dtype=np.uint8)
            self.tgt_position = np.array((0, 0, 0))
            self.tgt_pose = np.array([0])
            self._tgt_regress = np.array([0, 0, 0, 0])
        # Navigation phase.
        elif self.episodes is None:
            # Navigation phase when predefined episode information is
            # unavailable. In this case, sample a random image that is
            # a certain distance threshold (self.min_dist) away.
            images = list(self.images_to_idx.keys())
            d2t = 0.0
            trials = 0
            best_tgt_img = None
            best_tgt_dist = 0.0
            best_tgt_pos = None
            best_tgt_pose = None
            start_img_name = self.start_image
            # Search for a valid target till the number of trials runs out
            # or the condition is satisfied.
            while (d2t < self.min_dist and trials < 1000) or not self.is_valid_image(
                self._tgt_image_name
            ):
                # Sample random image.
                self._tgt_image_name = self._rng.choice(images)
                tgt_position = self._get_position(self._tgt_image_name)
                tgt_pose = self._get_pose(self._tgt_image_name)
                tgt_img_name = self._tgt_image_name
                tgt_nodeix = self.images_to_nodes[tgt_img_name]
                start_nodeix = self.images_to_nodes[start_img_name]
                try:
                    d2t = self.distances[start_nodeix][tgt_nodeix]
                except KeyError:
                    d2t = -math.inf
                # Keep track of the best target so far.
                if d2t > best_tgt_dist:
                    best_tgt_dist = d2t
                    best_tgt_img = self._tgt_image_name
                    best_tgt_pos = tgt_position
                    best_tgt_pose = tgt_pose
                trials += 1
            # Set the target as the best target found so far.
            self._tgt_image_name = best_tgt_img
            self._tgt_position = np.array(best_tgt_pos)
            self._tgt_pose = np.array([best_tgt_pose])
            tgt_idx = self.images_to_idx[self._tgt_image_name]
            self._tgt_image = self.scene_images[tgt_idx]
            self._tgt_depth = self._process_depth(self.scene_depth[tgt_idx])
            self._tgt_regress = self._get_regress(self._tgt_position, self._tgt_pose)
            self._spd_to_tgt = self._base_shortest_path_info(self._tgt_image_name)[1]
        else:
            # Navigation phase when predefined episode information is
            # provided.
            tgt_img = self.episodes[self.episodes_idx]["goal_image"]
            self._tgt_image_name = tgt_img
            self._tgt_position = np.array(self._get_position(tgt_img))
            self._tgt_pose = np.array([self._get_pose(tgt_img)])
            tgt_idx = self.images_to_idx[self._tgt_image_name]
            self._tgt_image = self.scene_images[tgt_idx]
            self._tgt_depth = self._process_depth(self.scene_depth[tgt_idx])
            self._tgt_regress = self._get_regress(self._tgt_position, self._tgt_pose,)
            self._spd_to_tgt = self._base_shortest_path_info(self._tgt_image_name)[1]

    def _sample_oracle_targets(self):
        r"""Sample random targets for oracle agent to navigate to.
        """
        self._random_targets = self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps,
        )
        # Remove images whose nodes are unreachable
        self._random_targets = list(filter(self.is_valid_image, self._random_targets))
        self._random_targets_ix = 0

    def _nav_error_now(self) -> float:
        r"""At this very step, what is the navigation error?
        """
        shortest_path_dist = self._base_shortest_path_info(self._tgt_image_name)[1]
        return shortest_path_dist

    def _success_rate_now(self) -> float:
        r"""At this very step, what is the success rate?
        """
        nav_error = self._nav_error_now()
        if (
            self.steps >= self.t_exp
            and self._prev_action == 3
            and nav_error < self._nav_success_thresh
        ):
            # An episode is successful only if the STOP action was executed
            # within a sucess threshold distance from the target.
            return 1.0
        return 0.0

    def _spl_now(self):
        r"""At this very step, what is SPL? SPL is defined as the
        success-rate normalized by the path-length.
        See https://arxiv.org/abs/1807.06757.
        """
        success_rate = self._success_rate_now()
        if self.steps >= self.t_exp:
            start_nodeix = self.images_to_nodes[self.start_image]
            tgt_nodeix = self.images_to_nodes[self._tgt_image_name]
            spd = self.distances[start_nodeix][tgt_nodeix]
            return success_rate * spd / max(self._path_length_so_far, spd)
        else:
            return 0.0

    def _heading_error_now(self):
        """
        At this very step, what is the error in heading b/w target
        and current position?
        """
        curr_angle = self.agent_pose
        tgt_angle = self.get_tgt_pose()[0].item()
        diff_angle = curr_angle - tgt_angle
        diff_angle = math.atan2(math.sin(diff_angle), math.cos(diff_angle))
        diff_angle = abs(diff_angle)
        return diff_angle

    def get_tgt_position(self):
        return np.copy(self.tgt_position)

    def get_tgt_pose(self):
        return np.copy(self.tgt_pose)

    def _respawn_agent(self):
        r"""Respawn the agent at the starting image of exploration and
        update the relevant variables.
        """
        self.agent_image = self.exp_start_image
        self.start_image = self.agent_image
        self.start_position = self._get_position(self.agent_image)
        self.start_pose = self._get_pose(self.agent_image)
        self.agent_position = copy.deepcopy(self.start_position)
        self.agent_pose = self.start_pose
        # Reset delta to 0
        self.delta = (0.0, 0.0, 0.0, 0.0)
        self._start_regress = self._get_regress(
            np.array(self.start_position), np.array([self.start_pose]),
        )

    def _get_sp_action(self):
        if self.steps < self.t_exp:
            return self._rng.randint(0, self.action_space.n - 1)
        else:
            return self._base_shortest_path_action(self._tgt_image_name)

    def set_t_exp_and_nav(self, t_exp: int, t_nav: int):
        self.t_exp = t_exp
        self.t_nav = t_nav
        self.max_steps = self.t_exp + self.t_nav

    def set_split(self, split):
        self.split = split
        if self.split == "test":
            self.episodes = json.load(open(POINTNAV_TEST_EPISODES_PATH, "r"))

    def _episode_over(self) -> bool:
        r"""Decides episode termination if the agent exceeding max_steps or
        executed the stop action.
        """
        episode_over = False
        # When this is called at self.steps == self.t_exp, that is the last
        # exploration action taken. So, make sure that self.steps > self.t_exp
        # when termination is issued for action = 3.
        if self.steps >= self.max_steps or (
            self.steps > self.t_exp and self._prev_action == 3
        ):
            episode_over = True
        return episode_over

    def set_min_dist(self, min_dist):
        r"""Sets minimum distance b/w source and target for navigation.
        """
        self.min_dist = min_dist

    def generate_topdown_occupancy(self) -> np.array:
        r"""Generates the top-down occupancy map of the environment.
        """
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        grid_size = 20.0
        env_size = max(max_z - min_z, max_x - min_x, 8000.0)
        x_pad = (env_size - (max_x - min_x)) // 2
        z_pad = (env_size - (max_z - min_z)) // 2
        min_x = min_x - x_pad
        min_z = min_z - z_pad
        max_x = max_x + x_pad
        max_z = max_z + z_pad
        # ======= Create an initial top-down layout of the environment ========
        if self.top_down_env is None:
            nodes = self.data_conn[self.scene_idx]["nodes"]
            positions = (
                np.array(
                    [[node["world_pos"][2], node["world_pos"][0]] for node in nodes]
                )
                * self.scale
            )
            positions = (positions - np.array([[min_x, min_z]])) / grid_size
            full_size = int(max(max_z - min_z, max_x - min_x) / grid_size)
            grid = np.zeros((full_size, full_size, 3), dtype=np.uint8)
            grid.fill(255)
            # Draw the valid locations.
            radius = max(grid.shape[0] // 50, 1)
            for i in range(positions.shape[0]):
                pos = (int(positions[i][0]), int(positions[i][1]))
                grid = cv2.circle(grid, pos, radius, (125, 125, 125), -1)
            self.top_down_env = np.copy(grid)
        else:
            grid = np.copy(self.top_down_env)
            radius = max(grid.shape[0] // 50, 1)
        # ==================== Draw the path taken so far =====================
        if self.steps < self.t_exp:
            # Exploration phase: Only draw the exploration trajectories.
            for i in range(len(self.exp_path_so_far) - 1):
                p1 = (
                    int((self.exp_path_so_far[i][0] - min_x) / grid_size),
                    int((self.exp_path_so_far[i][2] - min_z) / grid_size),
                )
                p2 = (
                    int((self.exp_path_so_far[i + 1][0] - min_x) / grid_size),
                    int((self.exp_path_so_far[i + 1][2] - min_z) / grid_size),
                )
                grid = cv2.line(grid, p1, p2, (0, 0, 0), max(radius // 2, 1))
        else:
            # Navigation phase: Draw both exploration and navigation
            # trajectories, but in different colors.
            # Draw the exploration points in cyan.
            for exp_pt in self.exp_path_so_far:
                pt = (
                    int((exp_pt[0] - min_x) / grid_size),
                    int((exp_pt[2] - min_z) / grid_size),
                )
                grid = cv2.circle(
                    grid, pt, int(1.2 * radius), (0, 255, 255), max(radius // 3, 1),
                )
            # Draw the navigation path in black.
            for i in range(len(self.nav_path_so_far) - 1):
                p1 = (
                    int((self.nav_path_so_far[i][0] - min_x) / grid_size),
                    int((self.nav_path_so_far[i][2] - min_z) / grid_size),
                )
                p2 = (
                    int((self.nav_path_so_far[i + 1][0] - min_x) / grid_size),
                    int((self.nav_path_so_far[i + 1][2] - min_z) / grid_size),
                )
                grid = cv2.line(grid, p1, p2, (0, 0, 0), max(radius // 2, 1))
            # Draw the navigation target.
            tgt_img = self._tgt_image_name
            curr_pos = self._get_position(tgt_img)
            curr_pos = np.array([curr_pos[0], curr_pos[2]])
            curr_pos = (curr_pos - np.array([min_x, min_z])) / grid_size
            curr_theta = self._get_pose(tgt_img)
            grid = draw_agent(grid, curr_pos, curr_theta, (255, 0, 0), size=radius,)
        # ========================= Draw the agent ============================
        curr_pos = self._get_position(self.agent_image)
        curr_pos = np.array([curr_pos[0], curr_pos[2]])
        curr_pos = (curr_pos - np.array([min_x, min_z])) / grid_size
        curr_theta = self._get_pose(self.agent_image)
        grid = draw_agent(grid, curr_pos, curr_theta, (0, 255, 0), size=radius,)

        return grid


class AVDNavVisitLandmarksEnv(AVDNavEnv):
    """Implements the environment for a pointgoal navigation task which consists of
    a map-building exploration phase followed by navigation. The oracle agent visits
    landmarks in the environment during exploration.
    """

    metadata = {"render.modes": ["human", "rgb"]}

    def __init__(
        self,
        nRef: int = 20,
        max_steps: Optional[int] = None,
        t_exp: Optional[int] = None,
        t_nav: Optional[int] = None,
    ):
        self.nRef = nRef
        super().__init__(max_steps=max_steps, t_exp=t_exp, t_nav=t_nav)

    def _initialize_environment_variables(self):
        r"""Additionally initializes landmarks related variables.
        """
        super()._initialize_environment_variables()
        # Defines landmarks sampling related variables.
        self.cluster_root_dir = CLUSTER_ROOT_DIR
        self.ref_sample_intervals = None

    def reset(self, scene_idx: Optional[int] = None):
        # Spawn agent in a new scene at a random image.
        if self.episodes is None:
            print("Sampling random episodes!")
            if scene_idx is None:
                scenes = getattr(self, "{}_scenes".format(self.split))
                self.scene_idx = self._rng.choice(scenes)
            else:
                self.scene_idx = scene_idx
        else:
            if self.episodes_idx is None:
                self.episodes_idx = 0
            else:
                self.episodes_idx = (self.episodes_idx + 1) % len(self.episodes)
            self.scene_idx = self.episodes[self.episodes_idx]["scene_idx"]
            print(
                "Sampling from pre-defined episodes! Scene id: "
                f"{self.scene_idx}, Episode id: {self.episodes_idx}"
            )
        self.scene_id = self.all_scenes[self.scene_idx]
        scene_conn = self.data_conn[self.scene_idx]
        self.images_to_idx = scene_conn["images_to_idx"]
        self.images_to_camera = scene_conn["images_to_camera"]
        self.images_to_nodes = scene_conn["images_to_nodes"]
        self.scale = scene_conn["scale"]  # Converts positions to mm.
        if self.episodes is None:
            self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
            while not self.is_valid_image(self.agent_image):
                self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
        else:
            self.agent_image = self.episodes[self.episodes_idx]["start_image"]
        # Initialize the environment variables.
        self.start_position = self._get_position(self.agent_image)
        self.start_pose = self._get_pose(self.agent_image)
        self.exp_start_image = self.agent_image
        self.exp_start_position = copy.deepcopy(self.start_position)
        self.exp_start_pose = self.start_pose
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
        # Sample random exploration oracle targets.
        self._sample_oracle_targets()
        self._stuck_at_oracle = False
        # Path taken so far.
        self.exp_path_so_far = []
        self.nav_path_so_far = []
        self.top_down_env = None
        # Count-based rewards.
        self._setup_counts()
        # Create occupancy map related variables.
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
        # Sample target.
        self._sample_target()
        # Navigation evaluation stuff.
        self._path_length_so_far = 0.0
        self._start_regress = np.array([0, 0, 0, 0])

        return self._get_obs()

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

    def _sample_oracle_targets(self):
        r"""Sample random targets for oracle agent to navigate to.
        """
        self._random_targets = self.get_route_visiting_refs()
        self._random_targets += self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps,
        )
        # Remove images whose nodes are unreachable
        self._random_targets = list(filter(self.is_valid_image, self._random_targets,))
        self._random_targets_ix = 0

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


class AVDNavVisitObjectsEnv(AVDNavEnv):
    def __init__(
        self,
        max_steps: Optional[int] = None,
        t_exp: Optional[int] = None,
        t_nav: Optional[int] = None,
    ):
        super().__init__(
            HEIGHT=HEIGHT, WIDTH=WIDTH, max_steps=max_steps, t_exp=t_exp, t_nav=t_nav,
        )

    def reset(self, scene_idx: Optional[int] = None):
        # Spawn agent in a new scene at a random image.
        if self.episodes is None:
            print("Sampling random episodes!")
            if scene_idx is None:
                scenes = getattr(self, "{}_scenes".format(self.split))
                self.scene_idx = self._rng.choice(scenes)
            else:
                self.scene_idx = scene_idx
        else:
            if self.episodes_idx is None:
                self.episodes_idx = 0
            else:
                self.episodes_idx = (self.episodes_idx + 1) % len(self.episodes)
            self.scene_idx = self.episodes[self.episodes_idx]["scene_idx"]
            print(
                "Sampling from pre-defined episodes! Scene id: "
                f"{self.scene_idx}, Episode id: {self.episodes_idx}"
            )
        self.scene_id = self.all_scenes[self.scene_idx]
        scene_conn = self.data_conn[self.scene_idx]
        self.images_to_idx = scene_conn["images_to_idx"]
        self.images_to_camera = scene_conn["images_to_camera"]
        self.images_to_nodes = scene_conn["images_to_nodes"]
        self.scale = scene_conn["scale"]  # Converts positions to mm.
        if self.episodes is None:
            self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
            while not self.is_valid_image(self.agent_image):
                self.agent_image = self._rng.choice(list(self.images_to_idx.keys()))
        else:
            self.agent_image = self.episodes[self.episodes_idx]["start_image"]
        # Initialize the environment variables.
        self.start_position = self._get_position(self.agent_image)
        self.start_pose = self._get_pose(self.agent_image)
        self.exp_start_image = self.agent_image
        self.exp_start_position = copy.deepcopy(self.start_position)
        self.exp_start_pose = self.start_pose
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
        # Sample exploration oracle targets.
        self._sample_oracle_targets()
        self._stuck_at_oracle = False
        # Path taken so far.
        self.exp_path_so_far = []
        self.nav_path_so_far = []
        self.top_down_env = None
        # Count-based rewards.
        self._setup_counts()
        # Create occupancy map related variables.
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
        # Sample target.
        self._sample_target()
        # Navigation evaluation stuff.
        self._path_length_so_far = 0.0
        self._start_regress = np.array([0, 0, 0, 0])

        return self._get_obs()

    def _sample_oracle_targets(self):
        r"""Sample objects as targets for oracle agent to navigate to.
        """
        self._random_targets = self.get_route_visiting_objs()
        self._random_targets += self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps,
        )
        # Remove images whose nodes are unreachable
        self._random_targets = list(filter(self.is_valid_image, self._random_targets,))
        self._random_targets_ix = 0

    def get_route_visiting_objs(self):
        r"""Get a route visiting the object instances in a greedy fashion.
        """
        # Sample images of object instances.
        sampled_ids = self.valid_instances.keys()
        pose_image_names = []
        for inst_id in sampled_ids:
            img = self._rng.choice(self.valid_instances[inst_id])
            if self.is_valid_image(img):
                pose_image_names.append(img)
        # Get only unique images in case of repeated content.
        images = set(pose_image_names)
        # Sample a route visiting the valid objects.
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
    id="avd-nav-random-oracle-v0", entry_point="gym_avd.envs:AVDNavEnv",
)

register(
    id="avd-nav-landmarks-oracle-v0",
    entry_point="gym_avd.envs:AVDNavVisitLandmarksEnv",
)

register(
    id="avd-nav-objects-oracle-v0", entry_point="gym_avd.envs:AVDNavVisitObjectsEnv",
)
