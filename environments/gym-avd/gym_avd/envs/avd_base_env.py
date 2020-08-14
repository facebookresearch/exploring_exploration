#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import gym
import cv2
import math
import copy
import h5py
import random
import numpy as np
import networkx as nx
import imageio
import scipy.ndimage

from typing import Any, Dict, List, Optional, Tuple

import gym_avd
from gym import spaces, utils
from gym.utils import seeding
from gym_avd.envs.config import *
from gym_avd.envs.utils import *
from gym.envs.registration import register


class AVDBaseEnv(gym.Env):
    """Implements the core task-independent functionality for AVD environments.
    """

    metadata = {"render.modes": ["human", "rgb"]}

    def __init__(self, WIDTH=224, HEIGHT=224, max_steps=None):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.data_root = ROOT_DIR
        h5_path = f"{self.data_root}/processed_images_{WIDTH}x{HEIGHT}.h5"
        npy_path = f"{self.data_root}/processed_scenes_{WIDTH}x{HEIGHT}.npy"
        self.data_h5 = h5py.File(h5_path, "r")
        try:
            self.data_conn = np.load(npy_path)
        except ValueError:
            self.data_conn = np.load(npy_path, allow_pickle=True)
        # Define environment splits.
        self._define_environment_splits()
        # Seed the environment.
        self.seed()
        self.collision_occurred = False
        self.split = "train"
        # Step configuration
        self.max_steps = MAX_STEPS if max_steps is None else max_steps
        self.steps = 0
        self.ndirs = 12
        # Define the observation and action spaces
        self._define_observation_space()
        self._define_action_space()
        # Initialize as None and update in reset()
        self._initialize_environment_variables()

    def _define_environment_splits(self):
        r"""Defines the set of environments and the train / val / test splits.
        """
        self.all_scenes = [
            "Home_001_1",
            "Home_001_2",
            "Home_002_1",
            "Home_003_1",
            "Home_003_2",
            "Home_004_1",
            "Home_004_2",
            "Home_005_1",
            "Home_005_2",
            "Home_006_1",
            "Home_008_1",
            "Home_014_1",
            "Home_014_2",
            "Office_001_1",
            "Home_007_1",
            "Home_010_1",
            "Home_011_1",
            "Home_013_1",
            "Home_015_1",
            "Home_016_1",
        ]
        self.train_scenes = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 18, 19]
        self.val_scenes = [11, 16, 17]
        self.test_scenes = [0, 1, 10, 14, 15]

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
                "delta": spaces.Box(
                    low=LOW_F, high=HIGH_F, shape=(4,), dtype=np.float32,
                ),
                "oracle_action": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32,),
                "collisions": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32,),
            }
        )

    def _define_action_space(self):
        self.action_space = spaces.Discrete(4)

    def _initialize_environment_variables(self):
        self.scene_id = None
        self.start_position = None
        self.start_pose = None
        self.agent_position = None
        self.agent_pose = None
        self.agent_image = None
        self.images_to_idx = None
        self.images_to_camera = None
        self.scene_images = None
        self.scene_depth = None
        self.path_so_far = []
        self.return_topdown_map = False
        self.visitation_counts = None
        AGENT_SPRITE = imageio.imread(
            os.path.join(
                gym_avd.__path__[0],
                "assets",
                "maps_topdown_agent_sprite",
                "100x100.png",
            )
        )
        AGENT_SPRITE = scipy.ndimage.interpolation.rotate(AGENT_SPRITE, -90)
        self.AGENT_SPRITE = np.ascontiguousarray(AGENT_SPRITE)

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

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._episode_over()
        info = self._get_info()

        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.data_h5.close()

    def _load_nav_graph(self):
        r"""Load connectivity graph for current scan and compute
        shortest paths, distances.
        """
        self.paths = dict(nx.all_pairs_dijkstra_path(self.graph))
        self.distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))

    def seed(self, seed: Optional[int] = None):
        r"""Define random number generators for different aspects of
        the environment.
        """
        seed = seeding.hash_seed(seed, max_bytes=2)
        self._rng = random.Random()
        self._rng_read = random.Random()
        self._rng_np = np.random.RandomState()
        self._rng.seed(seed)
        self._rng_read.seed(seed)
        self._rng_np.seed(seed)

    def _sample_oracle_targets(self):
        r"""Sample targets for oracle agent to navigate to.
        """
        self._random_targets = self._rng.choices(
            list(self.images_to_idx.keys()), k=self.max_steps
        )
        # Remove images whose nodes are unreachable.
        self._random_targets = list(filter(self.is_valid_image, self._random_targets))
        self._random_targets_ix = 0

    def set_split(self, split: str):
        r"""Set the split of environments to use for sampling.
        """
        self.split = split

    def set_return_topdown_map(self):
        r"""Compute the topdown map.
        """
        self.return_topdown_map = True

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

        return {
            "im": im,
            "depth": depth,
            "delta": delta,
            "oracle_action": oracle_action,
            "collisions": collision,
        }

    def _get_reward(self):
        reward = 0.0
        return reward

    def _action_to_img(self, action: int) -> str:
        r"""Converts the action to the next agent image.
        0 - turn left by 15 degrees
        1 - turn right by 15 degrees
        2 - move forward
        """
        camera = self.images_to_camera[self.agent_image]
        if action == 0:
            return camera[11]
        elif action == 1:
            return camera[10]
        # For forward action, act based on node connectivity rather
        # than image connectivity.
        elif action == 2:
            agent_nodeix = self.images_to_nodes[self.agent_image]
            agent_node = self.data_conn[self.scene_idx]["nodes"][agent_nodeix]
            neighbors = agent_node["neighbors"]
            agent_x = agent_node["world_pos"][2]
            agent_y = agent_node["world_pos"][0]
            agent_pose = self.agent_pose
            min_angle = math.inf
            min_nb_idx = None
            # Find the neighbor most aligned the agent's current direction.
            for nb_nodeix in neighbors:
                nb_node = self.data_conn[self.scene_idx]["nodes"][nb_nodeix]
                nb_x, nb_y = nb_node["world_pos"][2], nb_node["world_pos"][0]
                dir2nb = math.atan2(nb_y - agent_y, nb_x - agent_x)
                diff_angle = abs(norm_angle(dir2nb - self.agent_pose))
                if diff_angle < min_angle:
                    min_angle = diff_angle
                    min_nb_idx = nb_nodeix
            best_nb_node = self.data_conn[self.scene_idx]["nodes"][min_nb_idx]
            # Find the appropriate viewpoint in this node. This must be as
            # similar as possible to agent_pose.
            best_view_img = None
            best_view_diff = math.inf
            for view in best_nb_node["views"]:
                view_img = view["image_name"]
                view_pose = self._get_pose(view_img)
                view_diff = abs(norm_angle(view_pose - agent_pose))
                if view_diff < best_view_diff:
                    best_view_diff = view_diff
                    best_view_img = view_img

            best_view_pos = np.array(self._get_position(best_view_img))
            curr_view_pos = np.array(self.agent_position)
            best_view_pose = self._get_pose(best_view_img)
            curr_view_pose = self.agent_pose
            delta_pose = abs(norm_angle(best_view_pose - curr_view_pose))
            # Don't act if the following are not satisfied.
            if (
                np.linalg.norm(best_view_pos - curr_view_pos) > 2000
                or abs(min_angle) > math.radians(45)
                or delta_pose > math.radians(25)
            ):  # > 25 degrees
                return self.agent_image
            else:
                return best_view_img
        else:
            # Do nothing action
            return self.agent_image

    def get_environment_extents(self) -> Tuple[float, float, float, float]:
        r"""Computes the min and max extents of the environment along the
        X and Z axes.
        """
        nodes = self.data_conn[self.scene_idx]["nodes"]
        # Need to swap X and Z axes since moving clockwise means moving
        # from Z to X. Example:  In data_conn, see how the angle
        # atan2(dirZ, dirX) changes when you turn clockwise.
        # The angle becomes more negative.
        positions = [[node["world_pos"][2], node["world_pos"][0]] for node in nodes]
        positions = np.array(positions)
        max_x = positions[:, 0].max() * self.scale
        max_z = positions[:, 1].max() * self.scale
        min_x = positions[:, 0].min() * self.scale
        min_z = positions[:, 1].min() * self.scale
        return (min_x, min_z, max_x, max_z)

    def _get_img(self, image_name: str) -> np.array:
        idx = self.images_to_idx[image_name]
        im = self.scene_images[idx]
        return im

    def _get_depth(self, image_name: str) -> np.array:
        idx = self.images_to_idx[image_name]
        depth = self.scene_depth[idx]
        # Fill the holes in the depth map
        kernel = np.ones((7, 7), depth.dtype)
        depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
        depth = self._process_depth(depth)
        return depth

    def _get_position(self, image_name: str) -> List[float]:
        r"""Get the (x, y, z) world position of image_name.
        """
        camera = self.images_to_camera[image_name]
        agent_position = copy.deepcopy(camera[3])
        agent_position = [v * self.scale for v in agent_position]
        # Need to swap X and Z axes since moving clockwise means moving from
        # Z to X. Example:  In data_conn, see how the angle
        # atan2(dirZ, dirX) changes when you turn clockwise.
        # The angle becomes more negative.
        agent_position = [agent_position[2], agent_position[1], agent_position[0]]
        return agent_position

    def _get_pose(self, image_name: str) -> float:
        r"""Get the world pose of image_name.
        """
        camera = self.images_to_camera[image_name]
        dirX, dirZ = camera[4][0], camera[4][2]
        # Need to swap X and Z axes since moving clockwise means moving from
        # Z to X. Example:  In data_conn, see how the angle
        # atan2(dirZ, dirX) changes when you turn clockwise.
        # The angle becomes more negative.
        dirZ, dirX = dirX, dirZ
        agent_pose = math.atan2(dirZ, dirX)
        return agent_pose

    def _episode_over(self):
        episode_over = False
        # This changes for the TDN setup. It will be manually reset here
        # after the last step (self.steps == self.max_steps).
        # So don't set this to true.
        if self.steps > self.max_steps:
            episode_over = True
        return episode_over

    def _get_info(self):
        infos = {}
        # Compute count-based reward.
        scene_conn = self.data_conn[self.scene_idx]
        curr_node_ix = scene_conn["images_to_nodes"][self.agent_image]
        self.visitation_counts[curr_node_ix] += 1.0
        count_based_reward = 1 / math.sqrt(self.visitation_counts[curr_node_ix])
        infos["count_based_reward"] = count_based_reward
        infos["environment_statistics"] = {"scene_id": self.scene_id}
        # Compute top-down map
        if self.return_topdown_map:
            infos["topdown_map"] = self.generate_topdown_occupancy()
        return infos

    def _process_depth(self, depth: np.array) -> np.array:
        depth = depth.astype(np.float32)
        return depth

    def _setup_counts(self):
        scene = self.data_conn[self.scene_idx]
        self.visitation_counts = np.zeros(len(scene["nodes"]))

    def get_agent_position(self) -> List[float]:
        return copy.deepcopy(self.agent_position)

    def get_agent_pose(self) -> float:
        return self.agent_pose

    def _get_oracle_action(self) -> int:
        r"""Computes shortest-path action to the next oracle target.
        """
        oracle_action = 3
        # Action = 3 means that the target was already reached.
        while oracle_action == 3:
            ref_image = self._random_targets[self._random_targets_ix]
            oracle_action = self._base_shortest_path_action(ref_image)
            # If target was already reached and if more targets are remaining.
            if (oracle_action == 3) and self._random_targets_ix < len(
                self._random_targets
            ) - 1:
                self._random_targets_ix += 1
            else:
                break
        return oracle_action

    def _base_shortest_path_action(self, ref_image: str) -> int:
        r"""Returns the next action on the shortest path to a reference image.
        """
        agent_nodeix = self.images_to_nodes[self.agent_image]
        ref_nodeix = self.images_to_nodes[ref_image]
        ref_pose = self._get_pose(ref_image)
        # If the agent has reached the target node, align with the reference
        # pose. If already aligned, return STOP(3).
        dist_agent2ref = self.distances[agent_nodeix][ref_nodeix]
        if agent_nodeix == ref_nodeix or dist_agent2ref < 450.0:
            angle_diff = norm_angle(ref_pose - self.agent_pose)
            # 20 degrees threshold because rotation action is 30 degrees.
            if abs(angle_diff) < math.radians(20):
                return 3
            elif angle_diff < 0:
                # Turn left
                return 0
            else:
                # Turn right
                return 1
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

    def _base_shortest_path_info(self, ref_image: str):
        r"""Returns the shortest path and the shortest path length from the
        current position to ref_image.
        """
        return self._base_generic_shortest_path_info(self.agent_image, ref_image,)

    def _base_generic_shortest_path_info(self, start_image: str, ref_image: str):
        r"""Returns the shortest path and the shortest path length from the
        start_image to ref_image.
        """
        agent_nodeix = self.images_to_nodes[start_image]
        ref_nodeix = self.images_to_nodes[ref_image]
        ref_pose = self._get_pose(ref_image)
        if agent_nodeix == ref_nodeix:
            shortest_path = []
            shortest_path_length = 0
        else:
            try:
                shortest_path = self.paths[agent_nodeix][ref_nodeix]
            except:
                if agent_nodeix not in self.paths:
                    print(
                        "Agents node ({}) itself cannot be found!".format(agent_nodeix)
                    )
                else:
                    print("No path to reference node ({})!".format(ref_nodeix))
                raise ValueError("Wrong key!")

            # Compute shortest path length from start_image to ref_image.
            shortest_path_length = 0
            # Exlude path[0] since it is agent_nodeix itself.
            scene_conn = self.data_conn[self.scene_idx]
            for i in range(1, len(shortest_path)):
                prev_nodeix = shortest_path[i - 1]
                curr_nodeix = shortest_path[i]
                prev_node_pos = scene_conn["nodes"][prev_nodeix]["world_pos"]
                curr_node_pos = scene_conn["nodes"][curr_nodeix]["world_pos"]
                prev_node_pos = np.array(prev_node_pos)
                curr_node_pos = np.array(curr_node_pos)
                dist = self.scale * np.linalg.norm(curr_node_pos - prev_node_pos)
                shortest_path_length += dist.item()

        return shortest_path, shortest_path_length

    def get_geodesic_distance(self, img1: str, img2: str) -> float:
        r"""Compute geodesic distance from img1 to img2.
        """
        return self._base_generic_shortest_path_info(img1, img2)[1]

    def get_sp_action(self, tgt_img: float) -> int:
        r"""Computes the shortest path action from agent position to the
        target image."""
        oracle_action = self._base_shortest_path_action(tgt_img)
        return oracle_action

    def set_reward_bonus(self, reward_bonus_type, reward_K):
        pass

    def is_valid_image(self, x: str) -> bool:
        x_nodeix = self.images_to_nodes[x]
        x_node = self.data_conn[self.scene_idx]["nodes"][x_nodeix]
        if len(x_node["neighbors"]) == 0:
            return False
        else:
            return True

    def set_nref(self, nRef: int):
        pass

    def set_dist_thresh(self, thresh: float):
        pass

    def set_ref_intervals(self, intervals):
        pass

    def set_nlandmarks(self, val):
        pass

    def generate_topdown_occupancy(self) -> np.array:
        r"""Generates the top-down occupancy map of the environment.
        """
        min_x, min_z, max_x, max_z = self.get_environment_extents()
        current_position = self._get_position(self.agent_image)
        current_position = np.array([current_position[0], current_position[2],])
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
        for i in range(len(self.path_so_far) - 1):
            p1 = (
                int((self.path_so_far[i][0] - min_x) / grid_size),
                int((self.path_so_far[i][2] - min_z) / grid_size),
            )
            p2 = (
                int((self.path_so_far[i + 1][0] - min_x) / grid_size),
                int((self.path_so_far[i + 1][2] - min_z) / grid_size),
            )
            grid = cv2.line(grid, p1, p2, (0, 0, 0), max(radius // 2, 1))

        # ========================= Draw the agent ============================
        grid = draw_agent(grid, map_position, map_pose, (0, 255, 0), size=radius)

        return grid

    def get_closest_node(self, point: Tuple[float, float]):
        r"""Obtain the node in the graph closest to point.

        Args:
            point - (x, z) coordinates
        """
        nodes = self.data_conn[self.scene_idx]["nodes"]
        min_dist = math.inf
        min_dist_nodeix = None
        min_dist_nodepos = None
        for nodeix, node in enumerate(nodes):
            npos = np.array(node["world_pos"]) * self.scale
            node_pos = (npos[2], npos[0])
            d = math.sqrt((point[0] - node_pos[0]) ** 2 + (point[1] - node_pos[1]) ** 2)
            if d < min_dist:
                min_dist = d
                min_dist_nodeix = nodeix
                min_dist_nodepos = node_pos

        return min_dist_nodeix, min_dist_nodepos

    def compute_delta(
        self,
        old_position: List[float],
        old_pose: float,
        new_position: List[float],
        new_pose: float,
    ) -> Tuple[float, float, float, float]:
        r"""Computes the odometer reading from old_position to new_position
        in the egocentric coordinates of the starting pose of the episode.
        """
        dx = new_position[0] - old_position[0]
        dz = new_position[2] - old_position[2]
        dr = math.sqrt(dx ** 2 + dz ** 2)
        dtheta = math.atan2(dz, dx) - self.start_pose
        dhead = norm_angle(new_pose - old_pose)
        delev = 0.0
        delta = (dr, dtheta, dhead, delev)
        return delta


register(
    id="avd-base-v0", entry_point="gym_avd.envs:AVDBaseEnv",
)
