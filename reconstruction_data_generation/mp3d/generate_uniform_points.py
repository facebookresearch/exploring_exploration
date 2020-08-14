#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import habitat
import argparse
import numpy as np
import multiprocessing as mp

from PIL import Image


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


def write_data(data_tuple):
    img, img_name = data_tuple
    img = Image.fromarray(img)
    img.save(img_name)


def main(args):

    pool = mp.Pool(32)

    # ====================== Create environment ==========================
    config = habitat.get_config(config_paths=args.config_path)
    config.defrost()
    # Update path to SCENES_DIR, DATA_PATH
    config.DATASET.SCENES_DIR = os.path.join(args.habitat_root, "data/scene_datasets")
    config.DATASET.DATA_PATH = os.path.join(
        args.habitat_root,
        "data/datasets/pointnav/mp3d/v1_unique/{split}/{split}.json.gz",
    )
    config.freeze()
    env = habitat.Env(config=config)

    # Assumes each episode is in a unique environment
    num_episodes = len(env._dataset.episodes)
    all_images = []
    for epcount in range(num_episodes):
        env.reset()
        scene_id = env.current_episode.scene_id
        scene_name = scene_id.split("/")[-1]
        print("Gathering data for scene # {}: {}".format(epcount, scene_name))

        min_x, min_z, max_x, max_z = env._sim.get_environment_extents()
        # Sample a uniform grid of points separated by 4m
        uniform_grid_x = np.arange(min_x, max_x, 4)
        uniform_grid_z = np.arange(min_z, max_z, 4)

        agent_y = env._sim.get_agent_state().position[1]

        scene_images = []
        for x in uniform_grid_x:
            for z in uniform_grid_z:
                random_point = [x.item(), agent_y.item(), z.item()]
                if not env._sim.is_navigable(random_point):
                    print(f"=======> Skipping point ({x}, {agent_y}, {z})")
                    continue

                # Sample multiple viewing angles
                for heading in np.arange(-np.pi, np.pi, np.pi / 3):
                    # This is clockwise rotation about the vertical upward axis
                    rotation = [
                        0,
                        np.sin(heading / 2).item(),
                        0,
                        np.cos(heading / 2).item(),
                    ]

                    obs = env._sim.get_observations_at(random_point, rotation)
                    scene_images.append(obs["rgb"])

        all_images += scene_images

    safe_mkdir(args.save_dir)
    img_tuples = []
    for i, img in enumerate(all_images):
        img_path = os.path.join(args.save_dir, f"image_{i:07d}.png")
        img_tuples.append((img, img_path))

    _ = pool.map(write_data, img_tuples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--config-path", type=str, default="config.yaml")
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--habitat-root", type=str, default="./")

    args = parser.parse_args()

    main(args)
