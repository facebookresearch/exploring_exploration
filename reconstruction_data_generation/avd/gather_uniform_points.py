#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import argparse
import numpy as np
import multiprocessing as mp

from PIL import Image

import gym
import gym_avd


def str2bool(v):
    return True if v.lower() in ["t", "true", "y", "yes"] else False


parser = argparse.ArgumentParser()
parser.add_argument("--save_directory", type=str, default="uniform_samples")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--debug", type=str2bool, default=False)


args = parser.parse_args()
args.env_name = "avd-v1"


def write_data(data_tuple):
    img, img_name = data_tuple
    img = Image.fromarray(img)
    img.save(img_name)


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


safe_mkdir(args.save_directory)
save_dir = args.save_directory

pool = mp.Pool(32)

env = gym.make(args.env_name)


def gather_data(env, scenes, args):
    scene_images = []
    for scene_idx in scenes:
        print("Gathering data for scene: {}".format(scene_idx))
        _ = env.reset(scene_idx=scene_idx)
        min_x, min_z, max_x, max_z = env.get_environment_extents()

        # Sample nodes uniformly @ 2m
        all_nodes = env.data_conn[scene_idx]["nodes"]
        all_nodes_positions = [
            [node["world_pos"][2], node["world_pos"][0]] for node in all_nodes
        ]
        all_nodes_positions = np.array(all_nodes_positions) * env.scale

        range_x = np.arange(min_x, max_x, 2000.0)
        range_z = np.arange(min_z, max_z, 2000.0)
        relevant_nodes = []
        for x in range_x:
            for z in range_z:
                # Find closest node to this coordinate
                min_dist = math.inf
                min_dist_node = None
                for node, node_position in zip(all_nodes, all_nodes_positions):
                    d = np.sqrt(
                        (x - node_position[0]) ** 2 + (z - node_position[1])
                    ).item()
                    if d < min_dist:
                        min_dist = d
                        min_dist_node = node
                relevant_nodes.append(min_dist_node)

        relevant_images = []
        for node in relevant_nodes:
            for j in range(0, 12, 3):
                image_name = node["views"][j]["image_name"]
                relevant_images.append(env._get_img(image_name))

        scene_images += relevant_images

    return scene_images


for split in ["train", "val", "test"]:
    split_path = os.path.join(args.save_directory, split)
    safe_mkdir(split_path)
    split_scenes = getattr(env, "{}_scenes".format(split))
    print("========= Gathering data for split: {} =========".format(split))
    split_images = gather_data(env, split_scenes, args)

    img_tuples = []
    for i, img in enumerate(split_images):
        path = os.path.join(split_path, f"image_{i:07d}.png")
        img_tuples.append((img, path))

    _ = pool.map(write_data, img_tuples)
