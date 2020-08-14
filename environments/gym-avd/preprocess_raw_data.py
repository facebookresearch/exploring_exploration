#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to preprocess AVD raw data to a format compatible with
the simulation environment.
"""
import os
import cv2
import pdb
import copy
import math
import json
import h5py
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiprocessing import Pool

RESOLUTION = (84, 84)

all_scenes_list = [
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


class View:
    def __init__(self, image_name, angle, camera):
        self.image_name = image_name
        self.angle = angle
        self.camera = camera

    def get(self):
        return {
            "image_name": self.image_name,
            "angle": self.angle,
            "camera": self.camera,
        }


def read_images_of_node_rgb(img_paths):
    imgs = []
    global RESOLUTION

    fail_counts = 0
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path)
            img = np.flip(img, axis=2)
            img = cv2.resize(img, RESOLUTION, interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        except:
            imgs.append(np.zeros((*RESOLUTION, 3)).astype(np.uint8))
            fail_counts += 1

    if fail_counts > 0:
        print("========> Number of RGB read failures: {}".format(fail_counts))
    return imgs


def read_images_of_node_depth(img_paths):
    imgs = []
    global RESOLUTION

    fail_counts = 0
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
            # To preserve zeros as zeros
            img = cv2.resize(img, RESOLUTION, interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
        except:
            imgs.append(np.zeros(RESOLUTION).astype(np.uint8))
            fail_counts += 1

    if fail_counts > 0:
        print("========> Number of depth read failures: {}".format(fail_counts))
    return imgs


def process_scene(scene_path, pool):
    """
    Organize a given scene into nodes and load corresponding data.

    Outputs:
        scene - dictionary containing all details about the scene
        images_per_node - list of images contained in each node
        depths_per_node - list of depths contained in each node
    """
    images_path = os.path.join(scene_path, "jpg_rgb")
    image_structs_path = os.path.join(scene_path, "image_structs.mat")

    # Load data.
    image_structs = sio.loadmat(image_structs_path)
    scale = image_structs["scale"][0][0]
    image_structs = image_structs["image_structs"][0]

    """
    Each element of image_structs corresponds to information from a certain camera viewpoint.
    camera[0]  - image_name
    camera[1]  - t
    camera[2]  - R
    camera[3]  - world_pos
    camera[4]  - direction
    camera[5]  - quat
    camera[6]  - scaled_world_pos
    camera[7]  - image_id
    camera[8]  - camera_id
    camera[9]  - cluster_id
    camera[10] - rotate_cw (image_name)
    camera[11] - rotate_ccw (image_name)
    camera[12] - translate_forward (image_name)
    camera[13] - translate_backward (image_name)
    camera[14] - translate_right (image_name)
    camera[15] - translate_left (image_name)
    """

    all_images = set()
    images_to_camera = {}
    for camera in image_structs:
        camera = [v.squeeze().tolist() for v in camera]
        all_images.add(camera[0])
        images_to_camera[camera[0]] = camera

    scene = {
        "scene_path": scene_path,
        "nodes": [],
        "scale": scale,
        "calibration": None,
        "calibration_keys": None,
        "images_to_camera": images_to_camera,
        "images_to_idx": None,
        "images_to_nodes": {},
    }

    image_paths_per_node = []
    depth_paths_per_node = []
    # Load calibration information
    with open(os.path.join(scene_path, "cameras.txt")) as calib_file:
        calibration_data = calib_file.readlines()

    scene["calibration_keys"] = "res_x,res_y,fx,fy,cx,cy,k1,k2,p1,p2"
    scene["calibration"] = [float(data) for data in calibration_data[3].split()[2:12]]

    # Generate scene nodes. A node is an (x, y, z) location in the environment from which
    # multiple viewpoints are sampled.
    for image in list(all_images):
        if image not in all_images:
            continue

        node_idx = len(scene["nodes"])
        camera = images_to_camera[image]
        node = {"views": [], "world_pos": None, "neighbors": []}
        nb_cw = camera[10]  # Image obtained by rotating clockwise once

        image_paths = []
        depth_paths = []
        world_pos = []

        # Process neighbors of the current image
        while nb_cw != image:
            all_images.remove(nb_cw)
            nb_cam = images_to_camera[nb_cw]
            dirx, dirz = nb_cam[4][0], nb_cam[4][2]
            angle = math.atan2(dirz, dirx)  # Heading angle
            node["views"].append(View(nb_cw, angle, nb_cam).get())
            world_pos.append(nb_cam[3])
            image_paths.append(os.path.join(scene_path, "jpg_rgb", nb_cw))
            depth_paths.append(
                os.path.join(
                    scene_path, "high_res_depth", nb_cw.replace("01.jpg", "03.png")
                )
            )
            scene["images_to_nodes"][nb_cw] = node_idx
            nb_cw = nb_cam[10]

        # Position of the node is the mean of positions of all views in the node
        node["world_pos"] = np.stack(world_pos, axis=0).mean(axis=0).tolist()

        # Process the current image
        dirx, dirz = camera[4][0], camera[4][2]
        angle = math.atan2(dirz, dirx)
        node["views"].append(View(image, angle, camera).get())
        all_images.remove(image)
        image_paths.append(os.path.join(scene_path, "jpg_rgb", image))
        depth_paths.append(
            os.path.join(
                scene_path, "high_res_depth", image.replace("01.jpg", "03.png")
            )
        )
        scene["images_to_nodes"][image] = node_idx

        image_paths_per_node.append(image_paths)
        depth_paths_per_node.append(depth_paths)
        # Ensure all 12 directions are present
        assert len(node["views"]) == 12
        # Ensure no duplicates
        assert len(set([n["image_name"] for n in node["views"]])) == 12

        scene["nodes"].append(node)

    # Read all RGB and depth images
    images_per_node = pool.map(read_images_of_node_rgb, image_paths_per_node)
    depths_per_node = pool.map(read_images_of_node_depth, depth_paths_per_node)

    return scene, images_per_node, depths_per_node


def is_connected(node_1, node_2):
    ans = False
    for view_1 in node_1["views"]:
        for view_2 in node_2["views"]:
            if (
                view_1["camera"][12] == view_2["image_name"]
                or view_1["camera"][13] == view_2["image_name"]
            ):
                ans = True
                break
        if ans:
            break
    return ans


def create_scene_graph(scene):
    """
    Generate a connectivity graph for the scene where adjacent nodes
    are connected in an undirected fashion. Motion will be simulated 
    using this connectivity graph.

    scene - dictionary with keys as scale, nodes, scene_path
    """
    for idx in range(len(scene["nodes"]) - 1):
        for jdx in range(idx + 1, len(scene["nodes"])):
            loci = np.array(scene["nodes"][idx]["world_pos"])
            locj = np.array(scene["nodes"][jdx]["world_pos"])
            # if np.linalg.norm(loci - locj) > 3:
            #    continue
            if is_connected(scene["nodes"][idx], scene["nodes"][jdx]):
                # Undirected connection
                scene["nodes"][idx]["neighbors"].append(jdx)
                scene["nodes"][jdx]["neighbors"].append(idx)


def plot_scene(scene):
    """
    scene - dictionary with keys as scale, nodes, scene_path
    """
    scale = scene["scale"]
    fig, ax = plt.subplots(1)
    for node in scene["nodes"]:
        for view in node["views"]:
            camera = view["camera"]
            world_pos = camera[3]
            direction = (world_pos + camera[4] / 2) * scale
            plt.plot(world_pos[0] * scale, world_pos[2] * scale, "ro")
            # plt.plot([world_pos[0]*scale, direction[0]],
            #         [world_pos[2]*scale, direction[2]], 'b-')
        plt.plot(node["world_pos"][0] * scale, node["world_pos"][2] * scale, "bx")

    plt.axis("equal")
    plt.show()


def plot_connectivity(scene):
    """
    scene - dictionary with keys as scale, nodes, scene_path
    """
    scale = scene["scale"]
    fig, ax = plt.subplots(1)
    for node in scene["nodes"]:
        world_pos = node["world_pos"]
        plt.plot(world_pos[0], world_pos[2], "ro")
        for nb in node["neighbors"]:
            nb_world_pos = scene["nodes"][nb]["world_pos"]
            plt.plot(
                [world_pos[0], nb_world_pos[0]], [world_pos[2], nb_world_pos[2]], "b-"
            )

    plt.axis("equal")
    plt.show()


def main(args):
    all_scenes = []
    h5file = h5py.File(
        os.path.join(args.root_dir, "processed_images_{}x{}.h5".format(*RESOLUTION)),
        "w",
    )

    pool = Pool(128)
    for scene in all_scenes_list:
        scene_data, images_per_node, depths_per_node = process_scene(
            os.path.join(args.root_dir, scene), pool
        )
        create_scene_graph(scene_data)

        images_proc_scene = []
        for node_idx, images in enumerate(images_per_node):
            img_data = np.stack(images, axis=0)  # (N, H, W, C)
            images_proc_scene.append(img_data)

        depths_proc_scene = []
        for node_idx, depths in enumerate(depths_per_node):
            img_data = np.stack(depths, axis=0)[:, :, :, np.newaxis]  # (N, H, W, 1)
            depths_proc_scene.append(img_data)

        images_proc_scene = np.concatenate(images_proc_scene, axis=0)
        depths_proc_scene = np.concatenate(depths_proc_scene, axis=0)

        # Mapping from image_name to idx in images_proc_scene
        imgs_to_idx = {}
        count = 0
        for node in scene_data["nodes"]:
            for view in node["views"]:
                imgs_to_idx[view["image_name"]] = count
                count += 1

        h5file.create_dataset("{}/rgb".format(scene), data=images_proc_scene)
        h5file.create_dataset("{}/depth".format(scene), data=depths_proc_scene)

        scene_data["images_to_idx"] = imgs_to_idx
        all_scenes.append(scene_data)
        # Visualization of nodes in a scene
        # plot_connectivity(scene_data)

    h5file.close()

    np.save(
        os.path.join(args.root_dir, "processed_scenes_{}x{}.npy".format(*RESOLUTION)),
        all_scenes,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ActiveVisionDataset")
    args = parser.parse_args()

    main(args)
