#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import math
import scipy.ndimage
import numpy as np
import networkx as nx


def norm_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def create_nav_graph(scan):
    """
    scan - dictionary with keys as nodes, ...

    nodes is a list with each node containing a list of neighbors
    """
    G = nx.Graph()

    def distance(pos1, pos2):
        # Returns Euclidean distance in 3D space
        return np.linalg.norm(pos1 - pos2)

    for nodeix, node in enumerate(scan["nodes"]):
        for nbrix in node["neighbors"]:
            nbr = scan["nodes"][nbrix]
            node_pos = np.array(node["world_pos"]) * scan["scale"]
            nbr_pos = np.array(nbr["world_pos"]) * scan["scale"]
            G.add_edge(nodeix, nbrix, weight=distance(node_pos, nbr_pos))

    return G


def draw_border(img, color=(255, 0, 0)):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)


def draw_triangle(img, loc1, loc2, loc3, color=(0, 255, 0)):
    triangle_cnt = np.array([loc1, loc2, loc3])
    cv2.drawContours(img, [triangle_cnt], 0, color, -1)


def draw_agent(image, position, pose, color, size=5):
    loc1 = (int(position[0] - size), int(position[1] - size))
    loc2 = (int(position[0]), int(position[1] + size))
    loc3 = (int(position[0] + size), int(position[1] - size))

    center = (int(position[0]), int(position[1]))
    loc4 = (
        int(center[0] + 2 * size * math.cos(pose)),
        int(center[1] + 2 * size * math.sin(pose)),
    )

    draw_triangle(image, loc1, loc2, loc3, color=color)
    image = cv2.line(image, center, loc4, (255, 255, 255), size // 2)
    return image


def draw_agent_sprite(image, position, pose, sprite, size=5):
    # Rotate before resize
    rotated_sprite = scipy.ndimage.interpolation.rotate(sprite, -pose * 180 / np.pi)
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite image should stay the same.
    initial_agent_size = sprite.shape[0]
    new_size = rotated_sprite.shape[0]

    # Rescale to a fixed size
    rotated_sprite = cv2.resize(
        rotated_sprite,
        (
            int(3 * size * new_size / initial_agent_size),
            int(3 * size * new_size / initial_agent_size),
        ),
    )

    # Add the rotated sprite to the image while ensuring boundary limits
    start_x = int(position[0]) - (rotated_sprite.shape[1] // 2)
    start_y = int(position[1]) - (rotated_sprite.shape[0] // 2)
    end_x = start_x + rotated_sprite.shape[1] - 1
    end_y = start_y + rotated_sprite.shape[0] - 1

    if start_x < 0:
        rotated_sprite = rotated_sprite[:, (-start_x):]
        start_x = 0
    elif end_x >= image.shape[1]:
        rotated_sprite = rotated_sprite[:, : (image.shape[1] - end_x - 1)]
        end_x = image.shape[1] - 1

    if start_y < 0:
        rotated_sprite = rotated_sprite[
            (-start_y):,
        ]
        start_y = 0
    elif end_y >= image.shape[0]:
        rotated_sprite = rotated_sprite[
            : (image.shape[0] - end_y - 1),
        ]
        end_y = image.shape[0] - 1

    alpha_mask = rotated_sprite[..., 2:3].astype(np.float32) / 255.0
    background = image[start_y : (end_y + 1), start_x : (end_x + 1)].astype(np.float32)
    foreground = rotated_sprite[..., :3].astype(np.float32)

    blended_sprite = cv2.add(foreground * alpha_mask, background * (1 - alpha_mask))
    blended_sprite = blended_sprite.astype(np.uint8)
    image[start_y : (end_y + 1), start_x : (end_x + 1)] = blended_sprite

    return image
