#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def torch_to_np(image):
    image = (image.cpu().numpy()).transpose(1, 2, 0)
    image = image.astype(np.uint8)
    image = np.flip(image, axis=2)
    return image


def torch_to_np_depth(image, max_depth=10000.0):
    depth = (image.cpu().numpy())[0]
    depth = (np.clip(depth, 0, max_depth) / max_depth) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=2)
    return depth


class TensorboardWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into tensorboard from images frames.
        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.
        Returns:
            None.
        """
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.add_video(video_name, video_tensor, fps=fps, global_step=step_idx)


def write_video(frames, path, fps=10.0, video_format="MP4V"):
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    shape = frames[0].shape[:2][::-1]  # (WIDTH, HEIGHT)
    vidwriter = cv2.VideoWriter(path, fourcc, fps, shape)
    for frame in frames:
        vidwriter.write(frame[:, :, ::-1])  # Convert to BGR
    vidwriter.release()


def create_reference_grid(refs_uint8):
    """
    Inputs:
        refs_uint8 - (nRef, H, W, C) numpy array
    """
    refs_uint8 = np.copy(refs_uint8)
    nRef, H, W, C = refs_uint8.shape

    nrow = int(math.sqrt(nRef))

    ncol = nRef // nrow  # (number of images per column)
    if nrow * ncol < nRef:
        ncol += 1
    final_grid = np.zeros((nrow * ncol, *refs_uint8.shape[1:]), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    final_grid[:nRef] = refs_uint8
    final_grid = final_grid.reshape(
        ncol, nrow, *final_grid.shape[1:]
    )  # (ncol, nrow, H, W, C)
    final_grid = final_grid.transpose(0, 2, 1, 3, 4)
    final_grid = final_grid.reshape(ncol * H, nrow * W, C)
    return final_grid


def draw_border(images, color=(255, 0, 0), thickness=5):
    """Draw image border.

    Inputs:
        images - (N, H, W, C) numpy array
    """
    images[:, :thickness, :, 0] = color[0]
    images[:, :thickness, :, 1] = color[1]
    images[:, :thickness, :, 2] = color[2]

    images[:, -thickness:, :, 0] = color[0]
    images[:, -thickness:, :, 1] = color[1]
    images[:, -thickness:, :, 2] = color[2]

    images[:, :, :thickness, 0] = color[0]
    images[:, :, :thickness, 1] = color[1]
    images[:, :, :thickness, 2] = color[2]

    images[:, :, -thickness:, 0] = color[0]
    images[:, :, -thickness:, 1] = color[1]
    images[:, :, -thickness:, 2] = color[2]

    return images
