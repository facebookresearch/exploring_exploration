#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

approx_eq = lambda a, b, eps: torch.lt(torch.abs(a - b), eps)


def process_pose(pose):
    # pose - num_processes x 4 element Tensor with (r, theta, phi_head, phi_elev) - angles in radians
    # Output - num_processes x 3 torch tensor representing distance, cos and sin of relative theta
    pose_processed = torch.stack(
        (pose[:, 0], torch.cos(pose[:, 1]), torch.sin(pose[:, 1])), dim=1
    ).to(pose.device)
    return pose_processed


def process_poseref(pose, map_shape, map_scale, angles, eps):
    # pose - batch_size x 3 - (r, theta, head) of the reference view
    r = pose[:, 0]
    t = pose[:, 1]
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    mh, mw = map_shape[1:]
    # This convention comes from transform_to_map() in model_pose.py
    ref_on_map_x = torch.clamp(mw / 2 + x / map_scale, 0, mw - 1)
    ref_on_map_y = torch.clamp(mh / 2 + y / map_scale, 0, mh - 1)
    # Mapping heading angles to map locations
    ref_on_map_dir = torch.zeros(pose.shape[0]).to(pose.device)
    normalized_angles = torch.atan2(torch.sin(pose[:, 2]), torch.cos(pose[:, 2]))
    for i in range(angles.shape[0]):
        ref_on_map_dir[approx_eq(normalized_angles, angles[i].item(), eps)] = i
    return torch.stack([ref_on_map_x, ref_on_map_y, ref_on_map_dir], dim=1).long()


def process_poseref_raw(pose, map_shape, map_scale, angles, eps):
    # pose - batch_size x 3 - (r, theta, head) of the reference view
    r = pose[:, 0]
    t = pose[:, 1]
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    mh, mw = map_shape[1:]
    # This convention comes from transform_to_map() in model_pose.py
    ref_on_map_x = torch.clamp(mw / 2 + x / map_scale, 0, mw - 1)
    ref_on_map_y = torch.clamp(mh / 2 + y / map_scale, 0, mh - 1)
    normalized_angles = torch.atan2(torch.sin(pose[:, 2]), torch.cos(pose[:, 2]))
    return torch.stack([ref_on_map_x, ref_on_map_y, normalized_angles], dim=1)


def position_loss_fn(pred, gt):
    """
    pred - (bs, 3) ---> (r, cos_phi, sin_phi)
    gt   - (bs, 3) ---> (r, cos_phi, sin_phi)
    pred won't be normalized, gt will be normalized cos, sin values
    """
    pred_cossin = norm_cossin(pred[:, 1:])
    gt_cossin = gt[:, 1:]
    pred_r = pred[:, 0]
    gt_r = gt[:, 0]
    pred_x = pred_r * pred_cossin[:, 0]
    pred_y = pred_r * pred_cossin[:, 1]
    gt_x = gt_r * gt_cossin[:, 0]
    gt_y = gt_r * gt_cossin[:, 1]
    loss = (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2
    return loss


def norm_cossin(input):
    """Convert unnormalized cos, sin predictions into [0, 1] range.
    """
    # Normalize cos, sin predictions
    if isinstance(input, torch.Tensor):
        input = input / (torch.norm(input, dim=1).unsqueeze(1) + 1e-8)
    elif isinstance(input, np.ndarray):
        input = input / (np.linalg.norm(input, axis=1)[:, np.newaxis] + 1e-8)
    else:
        raise ValueError("Incorrect type for norm_cossin!")

    return input


def process_odometer(poses):
    """Converts odometer readings in polar coordinates to xyt coordinates.

    Inputs:
        pose - (bs, 4) Tensor with (r, theta, phi_head, phi_elev)
             - where angles are in radians
    Outputs:
        pose_processed - (bs, 4) Tensor with (y, x, phi_head, phi_elev)
    """
    pose_processed = torch.stack(
        [
            poses[:, 0] * torch.sin(poses[:, 1]),
            poses[:, 0] * torch.cos(poses[:, 1]),
            poses[:, 2],
            poses[:, 3],
        ],
        dim=1,
    )
    return pose_processed


def np_normalize(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


def xyt2polar(poses):
    """Converts poses from carteisan (xyt) to polar (rpt) coordinates.

    Inputs:
        poses - (bs, 3) Tensor --- (x, y, theta)
    Outputs:
        poses Tensor with (r, phi, theta) conventions
    """
    return torch.stack(
        [
            torch.norm(poses[:, :2], dim=1),  # r
            torch.atan2(poses[:, 1], poses[:, 0]),  # phi
            poses[:, 2],
        ],
        dim=1,
    )


def polar2xyt(poses):
    """Converts poses from polar (rpt) to cartesian (xyt) coordinates.

    Inputs:
        poses - (bs, 3) Tensor --- (r, phi, theta)
    Outputs:
        poses Tensor with (x, y, theta) conventions
    """
    return torch.stack(
        [
            poses[:, 0] * torch.cos(poses[:, 1]),  # x
            poses[:, 0] * torch.sin(poses[:, 1]),  # y
            poses[:, 2],
        ],
        dim=1,
    )


def compute_egocentric_coors(delta, prev_pos, scale):
    """
    delta - (N, 4) --- (y, x, phi_head, phi_elev)
    prev_pos - (N, 4) --- (y, x, phi_head, phi_elev)
    """
    dy, dx, dt = delta[:, 0], delta[:, 1], delta[:, 2]
    x, y, t = prev_pos[:, 0], prev_pos[:, 1], prev_pos[:, 2]
    dr = torch.sqrt(dx ** 2 + dy ** 2)
    dp = torch.atan2(dy, dx) - t
    dx_ego = dr * torch.cos(dp) / scale
    dy_ego = dr * torch.sin(dp) / scale
    dt_ego = dt

    return torch.stack([dx_ego, dy_ego, dt_ego], dim=1)


def subtract_pose(pose_common, poses):
    """
    Convert poses to frame-of-reference of pose_common.

    Inputs:
        pose_common - (N, 3) --- (y, x, phi)
        poses - (N, 3) --- (y, x, phi)

    Outputs:
        poses_n - (N, 3) --- (x, y, phi) in the new coordinate system
    """

    x = poses[:, 1]
    y = poses[:, 0]
    phi = poses[:, 2]

    x_c = pose_common[:, 1]
    y_c = pose_common[:, 0]
    phi_c = pose_common[:, 2]

    # Polar coordinates in the new frame-of-reference
    r_n = torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    theta_n = torch.atan2(y - y_c, x - x_c) - phi_c
    # Convert to cartesian coordinates
    x_n = r_n * torch.cos(theta_n)
    y_n = r_n * torch.sin(theta_n)
    phi_n = phi - phi_c
    # Normalize phi to lie between -pi to pi
    phi_n = torch.atan2(torch.sin(phi_n), torch.cos(phi_n))

    poses_n = torch.stack([x_n, y_n, phi_n], dim=1)

    return poses_n


def add_pose(pose_common, dposes, mode="yxt"):
    """
    Convert dposes from frame-of-reference of pose_common to global pose.

    Inputs:
        pose_common - (N, 3)
        dposes - (N, 3)

    Outputs:
        poses - (N, 3)
    """

    assert mode in ["xyt", "yxt"]

    if mode == "yxt":
        dy, dx, dt = torch.unbind(dposes, dim=1)
        y_c, x_c, t_c = torch.unbind(pose_common, dim=1)
    else:
        dx, dy, dt = torch.unbind(dposes, dim=1)
        x_c, y_c, t_c = torch.unbind(pose_common, dim=1)

    dr = torch.sqrt(dx ** 2 + dy ** 2)
    dphi = torch.atan2(dy, dx) + t_c
    x = x_c + dr * torch.cos(dphi)
    y = y_c + dr * torch.sin(dphi)
    t = t_c + dt
    # Normalize angles to lie between -pi to pi
    t = torch.atan2(torch.sin(t), torch.cos(t))

    if mode == "yxt":
        poses = torch.stack([y, x, t], dim=1)
    else:
        poses = torch.stack([x, y, t], dim=1)

    return poses
