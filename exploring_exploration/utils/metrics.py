#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import logging
import numpy as np

from exploring_exploration.utils.geometry import norm_cossin
from exploring_exploration.utils.common import norm_angle


def precision_at_k(pred_scores, gt_scores, k=5, gt_K=5):
    """
    Measures the fraction of correctly retrieved classes among the top-k
    retrievals.

    Inputs:
        pred_scores - (N, nclasses) logits
        gt_scores   - (N, nclasses) similarity scores
        k           - the top-k retrievals from pred_scores to consider
        gt_K        - how many of the most similar classes in gt_scores
                      should be considered as ground-truth

    Outputs:
        prec_scores - (N, ) precision@k scores per batch element
    """
    device = pred_scores.device
    N, nclasses = pred_scores.shape

    relevant_idxes = (torch.topk(gt_scores, gt_K, dim=1).indices).cpu()  # (N, gt_K)
    relevant_idxes_indicator = torch.zeros(N, nclasses)
    relevant_idxes_indicator.scatter_(1, relevant_idxes, 1.0)

    pred_idxes = (torch.topk(pred_scores, k, dim=1).indices).cpu()  # (N, k)
    pred_idxes_indicator = torch.zeros(N, nclasses)
    pred_idxes_indicator.scatter_(1, pred_idxes, 1.0)

    intersection_indicator = (
        relevant_idxes_indicator * pred_idxes_indicator
    )  # (N, nclasses)
    prec_scores = (intersection_indicator.sum(dim=1) / k).to(device)

    return prec_scores


def s_metric(agent_pos, target_pos, thresh, stop_called):
    # Success rate
    if not stop_called:
        return 0.0

    dist = np.linalg.norm(np.array(agent_pos) - np.array(target_pos)).item()
    score = 0.0 if dist > thresh else 1.0
    return score


def spl_metric(
    agent_pos, target_pos, thresh, path_length, shortest_path_length, stop_called
):
    # Success rate normalized by Path Length
    if not stop_called:
        return 0.0

    dist = np.linalg.norm(np.array(agent_pos) - np.array(target_pos)).item()
    score = (
        0.0
        if dist > thresh
        else shortest_path_length / (max(shortest_path_length, path_length) + 1e-7)
    )
    return score


def compute_pose_metrics(
    true_poses, pred_poses, true_pose_angles, pred_pose_angles, env_name
):
    """
    Inputs:
        true_poses - array of ground truth poses
        pred_poses - array of predicted poses
        true_pose_angles - array of ground truth heading angles
        pred_pose_angles - array of predicted heading angles
        env_name - name of current environment

    Outputs
        metrics     - a dictionary containing the different metrics measured
    """
    metrics = {}
    heading_err = np.abs(norm_angle(pred_pose_angles - true_pose_angles))
    avg_heading_err = math.degrees(heading_err.mean().item())
    heading_err_per_episode = np.degrees(heading_err)

    # Compute angular error
    norm_gt_pose = torch.Tensor(true_poses[:, 1:])
    norm_gt_angle = torch.atan2(norm_gt_pose[:, 1], norm_gt_pose[:, 0])
    norm_pred_pose = norm_cossin(torch.Tensor(pred_poses[:, 1:]))
    norm_pred_angle = torch.atan2(norm_pred_pose[:, 1], norm_pred_pose[:, 0])
    norm_ae = torch.abs(norm_angle(norm_pred_angle - norm_gt_angle))
    norm_ae_avg = math.degrees(norm_ae.cpu().mean().item())

    norm_ae_per_episode = np.degrees(norm_ae.cpu().numpy())

    # Compute distance prediction error
    distance_err = np.sqrt(((true_poses[:, 0] - pred_poses[:, 0]) ** 2))
    if "avd" not in env_name:
        distance_err = distance_err * 1000.0  # Convert to mm
    avg_distance_err = distance_err.mean()

    distance_err_per_episode = distance_err

    # Compute position error
    gt_r = torch.Tensor(true_poses[:, 0])
    gt_x = gt_r * torch.cos(norm_gt_angle)
    gt_y = gt_r * torch.sin(norm_gt_angle)
    pred_r = torch.Tensor(pred_poses[:, 0])
    pred_x = pred_r * torch.cos(norm_pred_angle)
    pred_y = pred_r * torch.sin(norm_pred_angle)
    position_err = torch.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    if "avd" not in env_name:
        position_err = position_err * 1000.0  # Convert to mm
    mean_position_err = position_err.mean().item()

    position_err_per_episode = position_err.cpu().numpy()

    # Compute position error, heading error as a function of difficulty
    difficulty_bins = list(range(500, 7000, 500))
    position_errors_vs_diff = []
    heading_errors_vs_diff = []
    heading_err = torch.Tensor(heading_err)
    for i in range(len(difficulty_bins) - 1):
        dl, dh = difficulty_bins[i], difficulty_bins[i + 1]
        if "avd" not in env_name:
            diff_mask = (gt_r * 1000.0 < dh) & (gt_r * 1000.0 >= dl)
        else:
            diff_mask = (gt_r < dh) & (gt_r >= dl)
        position_error_curr = position_err[diff_mask]
        heading_error_curr = heading_err[diff_mask]
        if diff_mask.sum() == 0:
            position_errors_vs_diff.append(0)
            heading_errors_vs_diff.append(0)
        else:
            position_errors_vs_diff.append(position_error_curr.mean())
            heading_errors_vs_diff.append(math.degrees(heading_error_curr.mean()))

    # Compute pose success rates at various thresholds
    success_thresholds = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]
    success_rates = [
        (position_err < sthresh).float().mean().item() for sthresh in success_thresholds
    ]

    logging.info("Success rates and thresholds:")
    logging.info(
        " | ".join(["{:6.0f}".format(sthresh) for sthresh in success_thresholds])
    )
    logging.info(" | ".join(["{:6.4f}".format(srate) for srate in success_rates]))

    logging.info("Position, heading errors at different difficulty levels:")
    logging.info(" || ".join(["{:6.2f}".format(dlevel) for dlevel in difficulty_bins]))
    logging.info(
        " || ".join(["{:6.2f}".format(perror) for perror in position_errors_vs_diff])
    )
    logging.info(
        " || ".join(["{:6.2f}".format(herror) for herror in heading_errors_vs_diff])
    )

    metrics["norm_ae"] = norm_ae_avg
    metrics["distance_err"] = avg_distance_err
    metrics["position_err"] = mean_position_err
    metrics["heading_err"] = avg_heading_err
    for thresh, rate in zip(success_thresholds, success_rates):
        metrics["success_rate @ {:.1f}".format(thresh)] = rate
    for level_0, level_1, err in zip(
        difficulty_bins[:-1], difficulty_bins[1:], position_errors_vs_diff
    ):
        metrics[
            "position_err @ distances b/w {:.1f} to {:.1f}".format(level_0, level_1)
        ] = err

    for level_0, level_1, err in zip(
        difficulty_bins[:-1], difficulty_bins[1:], heading_errors_vs_diff
    ):
        metrics[
            "heading_err @ distances b/w {:.1f} to {:.1f}".format(level_0, level_1)
        ] = err

    per_episode_metrics = {
        "heading_err": heading_err_per_episode,
        "norm_ae": norm_ae_per_episode,
        "position_err": position_err_per_episode,
    }
    return metrics, per_episode_metrics
