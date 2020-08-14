#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from exploring_exploration.utils.median_pooling import MedianPool1d
from exploring_exploration.utils.common import (
    unsq_exp,
    flatten_two,
    unflatten_two,
)
from exploring_exploration.utils.geometry import (
    xyt2polar,
    polar2xyt,
    process_poseref,
    process_poseref_raw,
    position_loss_fn,
    add_pose,
)
from einops import rearrange, repeat


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def get_pose_criterion():
    # pred won't be normalized, gt will be normalized cos, sin values
    pose_loss_fn = lambda pred, gt: position_loss_fn(pred, gt)
    return pose_loss_fn


def get_pose_label_shape():
    lab_shape = (3,)
    return lab_shape


def compute_pose_sptm(
    obs_feats_sim,
    obs_feats_pose,
    obs_odometer,
    ref_feats_sim,
    ref_feats_pose,
    config,
    models,
    device,
    env_name,
):
    """
    Inputs:
        obs_feats_sim  - (T, N, feat_size_sim)
        obs_feats_pose - (T, N, feat_size_pose)
        obs_odometer   - (T, N, 4)
        ref_feats_sim  - (N, nRef, feat_size_sim)
        ref_feats_pose - (N, nRef, feat_size_pose)

    Outputs:
        predicted_poses - (N, nref, num_poses) for ref_feats based
        on past T observations
    """

    map_shape = config["map_shape"]
    map_scale = config["map_scale"]
    bin_size = config["bin_size"]
    angles = config["angles"]
    median_filter_size = config["median_filter_size"]
    vote_kernel_size = config["vote_kernel_size"]
    match_thresh = config["match_thresh"]

    rnet = models["rnet"]
    posenet = models["posenet"]  # Pairwise pose predictor
    pose_head = models["pose_head"]

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=vote_kernel_size, sigma=2.5, channels=1
    )
    gaussian_kernel = gaussian_kernel.to(device)

    # ========== Compute features for similarity prediction ===========
    T, N = obs_feats_sim.shape[:2]
    nRef = ref_feats_sim.shape[1]
    feat_size_sim = obs_feats_sim.shape[2]
    feat_size_pose = obs_feats_pose.shape[2]

    # ======== Compute the positions of each observation on the map ===========
    # obs_odometer ---> (T, N, 4) ---> (y, x, phi_head, phi_elev)
    obs_poses = torch.index_select(
        obs_odometer, 2, torch.LongTensor([1, 0, 2]).to(device)
    )  # (T, N, 3) ---> (x, y, phi_head)

    # ========== Compute pairwise scores with all prior observations ==========
    all_pred_poses = []
    all_voting_maps = []
    all_topk_idxes = []
    all_paired_scores = []
    all_paired_poses_polar = []
    median_filter = MedianPool1d(median_filter_size, 1, median_filter_size // 2)
    median_filter.to(device)
    for t in range(T - 1, T):
        obs_feats_sim_curr = obs_feats_sim[: (t + 1)]  # (t+1, N, feat_size_sim)
        obs_feats_sim_curr = unsq_exp(
            obs_feats_sim_curr, nRef, dim=2
        )  # (t+1, N, nRef, feat_size_sim)
        ref_feats_sim_curr = unsq_exp(
            ref_feats_sim, t + 1, dim=0
        )  # (t+1, N, nRef, feat_size_sim)
        obs_feats_sim_curr = obs_feats_sim_curr.view(
            -1, feat_size_sim
        )  # ((t+1)*N*nRef, feat_size_sim)
        ref_feats_sim_curr = ref_feats_sim_curr.view(
            -1, feat_size_sim
        )  # ((t+1)*N*nRef, feat_size_sim)

        with torch.no_grad():
            paired_scores = rnet.compare(
                torch.cat([obs_feats_sim_curr, ref_feats_sim_curr], dim=1)
            )  # ((t+1)*N*nRef, 2)
            paired_scores = F.softmax(paired_scores, dim=1)[:, 1]  # ((t+1)*N*nRef, )
            paired_scores = paired_scores.view(t + 1, N * nRef)  # (t+1, N*nRef)
            # Apply median filtering
            paired_scores = rearrange(
                paired_scores, "t f -> f () t"
            )  # (N*nRef, 1, t+1)
            paired_scores = median_filter(paired_scores)  # (N*nRef, 1, t+1)
            paired_scores = rearrange(
                paired_scores, "f () t -> t n r", n=N
            )  # (t+1, N, nRef)
        # Top K matches
        k = min(paired_scores.shape[0], 10)
        topk_scores, topk_idx = torch.topk(paired_scores, k=k, dim=0)  # (k, N, nRef)

        # Compute pose predictions for each match
        obs_poses_curr = obs_poses[: (t + 1)]  # (t+1, N, 3)
        obs_poses_curr = unsq_exp(obs_poses_curr, nRef, dim=2)  # (t+1, N, nRef, 3)
        obs_feats_pose_curr = unsq_exp(
            obs_feats_pose[: (t + 1)], nRef, dim=2
        )  # (t+1, N, nRef, feat_size_pose)
        topk_idx_exp = unsq_exp(
            topk_idx, feat_size_pose, dim=3
        )  # (k, N, nRef, feat_size_pose)
        topk_obs_feats_pose = torch.gather(
            obs_feats_pose_curr, 0, topk_idx_exp
        )  # (k, N, nRef, feat_size_pose)
        topk_idx_exp = topk_idx.unsqueeze(3).expand(-1, -1, -1, 3)
        topk_obs_poses = torch.gather(
            obs_poses_curr, 0, topk_idx_exp
        )  # (k, N, nRef, 3)

        ref_feats_pose_k = unsq_exp(
            ref_feats_pose, k, dim=0
        )  # (k, N, nRef, feat_size_pose)
        topk_obs_feats_pose = topk_obs_feats_pose.view(
            -1, feat_size_pose
        )  # (k * N * nRef, feat_size_pose)
        topk_obs_poses = topk_obs_poses.view(-1, 3)  # (k * N * nRef, 3)
        ref_feats_pose_k = ref_feats_pose_k.view(
            -1, feat_size_pose
        )  # (k * N * nRef, feat_size_pose)

        with torch.no_grad():
            # (k * N * nRef, 3) ---> delta_x, delta_y, delta_theta
            pred_dposes = posenet.get_pose_xyt_feats(
                topk_obs_feats_pose, ref_feats_pose_k
            )
            if "avd" in env_name:
                pred_dposes[:, :2] *= 1000.0  # (m -> mm)

        # Convert pred_dposes from observation centric coordinate
        # to the world coordinates.
        # (k * N * nRef, 3) ---> delta_x, delta_y, delta_theta
        pred_dposes_polar = xyt2polar(pred_dposes)
        pred_dposes_polar[:, 1] += topk_obs_poses[
            :, 2
        ]  # add the observation's world heading
        pred_dposes_world = polar2xyt(
            pred_dposes_polar
        )  # Convert delta pose to world coordinate
        pred_poses_world = (
            pred_dposes_world + topk_obs_poses
        )  # Get real world pose in xyt system
        all_pred_poses.append(pred_poses_world.view(k, N, nRef, 3))

        # Create the voting map based on the predicted poses
        pred_poses_polar = xyt2polar(
            pred_poses_world
        )  # Convert to (r, phi, theta) coordinates
        pred_poses_map = process_poseref(
            pred_poses_polar, map_shape, map_scale, angles, bin_size / 2
        ).long()  # (k * N * nRef, 3)
        pred_poses_oh = torch.zeros(k * N * nRef, *map_shape).to(
            device
        )  # (k * N * nRef, 1, mh, mw)
        pred_poses_oh[
            range(k * N * nRef), 0, pred_poses_map[:, 1], pred_poses_map[:, 0]
        ] = 1
        with torch.no_grad():
            pred_poses_smooth = gaussian_kernel(pred_poses_oh)
        pred_poses_smooth = pred_poses_smooth.view(
            k, N, nRef, *map_shape
        )  # (k, N, nRef, 1, mh, mw)

        # Top K matches filtered by match threshold
        thresh_filter = (topk_scores > match_thresh).float()  #  (k, N, nRef)
        thresh_filter = rearrange(thresh_filter, "k n r -> k n r () () ()")
        pred_poses_smooth = pred_poses_smooth * thresh_filter

        voting_map = pred_poses_smooth  # (k, N, nRef, 1, mh, mw)

        all_voting_maps.append(voting_map)
        all_topk_idxes.append(topk_idx)
        all_paired_scores.append(paired_scores)
        all_paired_poses_polar.append(pred_poses_polar.cpu().view(k, N, nRef, 3))

    all_pred_poses = torch.cat(all_pred_poses, dim=1)  # (k, N, nRef, 3)
    all_voting_maps = torch.cat(all_voting_maps, dim=1)  # (k, N, nRef, 1, mh, mw)
    all_topk_idxes = torch.cat(all_topk_idxes, dim=0)  # (10, N, nRef)
    all_paired_scores = torch.cat(all_paired_scores, dim=0)  # (T, N, nRef)
    all_paired_poses_polar = torch.cat(all_paired_poses_polar, dim=0)  # (k, N, nRef, 3)

    # ========== Predict pose ============
    all_voting_maps = rearrange(
        all_voting_maps, "k n r c h w -> k (n r) c h w"
    )  # (k, N*nRef, 1, mh, mw)
    predicted_poses, novote_masks = pose_head.forward(
        all_voting_maps
    )  # (N*nRef, num_poses)
    predicted_poses = unflatten_two(predicted_poses, N, nRef)  # (N, nRef, num_poses)
    novote_masks = unflatten_two(novote_masks, N, nRef)  # (N, nRef)

    all_pred_pose_angles = (
        all_pred_poses[:, :, :, 2].view(k, N * nRef).unsqueeze(2)
    )  # (k, N*nRef, 1)
    predicted_positions = pose_head.get_position_and_pose(
        all_voting_maps, all_pred_pose_angles
    )
    predicted_positions = unflatten_two(
        predicted_positions, N, nRef
    )  # (N, nRef, 3) ---> (x, y, t)

    outputs = {}
    outputs["predicted_poses"] = predicted_poses  # (N, nRef, num_poses)
    outputs["predicted_positions"] = predicted_positions  # (N, nRef, 3)
    outputs["novote_masks"] = novote_masks  # (N, nRef)

    return outputs


def compute_pose_sptm_ransac(
    obs_feats_sim,
    obs_feats_pose,
    obs_odometer,
    ref_feats_sim,
    ref_feats_pose,
    config,
    models,
    device,
    env_name,
):
    """
    Given a history of observations in the form of features, odometer readings,
    estimate the location of a set of references given their features.

    Inputs:
        obs_feats_sim  - (T, N, feat_size_sim)
        obs_feats_pose - (T, N, feat_size_pose)
        obs_odometer   - (T, N, 4)
        ref_feats_sim  - (N, nRef, feat_size_sim)
        ref_feats_pose - (N, nRef, feat_size_pose)

    Outputs:
        predicted_poses      - (N, nRef, num_poses)
        predicted_positions  - (N, nRef, 3)
        novote_masks         - (N, nRef)
        all_paired_poses_map - (T, N, nRef, 3)
        obs_poses            - (T, N, 3)
        inlier_mask          - (T, N, nRef)
        all_pairwise_scores  - (T, N, nRef)
        voting_maps          - (N, nRef, 1, mh, mw)
    """

    map_shape = config["map_shape"]
    map_scale = config["map_scale"]
    bin_size = config["bin_size"]
    angles = config["angles"]
    median_filter_size = config["median_filter_size"]
    match_thresh = config["match_thresh"]

    rnet = models["rnet"]
    posenet = models["posenet"]  # Pairwise pose predictor
    pose_head = models["pose_head"]
    ransac_estimator = models["ransac_estimator"]

    # ========== Compute features for similarity prediction ===========
    T, N = obs_feats_sim.shape[:2]
    nRef = ref_feats_sim.shape[1]
    feat_size_sim = obs_feats_sim.shape[2]

    # ========== Compute the positions of each observation on the map ============
    # obs_odometer ---> (T, N, 4) ---> (y, x, phi_head, phi_elev)
    obs_poses = torch.index_select(
        obs_odometer, 2, torch.LongTensor([1, 0, 2]).to(device)
    )  # (T, N, 3) ---> (x, y, phi_head)

    # ========== Compute pairwise similarity with all prior observations ============
    median_filter = MedianPool1d(median_filter_size, 1, median_filter_size // 2)
    median_filter.to(device)
    obs_feats_sim = repeat(obs_feats_sim, "t n f -> (t n r) f", r=nRef)
    ref_feats_sim = repeat(ref_feats_sim, "n r f -> (t n r) f", t=T)
    with torch.no_grad():
        paired_scores = rnet.compare(torch.cat([obs_feats_sim, ref_feats_sim], dim=1))
    paired_scores = F.softmax(paired_scores, dim=1)[:, 1]  # (T*N*nRef, )
    paired_scores = rearrange(paired_scores, "(t n r) -> t (n r)", t=T, n=N)
    if paired_scores.shape[0] > 1:
        # Apply median filtering
        paired_scores = rearrange(paired_scores, "t nr -> nr () t")
        paired_scores = median_filter(paired_scores)
        paired_scores = rearrange(paired_scores, "nr () t -> t nr")
    paired_scores = rearrange(paired_scores, "t (n r) -> t n r", t=T, n=N)

    # ========== Compute pairwise poses with all prior observations ============
    ref_feats_pose = repeat(ref_feats_pose, "n r f -> (t n r) f", t=T)
    obs_feats_pose = repeat(obs_feats_pose, "t n f -> (t n r) f", r=nRef)
    with torch.no_grad():
        pairwise_dposes = posenet.get_pose_xyt_feats(obs_feats_pose, ref_feats_pose)
        if "avd" in env_name:
            pairwise_dposes[:, :2] *= 1000.0  # (m -> mm)

    # ============= Add pairwise delta to observation pose ==============
    obs_poses_rep = repeat(obs_poses, "t n p -> (t n r) p", r=nRef)
    pairwise_poses_world = add_pose(obs_poses_rep, pairwise_dposes, mode="xyt")
    pairwise_poses_world = pairwise_poses_world.view(T, N, nRef, 3)  # (x, y, t)

    # ========== Define similarity weighted sampling function ==========
    paired_scores = rearrange(paired_scores, "t n r -> t (n r)")
    # When no samples fall above the match_thresh, set match_thresh to a lower value
    match_thresh_mask = (paired_scores > match_thresh).sum(dim=0) == 0  # (N*nRef, )
    batch_match_thresh = torch.ones(N * nRef).to(device) * match_thresh
    # If any element has zero samples above match threshold
    if match_thresh_mask.sum().item() > 0:
        batch_match_thresh[match_thresh_mask] = (
            paired_scores[:, match_thresh_mask].max(dim=0)[0] - 0.001
        )  # (N*nRef, )
    batch_match_thresh = batch_match_thresh.unsqueeze(0)  # (1, N*nRef)
    # Compute mask indicating validity of samples along time
    valid_masks = paired_scores > batch_match_thresh  # (T, N*nRef)

    # Assign zero weights to observations below a matching threshold
    sample_weights = (
        paired_scores * (paired_scores > batch_match_thresh).float()
    )  # (T, N*nRef)
    pairwise_poses_world = rearrange(pairwise_poses_world, "t n r p -> t (n r) p")
    (
        pred_pose_inliers,
        pred_position_inliers,
        voting_map_inliers,
        inlier_mask,
    ) = ransac_estimator.ransac_pose_estimation(
        pairwise_poses_world, sample_weights, valid_masks
    )
    novote_masks = match_thresh_mask
    # pred_pose_inliers - (N*nRef, num_poses), pred_position_inliers - (N*nRef, 3)
    # novote_masks - (N*nRef, ), voting_map_inliers - (N*nRef, 1, mh, mw)
    pairwise_poses_map = ransac_estimator.polar2map(
        xyt2polar(flatten_two(pairwise_poses_world))
    )  # (T*N*nRef, 3)
    pred_pose_inliers = unflatten_two(
        pred_pose_inliers, N, nRef
    )  # (N, nRef, num_poses)
    pred_position_inliers = unflatten_two(
        pred_position_inliers, N, nRef
    )  # (N, nRef, 3)
    pairwise_poses_map = pairwise_poses_map.view(T, N, nRef, 3)  # (T, N, nRef, 3)
    obs_poses = obs_poses  # (T, N, 3)
    novote_masks = novote_masks.view(N, nRef)  # (N, nRef)
    inlier_mask = inlier_mask.view(T, N, nRef)  # (T, N, nRef)
    pairwise_scores = paired_scores.view(T, N, nRef)  # (T, N, nRef)
    final_voting_maps = unflatten_two(
        voting_map_inliers, N, nRef
    )  # (N, nRef, 1, mh, mw)

    outputs = {}
    outputs["predicted_poses"] = pred_pose_inliers
    outputs["predicted_positions"] = pred_position_inliers
    outputs["novote_masks"] = novote_masks
    outputs["all_paired_poses_map"] = pairwise_poses_map
    outputs["obs_poses"] = obs_poses
    outputs["inlier_mask"] = inlier_mask
    outputs["all_pairwise_scores"] = pairwise_scores
    outputs["voting_maps"] = final_voting_maps
    outputs[
        "successful_votes"
    ] = inlier_mask  # The observations which successfully voted

    return outputs


class RansacPoseEstimator:
    def __init__(self, config, pose_head, device):
        self.config = config
        self.device = device
        self.gaussian_kernel = get_gaussian_kernel(
            kernel_size=self.config["vote_kernel_size"], sigma=2.5, channels=1
        ).to(self.device)
        self.pose_head = pose_head

    def polar2map(self, pose_polar):
        """
        pose_polar - (bs, 3)
        """
        pose_map = process_poseref_raw(
            pose_polar,
            self.config["map_shape"],
            self.config["map_scale"],
            self.config["angles"],
            self.config["bin_size"] / 2,
        )
        return pose_map

    def map2votes(self, pose_map):
        """
        Input:
            pose_map - (bs, 3) - (mapx, mapy, theta) - theta in radians
        Output:
            vote_map - (bs, 1, mh, mw)
        """
        map_shape = self.config["map_shape"]
        pose_oh = torch.zeros(pose_map.shape[0], *map_shape, device=self.device)
        pose_oh[
            range(pose_map.shape[0]), 0, pose_map[:, 1].long(), pose_map[:, 0].long()
        ] = 1
        with torch.no_grad():
            vote_map = self.gaussian_kernel(pose_oh)
        return vote_map

    def estimate_pose(self, pairwise_poses_world):
        """
        Inputs:
            pairwise_poses_world - (K, bs, 3) - (x, y, theta) in world coordinates
        Outputs:
            final_pose    - (bs, 3) -- pose
            final_position- (bs, 3) -- (x, y, theta) in world coordinates
        """
        K, bs = pairwise_poses_world.shape[:2]
        # Create the voting map based on the predicted poses
        # Convert to (r, phi, theta) coordinates
        pred_poses_polar = xyt2polar(flatten_two(pairwise_poses_world))  # (K*bs, 3)
        pred_poses_map = self.polar2map(pred_poses_polar)  # (K*bs, 3)
        voting_maps = self.map2votes(pred_poses_map)  # (K*bs, 1, mh, mw)
        pairwise_angles = flatten_two(pairwise_poses_world[:, :, 2])  # (K*bs, )

        voting_maps = unflatten_two(voting_maps, K, bs)  # (K, bs, 1, mh, mw)
        pairwise_angles = pairwise_angles.view(K, bs, 1)  # (K, bs, 1)
        final_pose, _ = self.pose_head.forward(voting_maps)  # (bs, num_poses), (bs, )
        final_position = self.pose_head.get_position_and_pose(
            voting_maps, pairwise_angles
        )  # (bs, 3)
        return final_pose, final_position

    def estimate_pose_mask(self, pairwise_poses_world, masks):
        """
        Inputs:
            pairwise_poses_world - (K, bs, 3) - (x, y, theta) in world coordinates
            masks         - (K, bs) - binary indicating which of the K elements are to be considered
        Outputs:
            final_pose    - (bs, 3) -- pose
            final_position- (bs, 3) -- (x, y, theta) in world coordinates
            voting_map    - (bs, 1, mh, mw)

        """
        K, bs = pairwise_poses_world.shape[:2]
        # Create the voting map based on the predicted poses
        # Convert to (r, phi, theta) coordinates
        pred_poses_polar = xyt2polar(flatten_two(pairwise_poses_world))  # (K*bs, 3)
        pred_poses_map = self.polar2map(pred_poses_polar)  # (K*bs, 3)
        voting_maps = self.map2votes(pred_poses_map)  # (K*bs, 1, mh, mw)
        pairwise_angles = flatten_two(pairwise_poses_world[:, :, 2])  # (K*bs, )
        # Mask out the irrelevant samples
        voting_maps = voting_maps * masks.view(K * bs, 1, 1, 1)

        voting_maps = unflatten_two(voting_maps, K, bs)  # (K, bs, 1, mh, mw)
        pairwise_angles = pairwise_angles.view(K, bs, 1)  # (K, bs, 1)
        final_pose, _ = self.pose_head.forward(voting_maps)  # (bs, num_poses), (bs, )
        final_position = self.pose_head.get_position_and_pose(
            voting_maps, pairwise_angles
        )  # (bs, 3)
        voting_maps = voting_maps.sum(dim=0)  # (bs, 1, mh, mw)
        voting_maps = voting_maps + 1e-9  # Add small non-zero votes to all locations
        voting_sum = voting_maps.view(bs, -1).sum(dim=1)  # (bs, )
        voting_sum = voting_sum.view(bs, 1, 1, 1)  # (bs, 1, 1, 1)
        voting_map = voting_maps / voting_sum  # (bs, 1, mh, mw)

        return final_pose, final_position, voting_map

    def distance_fn_1(self, p1, p2):
        """
        p1 - (bs, 3) poses (x, y, theta)
        p2 - (T, bs, 3) poses (x, y, theta)
        returns distance in (x, y) coordinates
        """
        T = p2.shape[0]
        p1_rsz = repeat(p1, "b p -> t b p", t=T)
        return torch.norm(p1_rsz[:, :, :2] - p2[:, :, :2], dim=2)

    def distance_fn_2(self, p1, p2):
        """
        p1 - (bs, 3) poses (x, y, theta)
        p2 - (T, bs, 3) poses (x, y, theta)
        returns distance in theta
        """
        T = p2.shape[0]
        p1_rsz = repeat(p1, "b p -> t b p", t=T)
        diff = p1_rsz[:, :, 2] - p2[:, :, 2]
        diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))  # (T, bs)
        return diff

    def sample_points(self, sample_weights):
        """
        Inputs:
            sample_weights - (T, bs)
        Outputs:
            Returns (K, bs) integer indices for each of the input sequences
        """
        K = self.config["ransac_batch"]
        sample_weights_t = sample_weights.transpose(0, 1)
        idxes = torch.multinomial(sample_weights_t, K, replacement=True)  # (bs, K)
        return idxes.transpose(0, 1)

    def ransac_pose_estimation(self, pairwise_poses_world, sample_weights, valid_masks):
        """
        Inputs:
            pairwise_poses_world - (T, bs, 3)
            sample_weights - (T, bs)
            valid_masks - (T, bs)
        Outputs:
            pred_pose_inliers - (bs, num_poses)
            pred_position_inliers - (bs, 3)
            voting_map_inliers - (bs, 1, mh, mw)
        """
        bs = pairwise_poses_world.shape[1]
        best_poses = None
        best_positions = None
        best_inlier_counts = torch.zeros(bs).to(self.device).long()
        rthresh1 = self.config["ransac_theta_1"]
        rthresh2 = self.config["ransac_theta_2"]
        for _ in range(self.config["ransac_niter"]):
            # Sample a subset of samples from each sequence
            sample_idxes = self.sample_points(sample_weights)  # (K, bs)
            sample_idxes = repeat(sample_idxes, "k b -> k b p", p=3)
            pairwise_poses_curr = torch.gather(pairwise_poses_world, 0, sample_idxes)
            # Estimate pose from subset
            pred_poses, pred_positions = self.estimate_pose(pairwise_poses_curr)
            # Estimate consensus
            pair_dist_1 = self.distance_fn_1(
                pred_positions, pairwise_poses_world
            )  # (T, bs)
            pair_dist_2 = self.distance_fn_2(
                pred_positions, pairwise_poses_world
            )  # (T, bs)
            n_inliers = (
                (pair_dist_1 < rthresh1) & (pair_dist_2 < rthresh2) & valid_masks
            ).sum(
                dim=0
            )  # (bs, )
            # Keep track of best estimates so far
            update_mask = n_inliers > best_inlier_counts
            if best_poses is None:
                best_poses = pred_poses
                best_positions = pred_positions
                best_inlier_counts = n_inliers
            else:
                best_poses[update_mask] = pred_poses[update_mask]
                best_positions[update_mask] = pred_positions[update_mask]
                best_inlier_counts[update_mask] = n_inliers[update_mask]

        # Recompute the pose using all inliers
        pair_dist_1 = self.distance_fn_1(best_positions, pairwise_poses_world)
        pair_dist_2 = self.distance_fn_2(best_positions, pairwise_poses_world)
        inlier_mask = (
            (pair_dist_1 < rthresh1) & (pair_dist_2 < rthresh2) & valid_masks
        )  # (T, bs)
        (
            pred_pose_inliers,
            pred_position_inliers,
            voting_map_inliers,
        ) = self.estimate_pose_mask(
            pairwise_poses_world, inlier_mask.float(),
        )  # (bs, 3)

        return pred_pose_inliers, pred_position_inliers, voting_map_inliers, inlier_mask
