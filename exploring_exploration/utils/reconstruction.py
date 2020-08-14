#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from exploring_exploration.utils.common import (
    unflatten_two,
    flatten_two,
)
from exploring_exploration.utils.geometry import subtract_pose


def multi_label_classification_loss(x, y, reduction="batchmean"):
    """
    Multi-label classification loss - KL divergence between a uniform
    distribution over the GT classes and the predicted probabilities.
    Inputs:
        x - (bs, nclasses) predicted logits
        y - (bs, nclasses) with ones for the right classes and zeros
            for the wrong classes
    """
    x_logprob = F.log_softmax(x, dim=1)
    y_prob = F.normalize(
        y.float(), p=1, dim=1
    )  # L1 normalization to convert to probabilities
    loss = F.kl_div(x_logprob, y_prob, reduction=reduction)
    return loss


def rec_loss_fn_classify(
    x_logits, x_gt_feat, cluster_centroids, K=5, reduction="batchmean"
):
    """
    Given the predicted logits and ground-truth reference feature,
    find the top-K NN cluster centroids to the ground-truth feature.
    Using the top-k clusters as the ground-truth, use a multi-label
    classification loss.

    NOTE - this assumes that x_gt_feat and cluster_centroids are unit vectors.

    Inputs:
        x_logits - (bs, nclusters) predicted logits
        x_gt_feat - (bs, nclusters) reference feature that consists of
                    similarity scores between GT image and cluster centroids.
        cluster_centroids - (nclusters, feat_size) cluster centroids
    """
    bs, nclasses = x_logits.shape
    nclusters, feat_size = cluster_centroids.shape
    device = x_logits.device

    # Compute cosine similarity between x_gt_feat and cluster_centroids
    cosine_sim = x_gt_feat

    # Sample top-K similar clusters
    topK_outputs = torch.topk(cosine_sim, K, dim=1)

    # Generate K-hot encoding
    k_hot_encoding = (
        torch.zeros(bs, nclasses).to(device).scatter_(1, topK_outputs.indices, 1.0)
    )

    loss = multi_label_classification_loss(
        x_logits, k_hot_encoding, reduction=reduction
    )

    return loss


def compute_reconstruction_rewards(
    obs_feats,
    obs_odometer,
    tgt_feats,
    tgt_poses,
    cluster_centroids_t,
    decoder,
    pose_encoder,
):
    """
    Inputs:
        obs_feats           - (T, N, nclusters)
        obs_odometer        - (T, N, 3) --- (y, x, theta)
        tgt_feats           - (N, nRef, nclusters)
        tgt_poses           - (N, nRef, 3) --- (y, x, theta)
        cluster_centroids_t - (nclusters, feat_dim)
        decoder             - decoder model
        pose_encoder        - pose_encoder model

    Outputs:
        reward              - (N, nRef) float values indicating how many
                              GT clusters were successfully retrieved for
                              each target.
    """
    T, N, nclusters = obs_feats.shape
    nRef = tgt_feats.shape[1]
    device = obs_feats.device

    obs_feats_exp = obs_feats.unsqueeze(2)
    obs_feats_exp = obs_feats_exp.expand(
        -1, -1, nRef, -1
    ).contiguous()  # (T, N, nRef, nclusters)
    obs_odometer_exp = obs_odometer.unsqueeze(2)
    obs_odometer_exp = obs_odometer_exp.expand(
        -1, -1, nRef, -1
    ).contiguous()  # (T, N, nRef, 3)
    tgt_poses_exp = (
        tgt_poses.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    )  # (T, N, nRef, 3)

    # Compute relative poses
    obs_odometer_exp = obs_odometer_exp.view(T * N * nRef, 3)
    tgt_poses_exp = tgt_poses_exp.view(T * N * nRef, 3)
    obs_relpose = subtract_pose(
        obs_odometer_exp, tgt_poses_exp
    )  # (T*N*nRef, 3) --- (x, y, phi)

    # Compute pose encoding
    with torch.no_grad():
        obs_relpose_enc = pose_encoder(obs_relpose)  # (T*N*nRef, 16)
    obs_relpose_enc = obs_relpose_enc.view(T, N, nRef, -1)  # (T, N, nRef, 16)
    tgt_relpose_enc = torch.zeros(1, *obs_relpose_enc.shape[1:]).to(
        device
    )  # (1, N, nRef, 16)

    # Compute reconstructions
    obs_feats_exp = obs_feats_exp.view(T, N * nRef, nclusters)
    obs_relpose_enc = obs_relpose_enc.view(T, N * nRef, -1)
    tgt_relpose_enc = tgt_relpose_enc.view(1, N * nRef, -1)

    rec_inputs = {
        "history_image_features": obs_feats_exp,
        "history_pose_features": obs_relpose_enc,
        "target_pose_features": tgt_relpose_enc,
    }

    with torch.no_grad():
        pred_logits = decoder(rec_inputs)  # (1, N*nRef, nclusters)
    pred_logits = pred_logits.squeeze(0)  # (N*nRef, nclusters)
    pred_logits = unflatten_two(pred_logits, N, nRef)  # (N, nRef, nclusters)

    # Compute GT classes
    tgt_feats_sim = tgt_feats  # (N, nRef, nclusters)
    topk_gt = torch.topk(tgt_feats_sim, 5, dim=2)
    topk_gt_values = topk_gt.values  # (N, nRef, nclusters)
    topk_gt_thresh = topk_gt_values.min(dim=2).values  # (N, nRef)

    # ------------------ KL Div loss based reward --------------------
    reward = -rec_loss_fn_classify(
        flatten_two(pred_logits),
        flatten_two(tgt_feats),
        cluster_centroids_t.t(),
        K=2,
        reduction="none",
    ).sum(
        dim=1
    )  # (N*nRef, )
    reward = reward.view(N, nRef)

    return reward


def masked_mean(values, masks, axis=None):
    return (values * masks).sum(axis=axis) / (masks.sum(axis=axis) + 1e-10)
