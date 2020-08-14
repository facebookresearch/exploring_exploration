#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from exploring_exploration.utils.geometry import subtract_pose
from exploring_exploration.utils.common import (
    flatten_two,
    unflatten_two,
    unsq_exp,
)
from einops import rearrange


class SupervisedReconstruction:
    """Algorithm to learn a reconstruction task-head that reconstructs
    features at a new target location given features from existing locations.

    The default loss function is a multilabel loss. Existing features from
    training environments are clustered into K clusters. For each target
    location, the nearest J clusters are defined as positives and the
    remaining K-J clusters are defined as negatives. The model is required to
    identify these top-J clusters / "reconstruction" concepts.
    """

    def __init__(self, config):
        self.decoder = config["decoder"]
        self.pose_encoder = config["pose_encoder"]
        lr = config["lr"]
        eps = config["eps"]
        self.max_grad_norm = config["max_grad_norm"]
        # The loss function is passed as an argument. The default loss is a
        # multi-label loss.
        self.rec_loss_fn = config["rec_loss_fn"]
        # The number of nearest neighbors to use as positives.
        self.rec_loss_fn_J = config["rec_loss_fn_J"]
        self.cluster_centroids = config["cluster_centroids"]
        # Make a prediction at regular intervals of this size.
        self.prediction_interval = config["prediction_interval"]
        self.optimizer = optim.Adam(
            itertools.chain(self.decoder.parameters(), self.pose_encoder.parameters(),),
            lr=lr,
            eps=eps,
        )

    def update(self, rollouts):
        T, N, nfeats = rollouts.obs_feats[:-1].shape
        nRef = rollouts.tgt_feats.shape[1]
        device = rollouts.obs_feats.device
        avg_loss = 0.0
        avg_loss_count = 0.0
        tgt_feats = rollouts.tgt_feats  # (N, nRef, nfeats)
        tgt_masks = rollouts.tgt_masks.squeeze(2)  # (N, nRef)
        obs_feats = unsq_exp(rollouts.obs_feats, nRef, dim=2)  # (T+1, N, nRef, nfeats)
        obs_poses = unsq_exp(
            rollouts.obs_odometer[:, :, :3], nRef, dim=2
        )  # (T+1, N, nRef, 3) - (y, x, phi)
        tgt_poses = unsq_exp(rollouts.tgt_poses, T + 1, dim=0)  # (T+1, N, nRef, 3)
        # Make a prediction after every prediction_interval steps, i.e.,
        # the agent has seen self.prediction_interval*(i+1) observations.
        for i in range(0, T, self.prediction_interval):
            L = min(i + self.prediction_interval, T)
            # Estimate relative pose b/w targets and observations.
            obs_relpose = subtract_pose(
                rearrange(tgt_poses[:L], "l b n f -> (l b n) f"),
                rearrange(obs_poses[:L], "l b n f -> (l b n) f"),
            )  # (L*N*nRef, 3) --- (x, y, phi)
            # ========================= Forward pass ==========================
            # Encode the poses of the observations and targets.
            obs_relpose_enc = self.pose_encoder(obs_relpose)  # (L*N*nRef, 16)
            obs_relpose_enc = obs_relpose_enc.view(L, N * nRef, -1)
            tgt_relpose_enc = torch.zeros(1, *obs_relpose_enc.shape[1:]).to(device)
            obs_feats_i = rearrange(obs_feats[:L], "l b n f -> l (b n) f")
            # These serve as inputs to an encoder-decoder transformer model.
            rec_inputs = {
                # encoder inputs
                "history_image_features": obs_feats_i,  # (L, N*nRef, nfeats)
                "history_pose_features": obs_relpose_enc,  # (L, N*nRef, 16)
                # decoder inputs
                "target_pose_features": tgt_relpose_enc,  # (1, N*nRef, 16)
            }
            pred_logits = self.decoder(rec_inputs).squeeze(0)  # (N*nRef, nclass)
            # =================== Compute reconstruction loss =================
            loss = self.rec_loss_fn(
                pred_logits,  # (N*nRef, nclass)
                flatten_two(tgt_feats),  # (N*nRef, nfeats)
                self.cluster_centroids,
                K=self.rec_loss_fn_J,
                reduction="none",
            ).sum(
                dim=1
            )  # (N*nRef, )
            loss = unflatten_two(loss, N, nRef)
            # Mask out invalid targets.
            loss = loss * tgt_masks
            loss = loss.mean()
            # ========================== Backward pass ========================
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.decoder.parameters(), self.pose_encoder.parameters(),
                ),
                self.max_grad_norm,
            )
            self.optimizer.step()

            avg_loss += loss.item()
            avg_loss_count += 1.0

        avg_loss = avg_loss / avg_loss_count
        losses = {"rec_loss": avg_loss}
        return losses
