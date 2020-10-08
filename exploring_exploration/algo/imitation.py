#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

from exploring_exploration.utils.common import (
    flatten_two,
    unflatten_two,
)


def get_onehot_tensor(idxes, size):
    device = idxes.device
    bs = idxes.shape[0]
    oh = torch.zeros(bs, size).to(device).scatter_(1, idxes, 1)
    return oh


class Imitation:
    """Algorithm to learn policy from expert trajectories via
    imitation learning. Incorporates inflection weighting from
    https://arxiv.org/pdf/1904.03461.pdf.
    """

    def __init__(self, config):
        self.encoder = config["encoder"]
        self.actor_critic = config["actor_critic"]
        lr = config["lr"]
        eps = config["eps"]
        self.max_grad_norm = config["max_grad_norm"]
        self.nactions = config["nactions"]
        self.encoder_type = config["encoder_type"]
        self.use_action_embedding = config["use_action_embedding"]
        self.use_collision_embedding = config["use_collision_embedding"]
        self.use_inflection_weighting = config["use_inflection_weighting"]
        self.optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad,
                    chain(self.encoder.parameters(), self.actor_critic.parameters()),
                )
            ),
            lr=lr,
            eps=eps,
        )
        if self.use_inflection_weighting:
            # inflection_factor = L / N where L = episode length and
            # N = # of inflection points in the episode.
            # The loss function will be biased towards inflection points in
            # the episode. Fewer the inflection points, larger the bias.
            # loss = inflection_factor * loss_inflection +
            #        1.0 * loss_non_inflection
            self.inflection_factor = 1.0
            # The inflection factor is updated during training by computing
            # a moving average estimate (with weighting inflection_beta).
            self.inflection_beta = 0.90
            # The inflection factor estimate for an episode is clipped to
            # this value to prevent explosion.
            self.trunc_factor_clipping = 10.0

    def update(self, rollouts):
        """Update the policy based on expert data in the rollouts.
        """
        T, N = rollouts.actions.shape[:2]
        expert_actions = rollouts.actions  # (T, N, 1)
        # Masks indicating when expert actions were *not* taken. This permits
        # a form of data augmentation where non-expert actions are taken to
        # accomodate distribution shifts b/w the expert and the learned policy.
        action_masks = rollouts.action_masks  # (T, N, 1)
        hxs = rollouts.recurrent_hidden_states[0].unsqueeze(0)  # (1, N, nfeats)
        masks = rollouts.masks[:-1]  # (T, N, nfeats)
        # ============= Update inflection factor if applicable ================
        if self.use_inflection_weighting:
            inflection_mask = self._get_inflection_mask(expert_actions)
            # Inverse frequency of inflection points.
            inflection_factor = T / (inflection_mask.sum(dim=0) + 1e-12)
            inflection_factor = torch.clamp(
                inflection_factor, 1.0, self.trunc_factor_clipping
            )
            self._update_inflection_factor(inflection_factor.mean().item())
        # ========================= Forward pass ==============================
        hxs = flatten_two(hxs)  # (N, nfeats)
        masks = flatten_two(masks)  # (T*N, nfeats)
        action_masks = flatten_two(action_masks).squeeze(1)  # (T*N, )
        policy_inputs = self._create_policy_inputs(rollouts)
        # (T*N, nactions)
        pred_action_log_probs = self.actor_critic.get_log_probs(
            policy_inputs, hxs, masks
        )
        # ==================== Compute the prediction loss ====================
        expert_actions = flatten_two(expert_actions).squeeze(1).long()  # (T*N,)
        action_loss = F.nll_loss(
            pred_action_log_probs, expert_actions, reduction="none"
        )  # (T*N, )
        # Weight the loss based on inflection points.
        if self.use_inflection_weighting:
            inflection_mask = flatten_two(inflection_mask).squeeze(1)  # (T*N,)
            action_loss = action_loss * (
                inflection_mask * self.inflection_factor + (1 - inflection_mask) * 1.0
            )
        # Mask the losses for non-expert actions.
        action_loss = (action_loss * action_masks).sum() / (action_masks.sum() + 1e-10)
        # ============================ Backward pass ==========================
        self.optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(
            chain(self.encoder.parameters(), self.actor_critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        losses = {}
        losses["action_loss"] = action_loss.item()
        return losses

    def _update_inflection_factor(self, inflection_factor):
        self.inflection_factor = (
            self.inflection_factor * self.inflection_beta
            + inflection_factor * (1 - self.inflection_beta)
        )

    def _create_policy_inputs(self, rollouts):
        """The policy inputs consist of features extract from the RGB and
        top-down occupancy maps, and learned encodings of the previous actions,
        and collision detections.
        """
        obs_im = rollouts.obs_im[:-1]  # (T, N, *obs_shape)
        encoder_inputs = [obs_im]
        if self.encoder_type == "rgb+map":
            encoder_inputs.append(rollouts.obs_sm[:-1])  # (T, N, *obs_shape)
            encoder_inputs.append(rollouts.obs_lm[:-1])  # (T, N, *obs_shape)
        encoder_inputs = [flatten_two(v) for v in encoder_inputs]
        obs_feats = self.encoder(*encoder_inputs)  # (T*N, nfeats)
        policy_inputs = {"features": obs_feats}
        if self.use_action_embedding:
            prev_actions = torch.zeros_like(rollouts.actions)  # (T, N, 1)
            prev_actions[1:] = rollouts.actions[:-1]
            prev_actions = flatten_two(prev_actions)  # (T*N, 1)
            policy_inputs["actions"] = prev_actions.long()
        if self.use_collision_embedding:
            prev_collisions = flatten_two(rollouts.collisions[:-1])  # (T*N, 1)
            policy_inputs["collisions"] = prev_collisions.long()
        return policy_inputs

    def _get_inflection_mask(self, actions):
        """Given a sequence of actions, it predicts a mask highlighting
        the inflection points, i.e., points when the actions in the
        sequence change between t-1 and t.
        """
        device = actions.device
        # actions - (T, N, 1) tensor
        prev_actions = actions[:-1]
        curr_actions = actions[1:]
        inflection_mask = (curr_actions != prev_actions).float()  # (T-1, N, 1)
        # First action is never an inflection point
        inflection_mask = torch.cat(
            [torch.zeros(1, *actions.shape[1:]).to(device), inflection_mask], dim=0
        )
        return inflection_mask
