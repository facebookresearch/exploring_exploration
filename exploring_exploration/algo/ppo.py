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


class PPO:
    """Algorithm to learn a policy via Proximal Policy Optimization:
    https://arxiv.org/abs/1707.06347 .
    Large parts of the code were borrowed from Ilya Kostrikov's codebase:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    """

    def __init__(self, config):
        self.encoder = config["encoder"]
        self.actor_critic = config["actor_critic"]
        lr = config["lr"]
        eps = config["eps"]
        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.encoder_type = config["encoder_type"]
        self.num_mini_batch = config["num_mini_batch"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.nactions = config["nactions"]
        self.value_loss_coef = config["value_loss_coef"]
        self.use_clipped_value_loss = config["use_clipped_value_loss"]
        self.use_action_embedding = config["use_action_embedding"]
        self.use_collision_embedding = config["use_collision_embedding"]

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

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_im_batch,
                    obs_sm_batch,
                    obs_lm_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    collisions_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    T,
                    N,
                ) = sample

                # ======================= Forward pass ========================
                encoder_inputs = [obs_im_batch]
                if self.encoder_type == "rgb+map":
                    encoder_inputs += [obs_sm_batch, obs_lm_batch]
                obs_feats = self.encoder(*encoder_inputs)
                policy_inputs = {"features": obs_feats}
                prev_actions = torch.zeros_like(actions_batch.view(T, N, 1))
                prev_actions[1:] = actions_batch.view(T, N, 1)[:-1]
                prev_actions = prev_actions.view(T * N, 1)
                prev_collisions = collisions_batch
                if self.use_action_embedding:
                    policy_inputs["actions"] = prev_actions.long()
                if self.use_collision_embedding:
                    policy_inputs["collisions"] = prev_collisions.long()
                # Reshape to do in a single forward pass for all steps
                policy_outputs = self.actor_critic.evaluate_actions(
                    policy_inputs,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    actions_batch,
                )
                values, action_log_probs, dist_entropy, _ = policy_outputs
                # ===================== Compute PPO losses ====================
                # Clipped surrogate loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()
                # Value function loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)
                # ======================= Backward pass =======================
                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    chain(self.encoder.parameters(), self.actor_critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                # ===================== Update statistics =====================
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        losses = {}
        losses["value_loss"] = value_loss_epoch
        losses["action_loss"] = action_loss_epoch
        losses["dist_entropy"] = dist_entropy_epoch
        return losses
