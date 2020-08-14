#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStoragePPO(object):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
        encoder_type="rgb+map",
    ):
        self.obs_im = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        if encoder_type == "rgb+map":
            self.obs_sm = torch.zeros(num_steps + 1, num_processes, *obs_shape)
            self.obs_lm = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.collisions = torch.zeros(num_steps + 1, num_processes, 1)

        self.encoder_type = encoder_type
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs_im = self.obs_im.to(device)
        if self.encoder_type == "rgb+map":
            self.obs_sm = self.obs_sm.to(device)
            self.obs_lm = self.obs_lm.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.collisions = self.collisions.to(device)

    def insert(
        self,
        obs_im,
        obs_sm,
        obs_lm,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        collisions,
    ):
        self.obs_im[self.step + 1].copy_(obs_im)
        if self.encoder_type == "rgb+map":
            self.obs_sm[self.step + 1].copy_(obs_sm)
            self.obs_lm[self.step + 1].copy_(obs_lm)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.collisions[self.step + 1].copy_(collisions)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_im[0].copy_(self.obs_im[-1])
        if self.encoder_type == "rgb+map":
            self.obs_sm[0].copy_(self.obs_sm[-1])
            self.obs_lm[0].copy_(self.obs_lm[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.collisions[0].copy_(self.collisions[-1])

    def update_prev_rewards(self, intrinsic_rewards, nsteps=1):
        # Update rewards from previous time step
        if self.step - nsteps >= 0:
            # Sanity check, do not carry over updates from previous time batch.
            self.rewards[self.step - nsteps] += intrinsic_rewards

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(
                num_processes, num_steps, num_processes * num_steps, num_mini_batch
            )
        )
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False
        )
        for indices in sampler:
            obs_im_batch = self.obs_im[:-1].view(-1, *self.obs_im.size()[2:])[indices]
            if self.encoder_type == "rgb+map":
                obs_sm_batch = self.obs_sm[:-1].view(-1, *self.obs_sm.size()[2:])[
                    indices
                ]
                obs_lm_batch = self.obs_lm[:-1].view(-1, *self.obs_lm.size()[2:])[
                    indices
                ]
            else:
                obs_sm_batch = None
                obs_lm_batch = None
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            collisions_batch = self.collisions[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield (
                obs_im_batch,
                obs_sm_batch,
                obs_lm_batch,
                obs_recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                collisions_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_im_batch = []
            if self.encoder_type == "rgb+map":
                obs_sm_batch = []
                obs_lm_batch = []
            else:
                obs_sm_batch = None
                obs_lm_batch = None
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            collisions_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_im_batch.append(self.obs_im[:-1, ind])
                if self.encoder_type == "rgb+map":
                    obs_sm_batch.append(self.obs_sm[:-1, ind])
                    obs_lm_batch.append(self.obs_lm[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                collisions_batch.append(self.collisions[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_im_batch = torch.stack(obs_im_batch, 1)
            if self.encoder_type == "rgb+map":
                obs_sm_batch = torch.stack(obs_sm_batch, 1)
                obs_lm_batch = torch.stack(obs_lm_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            collisions_batch = torch.stack(collisions_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_im_batch = _flatten_helper(T, N, obs_im_batch)
            if self.encoder_type == "rgb+map":
                obs_sm_batch = _flatten_helper(T, N, obs_sm_batch)
                obs_lm_batch = _flatten_helper(T, N, obs_lm_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            collisions_batch = _flatten_helper(T, N, collisions_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
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
            )

    def reset(self):
        self.obs_im.fill_(0)
        if self.encoder_type == "rgb+map":
            self.obs_sm.fill_(0)
            self.obs_lm.fill_(0)
        self.recurrent_hidden_states.fill_(0)
        self.rewards.fill_(0)
        self.value_preds.fill_(0)
        self.returns.fill_(0)
        self.action_log_probs.fill_(0)
        self.actions.fill_(0)
        self.masks.fill_(1)
        self.collisions.fill_(0)
        self.step = 0


class RolloutStorageImitation(RolloutStoragePPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_steps = self.obs_im.shape[0] - 1
        num_processes = self.obs_im.shape[1]
        self.action_masks = torch.ones(num_steps, num_processes, 1)

    def to(self, device):
        super().to(device)
        self.action_masks = self.action_masks.to(device)

    def insert(
        self,
        obs_im,
        obs_sm,
        obs_lm,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        collisions,
        action_masks,
    ):
        self.action_masks[self.step].copy_(action_masks)
        super().insert(
            obs_im,
            obs_sm,
            obs_lm,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            collisions,
        )

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(
                num_processes, num_steps, num_processes * num_steps, num_mini_batch
            )
        )
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False
        )
        for indices in sampler:
            obs_im_batch = self.obs_im[:-1].view(-1, *self.obs_im.size()[2:])[indices]
            if self.encoder_type == "rgb+map":
                obs_sm_batch = self.obs_sm[:-1].view(-1, *self.obs_sm.size()[2:])[
                    indices
                ]
                obs_lm_batch = self.obs_lm[:-1].view(-1, *self.obs_lm.size()[2:])[
                    indices
                ]
            else:
                obs_sm_batch = None
                obs_lm_batch = None
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            action_masks_batch = self.action_masks.view(-1, self.action_masks.size(-1))[
                indices
            ]
            collisions_batch = self.collisions[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield (
                obs_im_batch,
                obs_sm_batch,
                obs_lm_batch,
                obs_recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                collisions_batch,
                action_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_im_batch = []
            if self.encoder_type == "rgb+map":
                obs_sm_batch = []
                obs_lm_batch = []
            else:
                obs_sm_batch = None
                obs_lm_batch = None
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            collisions_batch = []
            action_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_im_batch.append(self.obs_im[:-1, ind])
                if self.encoder_type == "rgb+map":
                    obs_sm_batch.append(self.obs_sm[:-1, ind])
                    obs_lm_batch.append(self.obs_lm[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                collisions_batch.append(self.collisions[:-1, ind])
                action_masks_batch.append(self.action_masks[:, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_im_batch = torch.stack(obs_im_batch, 1)
            if self.encoder_type == "rgb+map":
                obs_sm_batch = torch.stack(obs_sm_batch, 1)
                obs_lm_batch = torch.stack(obs_lm_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            collisions_batch = torch.stack(collisions_batch, 1)
            action_masks_batch = torch.stack(action_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_im_batch = _flatten_helper(T, N, obs_im_batch)
            if self.encoder_type == "rgb+map":
                obs_sm_batch = _flatten_helper(T, N, obs_sm_batch)
                obs_lm_batch = _flatten_helper(T, N, obs_lm_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            collisions_batch = _flatten_helper(T, N, collisions_batch)
            action_masks_batch = _flatten_helper(T, N, action_masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
                obs_im_batch,
                obs_sm_batch,
                obs_lm_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                collisions_batch,
                action_masks,
                old_action_log_probs_batch,
                adv_targ,
                T,
                N,
            )

    def reset(self):
        super().reset()
        self.action_masks.fill_(0)


class RolloutStorageReconstruction(object):
    def __init__(
        self, num_steps, num_processes, feat_shape, odometer_shape, num_pose_refs,
    ):
        self.obs_feats = torch.zeros(num_steps + 1, num_processes, *feat_shape)
        self.obs_odometer = torch.zeros(num_steps + 1, num_processes, *odometer_shape)
        self.tgt_feats = torch.zeros(num_processes, num_pose_refs, *feat_shape)
        self.tgt_poses = torch.zeros(num_processes, num_pose_refs, 3)
        self.tgt_masks = torch.zeros(num_processes, num_pose_refs, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs_feats = self.obs_feats.to(device)
        self.obs_odometer = self.obs_odometer.to(device)
        self.tgt_feats = self.tgt_feats.to(device)
        self.tgt_poses = self.tgt_poses.to(device)
        self.tgt_masks = self.tgt_masks.to(device)

    def insert(self, obs_feats, obs_odometer):
        self.obs_feats[self.step + 1].copy_(obs_feats)
        self.obs_odometer[self.step + 1].copy_(obs_odometer)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_feats[0].copy_(self.obs_feats[-1])
        self.obs_odometer[0].copy_(self.obs_odometer[-1])

    def reset(self):
        self.obs_feats.fill_(0)
        self.obs_odometer.fill_(0)
        self.tgt_feats.fill_(0)
        self.tgt_poses.fill_(0)
        self.tgt_masks.fill_(1.0)
        self.step = 0


class RolloutStoragePoseEstimation(object):
    def __init__(
        self,
        num_steps,
        num_processes,
        feat_sim_shape,
        feat_pose_shape,
        odometer_shape,
        lab_shape,
        action_space,
        map_shape,
        nRef,
    ):
        self.nRef = nRef
        self.obs_feat_sim = torch.zeros(num_steps + 1, num_processes, *feat_sim_shape)
        self.obs_feat_pose = torch.zeros(num_steps + 1, num_processes, *feat_pose_shape)
        self.obs_odometer = torch.zeros(num_steps + 1, num_processes, *odometer_shape)
        self.poses = torch.zeros(num_steps + 1, num_processes, nRef, *lab_shape)
        self.pose_refs_feat_sim = torch.zeros(
            num_steps + 1, num_processes, nRef, *feat_sim_shape
        )
        self.pose_refs_feat_pose = torch.zeros(
            num_steps + 1, num_processes, nRef, *feat_pose_shape
        )
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            raise ValueError("Only accepts discrete actions")

        self.actions = torch.zeros(num_steps, num_processes, action_shape).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs_feat_sim = self.obs_feat_sim.to(device)
        self.obs_feat_pose = self.obs_feat_pose.to(device)
        self.obs_odometer = self.obs_odometer.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.pose_refs_feat_sim = self.pose_refs_feat_sim.to(device)
        self.pose_refs_feat_pose = self.pose_refs_feat_pose.to(device)
        self.poses = self.poses.to(device)

    def insert(
        self,
        obs_feat_sim,
        obs_feat_pose,
        obs_odometer,
        actions,
        masks,
        poses,
        pose_refs_feat_sim,
        pose_refs_feat_pose,
    ):
        self.obs_feat_sim[self.step + 1].copy_(obs_feat_sim)
        self.obs_feat_pose[self.step + 1].copy_(obs_feat_pose)
        self.obs_odometer[self.step + 1].copy_(obs_odometer)
        self.actions[self.step].copy_(actions)
        self.masks[self.step + 1].copy_(masks)
        self.pose_refs_feat_sim[self.step + 1].copy_(pose_refs_feat_sim)
        self.pose_refs_feat_pose[self.step + 1].copy_(pose_refs_feat_pose)
        self.poses[self.step + 1].copy_(poses)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_feat_sim[0].copy_(self.obs_feat_sim[-1])
        self.obs_feat_pose[0].copy_(self.obs_feat_pose[-1])
        self.obs_odometer[0].copy_(self.obs_odometer[-1])
        self.masks[0].copy_(self.masks[-1])
        self.poses[0].copy_(self.poses[-1])
        self.pose_refs_feat_sim[0].copy_(self.pose_refs_feat_sim[-1])
        self.pose_refs_feat_pose[0].copy_(self.pose_refs_feat_pose[-1])

    def reset(self):
        self.obs_feat_sim.fill_(0)
        self.obs_feat_pose.fill_(0)
        # Position of the reference on the map
        self.obs_odometer.fill_(0)
        self.poses.fill_(0)
        self.pose_refs_feat_sim.fill_(0)
        self.pose_refs_feat_pose.fill_(0)
        self.actions.fill_(0)
        self.masks.fill_(1)
        self.step = 0
