#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models

from exploring_exploration.utils.distributions import Categorical
from exploring_exploration.utils.common import init


class RGBEncoder(nn.Module):
    """Encoder that only takes RGB readings as inputs.
    """

    def __init__(self, fix_cnn=True):
        super().__init__()
        self.rgb_resnet_model = models.resnet18(pretrained=True)
        resnet_models = [self.rgb_resnet_model]
        if fix_cnn:
            for model in resnet_models:
                for param in model.parameters():
                    param.requires_grad = False
        num_ftrs = self.rgb_resnet_model.fc.in_features
        num_in = 0
        self.rgb_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.rgb_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.merge_fc = nn.Linear(num_in, 512)

    def forward(self, rgb):
        """
        Inputs:
            rgb    - (bs, C, H, W)
        Outputs:
            feat - (bs, 512)
        """
        feat_rgb = self.rgb_resnet_model(rgb)
        feat = self.merge_fc(feat_rgb)
        return feat

    def get_feats(self, rgb):
        return self(rgb)


class MapRGBEncoder(nn.Module):
    """Encoder that only takes RGB readings and egocentric occupancy
    maps as inputs.
    """

    def __init__(self, fix_cnn=True):
        super().__init__()
        self.rgb_resnet_model = models.resnet18(pretrained=True)
        self.large_map_resnet_model = models.resnet18(pretrained=True)
        self.small_map_resnet_model = models.resnet18(pretrained=True)
        resnet_models = [
            self.rgb_resnet_model,
            self.large_map_resnet_model,
            self.small_map_resnet_model,
        ]
        if fix_cnn:
            for model in resnet_models:
                for param in model.parameters():
                    param.requires_grad = False
        num_ftrs = self.large_map_resnet_model.fc.in_features
        num_in = 0
        self.rgb_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.rgb_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.large_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.large_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.small_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.small_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.merge_fc = nn.Linear(num_in, 512)

    def forward(self, rgb, small_maps, large_maps):
        """
        Inputs:
            rgb        - (bs, C, H, W)
            small_maps - (bs, C, H, W)
            large_maps - (bs, C, H, W)
        Outputs:
            feat - (bs, 512)
        """
        feat_lm = self.large_map_resnet_model(large_maps)
        feat_sm = self.small_map_resnet_model(small_maps)
        feat_rgb = self.rgb_resnet_model(rgb)
        feat = self.merge_fc(torch.cat([feat_lm, feat_sm, feat_rgb], dim=1))
        return feat

    def get_feats(self, rgb, small_maps, large_maps):
        return self(rgb, small_maps, large_maps)


# Borrowed from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten
            x = x.view(T, N, x.size(1))
            # Same deal with masks
            masks = masks.view(T, N, 1)
            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)
            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
        return x, hxs


class MLPBase(NNBase):
    """An MLP recurrent-encoder for the policy inputs. It takes features,
    past actions, and collisions as inputs.
    """

    def __init__(
        self,
        feat_dim=512,
        recurrent=False,
        hidden_size=512,
        action_config=None,
        collision_config=None,
    ):
        super().__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        main_input_dim = feat_dim
        # Model to encode actions
        self.use_action_embedding = False
        if action_config is not None:
            nactions = action_config["nactions"]
            action_embedding_size = action_config["embedding_size"]
            self.action_encoder = nn.Sequential(
                nn.Embedding(nactions, action_embedding_size),
                nn.Linear(action_embedding_size, action_embedding_size),
                nn.ReLU(),
                nn.Linear(action_embedding_size, action_embedding_size),
            )
            main_input_dim += action_embedding_size
            self.use_action_embedding = True
        # Model to encode collisions
        self.use_collision_embedding = False
        if collision_config is not None:
            collision_dim = collision_config["collision_dim"]
            collision_embedding_size = collision_config["embedding_size"]
            self.collision_encoder = nn.Sequential(
                nn.Embedding(collision_dim, collision_embedding_size),
                nn.Linear(collision_embedding_size, collision_embedding_size),
                nn.ReLU(),
                nn.Linear(collision_embedding_size, collision_embedding_size),
            )
            main_input_dim += collision_embedding_size
            self.use_collision_embedding = True
        # Feature merging
        self.main = nn.Sequential(
            init_(nn.Linear(main_input_dim, hidden_size)), nn.ReLU()
        )
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        # Critic for policy learning
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # Encode input features
        x = self._process_inputs(inputs)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs

    def _process_inputs(self, inputs):
        """
        inputs is a dictionary consisting of the following:
        {
            features: (bs, feat_dim)
            actions: (bs, 1)  (optional)
            collisions: (bs, 1) one hot vector (optional)
        }
        """
        input_values = [inputs["features"]]
        if self.use_action_embedding:
            act_feat = self.action_encoder(inputs["actions"].squeeze(1))
            input_values.append(act_feat)
        if self.use_collision_embedding:
            coll_feat = self.collision_encoder(inputs["collisions"].squeeze(1))
            input_values.append(coll_feat)
        input_values = torch.cat(input_values, dim=1)
        return self.main(input_values)


# Borrowed from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class Policy(nn.Module):
    def __init__(self, action_space, base_kwargs=None):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.base = MLPBase(**base_kwargs)
        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs

    def get_log_probs(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = (dist.probs + 1e-10).log()
        return action_log_probs
