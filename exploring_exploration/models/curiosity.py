#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ForwardDynamics(nn.Module):
    """Model that takes previous state, and action as inputs to predict
    the next state.
    """

    def __init__(self, n_actions):
        super().__init__()
        state_size = 512
        hidden_size = 256

        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Sequential(
                    nn.Linear(hidden_size + n_actions, hidden_size),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.fc2 = nn.Sequential(
                    nn.Linear(hidden_size + n_actions, hidden_size)
                )

            def forward(self, feat, act):
                x = feat
                x = self.fc1(torch.cat([x, act], dim=1))
                x = self.fc2(torch.cat([x, act], dim=1))
                return feat + x

        self.pre_rb = nn.Sequential(
            nn.Linear(state_size + n_actions, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.post_rb = nn.Linear(hidden_size, state_size)
        self.rb1 = ResidualBlock()
        self.rb2 = ResidualBlock()
        self.rb3 = ResidualBlock()
        self.rb4 = ResidualBlock()

    def forward(self, s, a):
        """
        Inputs:
            s - (bs, state_size)
            a - (bs, n_actions) onehot encoding
        Outputs:
            sp - (bs, state_size)
        """
        x = self.pre_rb(torch.cat([s, a], dim=1))
        x = self.rb1(x, a)
        x = self.rb2(x, a)
        x = self.rb3(x, a)
        x = self.rb4(x, a)
        sp = self.post_rb(x)
        return sp


class Phi(nn.Module):
    """A simple imagenet-pretrained encoder for state representation.
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.main = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

    def forward(self, x):
        """
        Inputs:
            x - (bs, C, H, W)
        Outputs:
            feat - (bs, 512)
        """
        feat = self.main(x).squeeze(3).squeeze(2)
        return feat


# Maintains running statistics of the mean and standard deviation of
# the episode returns. Used for reward normalization as done here:
# https://arxiv.org/pdf/1808.04355.pdf
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count
