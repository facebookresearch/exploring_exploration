#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import torch.nn.modules.transformer as transformer


class View(nn.Module):
    def __init__(self, *shape):
        # shape is a list
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class FeatureReconstructionModule(nn.Module):
    """An encoder-decoder model based on transformers for reconstructing
    concepts at a target location.
    """

    def __init__(self, nfeats, noutputs, nlayers=4):
        super().__init__()
        encoder_layer = transformer.TransformerEncoderLayer(nfeats + 16, 2, nfeats)
        decoder_layer = transformer.TransformerDecoderLayer(nfeats + 16, 2, nfeats)
        self.encoder = transformer.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = transformer.TransformerDecoder(decoder_layer, nlayers)
        self.predict_outputs = nn.Linear(nfeats + 16, noutputs)

    def forward(self, x):
        """
        Inputs:
            x - dictionary consisting of the following:
            {
                'history_image_features': (T, N, nfeats)
                'history_pose_features': (T, N, 16)
                'target_pose_features': (1, N, 16)
            }
        Outputs:
            pred_outputs - (1, N, noutputs)
        """
        target_pose_features = x["target_pose_features"][0]
        T, N, nfeats = x["history_image_features"].shape
        nRef = target_pose_features.shape[1]
        device = x["target_pose_features"].device
        # =================== Encode features and poses =======================
        encoder_inputs = torch.cat(
            [x["history_image_features"], x["history_pose_features"]], dim=2
        )  # (T, N, nfeats+16)
        encoded_features = self.encoder(encoder_inputs)  # (T, N, nfeats+16)
        # ================ Decode features for given poses ====================
        decoder_pose_features = target_pose_features.unsqueeze(0)  # (1, N, 16)
        # Initialize as zeros
        decoder_image_features = torch.zeros(
            *decoder_pose_features.shape[:2], nfeats
        ).to(
            device
        )  # (1, N, nfeats)
        decoder_inputs = torch.cat(
            [decoder_image_features, decoder_pose_features], dim=2
        )  # (1, N, nfeats+16)
        decoder_features = self.decoder(
            decoder_inputs, encoded_features
        )  # (1, N, nfeats+16)
        pred_outputs = self.predict_outputs(decoder_features).squeeze(0)
        return pred_outputs.unsqueeze(0)


class FeatureNetwork(nn.Module):
    """Network to extract image features.
    """

    def __init__(self):
        super().__init__()
        resnet = tmodels.resnet50(pretrained=True)
        self.net = nn.Sequential(
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
        feat = self.net(x).squeeze(3).squeeze(2)
        feat = F.normalize(feat, p=2, dim=1)
        return feat


class PoseEncoder(nn.Module):
    """Network to encode pose information.
    """

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16),)

    def forward(self, x):
        return self.main(x)
