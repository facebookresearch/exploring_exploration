#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from exploring_exploration.utils.geometry import process_pose

from einops import rearrange, reduce, asnumpy


class RetrievalNetwork(nn.Module):
    """Siamese network to estimate visual similarity of two images.
    """

    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feat_extract = nn.Sequential(
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
        self.compare = nn.Sequential(
            nn.Linear(512 * 2, 512), nn.ReLU(), nn.Linear(512, 2)
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
            elif type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight)

        if not pretrained:
            self.feat_extract.apply(init_weights)
        self.compare.apply(init_weights)

    def forward(self, imgA, imgB):
        featA = self.feat_extract(imgA).squeeze(3).squeeze(2)  # (bs, F)
        featB = self.feat_extract(imgB).squeeze(3).squeeze(2)  # (bs, F)
        featAB = torch.cat([featA, featB], dim=1)
        sim_pred = self.compare(featAB)
        return sim_pred

    def get_feats(self, x):
        return self.feat_extract(x).squeeze(3).squeeze(2)


class PairwisePosePredictor(nn.Module):
    """Siamese network to estimate the relative pose between the two images.
    By assuming that the two images are looking at the same object of interest,
    the pose prediction can be factorized into three components:

    The first two components are the depth of the object of interest in
    the two views, and the third component is the baseline angle between
    the two views.
    """

    def __init__(self, pretrained=False, use_classification=False, num_classes=15):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feat_extract = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_pool = resnet.avgpool
        self.compare = nn.Sequential(
            nn.Linear(512 * 2, 512), nn.ReLU(), nn.Linear(512, 2)
        )
        self.predict_depth = nn.Sequential(
            nn.Linear(512 * 9, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.use_classification = use_classification
        # The baseline angle prediction is split into the baseline magnitude
        # and the baseline sign. The baseline magnitude prediction can be
        # performed via classification or regression.
        if use_classification:
            self.predict_baseline = nn.Sequential(
                nn.Linear(512 * 9 * 2, 512), nn.ReLU(), nn.Linear(512, num_classes)
            )
            self.num_classes = num_classes
            angle_intervals = np.radians(np.linspace(0, 90, num_classes + 1))
            self.class_to_angle = torch.Tensor(
                (angle_intervals[:-1] + angle_intervals[1:]) / 2
            )
        else:
            self.predict_baseline = nn.Sequential(
                nn.Linear(512 * 9 * 2, 512), nn.ReLU(), nn.Linear(512, 1)
            )
        # Baseline sign prediction is done via classification.
        self.predict_baseline_sign = nn.Sequential(
            nn.Linear(512 * 9 * 2, 512), nn.ReLU(), nn.Linear(512, 2)
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
            elif type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight)

        if not pretrained:
            self.feat_extract.apply(init_weights)
        self.compare.apply(init_weights)
        self.predict_depth.apply(init_weights)
        self.predict_baseline.apply(init_weights)
        self.predict_baseline_sign.apply(init_weights)

    def forward(self, imgA, imgB):
        # Extract image features.
        featA = self.feat_extract(imgA)
        featB = self.feat_extract(imgB)
        featApool = self.feat_pool(featA).squeeze(3).squeeze(2)  # (bs, F)
        featBpool = self.feat_pool(featB).squeeze(3).squeeze(2)  # (bs, F)
        featABpool = torch.cat([featApool, featBpool], dim=1)
        # Predict visual similarity.
        sim_pred = self.compare(featAB)
        featAB = torch.cat([featA.view(-1, 512 * 9), featB.view(-1, 512 * 9)], dim=1)
        # Predict depth.
        pred_dA = self.predict_depth(featA.view(-1, 512 * 9))  # (bs, 1)
        pred_dB = self.predict_depth(featB.view(-1, 512 * 9))  # (bs, 1)
        # Predict baseline angle.
        pred_baseline = self.predict_baseline(featAB)  # (bs, 1 or num_classes)
        pred_baseline_sign = self.predict_baseline_sign(featAB)  # (bs, 2)
        pose_pred = torch.cat(
            [pred_dA, pred_dB, pred_baseline, pred_baseline_sign], dim=1
        )
        return sim_pred, pose_pred

    def get_pose(self, imgA, imgB):
        # Extract image features.
        featA = self.feat_extract(imgA)
        featB = self.feat_extract(imgB)
        featAB = torch.cat([featA.view(-1, 512 * 9), featB.view(-1, 512 * 9)], dim=1)
        # Predict depth.
        pred_dA = self.predict_depth(featA.view(-1, 512 * 9))
        pred_dB = self.predict_depth(featB.view(-1, 512 * 9))
        # Predict baseline angle.
        pred_baseline = self.predict_baseline(featAB)  # (bs, 1 or num_classes)
        pred_baseline_sign = self.predict_baseline_sign(featAB)  # (bs, 2)
        pose_pred = torch.cat(
            [pred_dA, pred_dB, pred_baseline, pred_baseline_sign], dim=1
        )
        return pose_pred

    def get_pose_feats(self, featA, featB):
        featAB = torch.cat([featA, featB], dim=1)
        # Predict depth.
        pred_dA = self.predict_depth(featA)
        pred_dB = self.predict_depth(featB)
        # Predict baseline angle.
        pred_baseline = self.predict_baseline(featAB)  # (bs, 1 (r num_classes)
        pred_baseline_sign = self.predict_baseline_sign(featAB)  # (bs, 2)
        pose_pred = torch.cat(
            [pred_dA, pred_dB, pred_baseline, pred_baseline_sign], dim=1
        )
        return pose_pred

    def get_feats(self, x):
        return self.feat_extract(x).view(-1, 512 * 9)

    def get_pose_xyt(self, imgA, imgB):
        """Predict pose in the form of x, y, theta.
        """
        # Predict the pose parameterization of depths and baseline angles.
        pose_pred = self.get_pose(imgA, imgB)
        pred_baseline_sign = torch.max(pose_pred[:, -2:], dim=1)[1]
        if self.use_classification:
            pred_baseline_classes = torch.max(pose_pred[:, 2:-2], dim=1)[1]
            pred_baseline = self.convert_alpha_class(pred_baseline_classes)[:, 0]
        else:
            pred_baseline = pose_pred[:, 2]
        pred_baseline[pred_baseline_sign == 0] *= -1
        # Compute position
        # x' = d - r*cos(alpha)
        xAB_pred = pose_pred[:, 0] - pose_pred[:, 1] * torch.cos(pred_baseline)
        # y' = -r*sin(alpha)
        yAB_pred = -pose_pred[:, 1] * torch.sin(pred_baseline)
        # Compute heading angle
        thetaAB_pred = pred_baseline  # theta' = alpha
        return torch.stack([xAB_pred, yAB_pred, thetaAB_pred], dim=1)

    def get_pose_xyt_feats(self, featA, featB):
        """Predict pose in the form of x, y, theta.
        """
        # Predict the pose parameterization of depths and baseline angles.
        pose_pred = self.get_pose_feats(featA, featB)
        pred_baseline_sign = torch.max(pose_pred[:, -2:], dim=1)[1]
        if self.use_classification:
            pred_baseline_classes = torch.max(pose_pred[:, 2:-2], dim=1)[1]
            pred_baseline = self.convert_alpha_class(pred_baseline_classes)[:, 0]
        else:
            pred_baseline = pose_pred[:, 2]
        pred_baseline[pred_baseline_sign == 0] *= -1
        # Compute position
        # x' = d - r*cos(alpha)
        xAB_pred = pose_pred[:, 0] - pose_pred[:, 1] * torch.cos(pred_baseline)
        # y' = -r*sin(alpha)
        yAB_pred = -pose_pred[:, 1] * torch.sin(pred_baseline)
        # Compute heading angle
        thetaAB_pred = pred_baseline  # theta' = alpha
        return torch.stack([xAB_pred, yAB_pred, thetaAB_pred], dim=1)

    def convert_alpha_class(self, alpha):
        # Convert angle from corresponding class to the actual value in
        # radians in the classification setup.
        device = alpha.device
        class_to_angle = self.class_to_angle.to(device)
        class_to_angle = class_to_angle.unsqueeze(0).expand(alpha.shape[0], -1)
        alpha_radians = torch.gather(class_to_angle, 1, alpha.unsqueeze(1).long())
        return alpha_radians


class ViewLocalizer:
    """Given a set of pairwise pose-predictions between a new point and other
    visually similar view-points, this module aggregates the votes and picks
    the location with highest vote as the pose prediction.
    """

    def __init__(self, map_scale):
        self.num_poses = 3
        self.map_scale = map_scale

    def forward(self, all_voting_maps_):
        """Each visually similar viewpoint votes for the location of the
        target viewpoint in the top-down view. These voting maps are then
        summed and the final pose is estimated by taking the location with the
        highest overall vote.

        Inputs:
            all_voting_maps_ - (k, bs, 1, mh, mw)
        Outputs:
            pose             - (bs, num_poses) tensor
            mask_maxmin      - (bs, ) tensor
        """
        bs = all_voting_maps_.shape[1]
        # ======================== Aggregate votes ============================
        all_voting_maps = all_voting_maps_.sum(dim=0)  # (bs, 1, mh, mw)
        # Add small non-zero votes to all locations
        all_voting_maps = all_voting_maps + 1e-9
        voting_sum = reduce(all_voting_maps, "b c h w -> b () () ()", "sum")
        voting_map = all_voting_maps / voting_sum  # (bs, 1, mh, mw)
        # =================== Compute argmax locations ========================
        device = voting_map.device
        mh, mw = voting_map.shape[2], voting_map.shape[3]
        midx, midy = mw // 2, mh // 2
        voting_map_flat = rearrange(asnumpy(voting_map[:, 0]), "b h w -> b (h w)")
        predy, predx = np.unravel_index(np.argmax(voting_map_flat, axis=1), (mh, mw))
        predy = (torch.Tensor(predy).to(device).float() - midy) * self.map_scale
        predx = (torch.Tensor(predx).to(device).float() - midx) * self.map_scale
        predr = torch.norm(torch.stack([predx, predy], dim=1), dim=1)
        predt = torch.atan2(predy, predx)
        # Handle edge-case where no votes were cast. When no votes are cast,
        # the pose prediction is set to zeros by default.
        diff_maxmin = np.max(voting_map_flat, axis=1) - np.min(
            voting_map_flat, axis=1
        )  # (bs, )
        mask_maxmin = (torch.Tensor(diff_maxmin) == 0).to(device)
        predr[mask_maxmin] = 0
        predt[mask_maxmin] = 0
        # Converts predicted pose to required format.
        pose = process_pose(torch.stack([predr, predt], dim=1))
        return pose, mask_maxmin

    def get_position_and_pose(self, all_voting_maps_, ref_poses):
        """The output format here is explicitly in terms of (x, y, heading).

        Inputs:
            all_voting_maps_ - (k, bs, 1, mh, mw)
            ref_poses        - (k, bs, 1)
        Outputs:
            output     - (bs, 3) tensor of X, Y, theta positions
        """
        k, bs, _, mh, mw = all_voting_maps_.shape
        device = all_voting_maps_.device
        # ======================== Aggregate votes ============================
        all_voting_maps = all_voting_maps_.sum(dim=0)  # (bs, 1, mh, mw)
        # Add small non-zero votes to all locations
        all_voting_maps = all_voting_maps + 1e-9
        voting_sum = reduce(all_voting_maps, "b c h w -> b () () ()", "sum")  # (bs, )
        voting_map = all_voting_maps / voting_sum  # (bs, 1, mh, mw)
        # =================== Compute argmax locations ========================
        midx, midy = voting_map.shape[3] // 2, voting_map.shape[2] // 2
        voting_map_np = voting_map[:, 0].cpu().detach().numpy()
        predy, predx = np.unravel_index(
            np.argmax(voting_map_np.reshape(bs, -1), axis=1), voting_map_np.shape[1:]
        )

        predy_int, predx_int = (
            torch.Tensor(predy).long().to(device),
            torch.Tensor(predx).long().to(device),
        )
        pred_coor_int = predy_int * mw + predx_int  # (bs, )
        pred_coor_int = (
            pred_coor_int.unsqueeze(0).expand(k, -1).unsqueeze(2)
        )  # (k, bs, 1)

        voting_masks = (
            torch.gather(all_voting_maps_.view(k, bs, -1), 2, pred_coor_int) > 0
        ).float()  # (k, bs, 1)

        predt = (
            (ref_poses * voting_masks).sum(dim=0) / (voting_masks.sum(dim=0) + 1e-8)
        ).squeeze(1)
        predy = (
            torch.Tensor(predy).to(device).float() - midy + 0.5
        ) * self.map_scale  # (bs, )
        predx = (
            torch.Tensor(predx).to(device).float() - midx + 0.5
        ) * self.map_scale  # (bs, )

        pose = torch.stack([predx, predy, predt], dim=1)

        return pose

    def to(self, device):
        pass

    def train(self):
        pass

    def eval(self):
        pass
