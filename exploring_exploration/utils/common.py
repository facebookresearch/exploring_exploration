#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from einops import rearrange, asnumpy


def flatten_two(x):
    """Flattens the first two dims.
    """
    return x.view(-1, *x.shape[2:])


def unflatten_two(x, sh1, sh2):
    """Unflattens the first two dims.
    """
    return x.view(sh1, sh2, *x.shape[1:])


def unsq_exp(x, reps, dim=0):
    """Unsqueezes along dimension dim and repeats along the axis reps times.
    """
    x_e = x.unsqueeze(dim)
    exp_args = [-1] * len(x_e.shape)
    exp_args[dim] = reps
    return x_e.expand(*exp_args).contiguous()


approx_eq = lambda a, b, eps: torch.lt(torch.abs(a - b), eps)


def norm_angle(x):
    """Normalizes an angle (scalar) between -pi to pi.
    """
    if isinstance(x, np.ndarray):
        return np.arctan2(np.sin(x), np.cos(x))
    elif isinstance(x, torch.Tensor):
        return torch.atan2(torch.sin(x), torch.cos(x))
    else:
        return math.atan2(math.sin(x), math.cos(x))


def freeze_params(module):
    """Freezes all parameters of a module by setting requires_grad to False.
    """
    for param in module.parameters():
        param.requires_grad = False


def unnormalize(data, mean, std):
    # data - (bs, H, W, C)
    data[:, :, :, 0] = data[:, :, :, 0] * std[0] + mean[0]
    data[:, :, :, 1] = data[:, :, :, 1] * std[1] + mean[1]
    data[:, :, :, 2] = data[:, :, :, 2] * std[2] + mean[2]
    return data


def process_image(img):
    """Apply imagenet normalization to a batch of images.
    """
    # img - (bs, C, H, W)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_proc = img.float() / 255.0

    img_proc[:, 0] = (img_proc[:, 0] - mean[0]) / std[0]
    img_proc[:, 1] = (img_proc[:, 1] - mean[1]) / std[1]
    img_proc[:, 2] = (img_proc[:, 2] - mean[2]) / std[2]

    return img_proc


def resize_image(img, shape=(84, 84), mode="bilinear"):
    """Resizes a batch of images.
    """
    # img - (bs, C, H, W) FloatTensor
    out_img = F.interpolate(img, size=shape, mode=mode)
    return out_img


def unprocess_image(img):
    """Undo imagenet normalization to a batch of images."""
    # img - (bs, C, H, W)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_unproc = np.copy(asnumpy(img))
    img_unproc[:, 0] = img_unproc[:, 0] * std[0] + mean[0]
    img_unproc[:, 1] = img_unproc[:, 1] * std[1] + mean[1]
    img_unproc[:, 2] = img_unproc[:, 2] * std[2] + mean[2]

    img_unproc = np.clip(img_unproc, 0.0, 1.0) * 255.0
    img_unproc = img_unproc.astype(np.uint8)
    img_unproc = rearrange(img_unproc, "b c h w -> b h w c")

    return img_unproc


# Weight initializations
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def random_range(start, end, interval=1):
    """Returns a randomized range of numbers.
    """
    vals = list(range(start, end, interval))
    random.shuffle(vals)
    return vals
