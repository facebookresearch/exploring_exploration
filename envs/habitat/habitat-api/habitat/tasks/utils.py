#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import math
import numpy as np
import quaternion  # noqa # pylint: disable=unused-import
import scipy.stats as stats

from typing import Tuple


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format
    """
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion

    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate

    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def compute_heading_from_quaternion(rotation: np.quaternion) -> float:
    r"""Converts np.quaternion to a heading angle scalar.

    Args:
        rotation - represents a counter-clockwise rotation about Y-axis.
    Returns:
        Heading angle with clockwise rotation from -Z to X being +ve.
    """
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(rotation.inverse(), direction_vector)
    # Flip sign to compute clockwise rotation
    phi = -cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

    return phi


def compute_quaternion_from_heading(heading: float) -> np.quaternion:
    r"""Converts a heading angle scalar to np.quaternion.

    Args:
        heading - represents a clockwise rotation angle in radians
                  from -Z to X.
    Returns:
        A quaternion that represents a counter-clockwise rotation about Y-axis.
    """
    # Real part of quaternion.
    q0 = math.cos(-heading / 2)
    # Imaginary part of quaternion.
    q = (0, math.sin(-heading / 2), 0)
    return np.quaternion(q0, *q)


def compute_egocentric_delta(
    position1: np.array,
    rotation1: np.quaternion,
    position2: np.array,
    rotation2: np.quaternion,
) -> np.array:
    r"""Computes the relative change in pose from (position1, rotation1) to
    (position2, rotation2).
    Conventions for position, rotation follow habitat simulator conventions,
    i.e., -Z is forward, X is rightward, Y is upward and
    the rotation is a counter-clockwise rotation about Y.

    Args:
        position1 - (x, y, z) position.
        position2 - (x, y, z) position.
        rotation1 - represents a counter-clockwise rotation about Y-axis.
        rotation2 - represents a counter-clockwise rotation about Y-axis.
    Returns:
        Relative change from (position1, rotation1) to (position2, rotation2)
        in polar coordinates (r, phi, theta)
    """
    x1, y1, z1 = position1
    x2, y2, z2 = position2
    # Clockwise rotations from -Z to X.
    theta_1 = compute_heading_from_quaternion(rotation1)
    theta_2 = compute_heading_from_quaternion(rotation2)
    # Compute relative delta in polar coordinates.
    # In this 2D coordinate system, X' is forward and Y' is rightward.
    D_rho = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
    D_phi = math.atan2(x2 - x1, -z2 + z1) - theta_1
    D_theta = theta_2 - theta_1
    return np.array((D_rho, D_phi, D_theta))


def truncated_normal_noise(eta, width):
    """Generates truncated normal noise scalar.

    Args:
        eta - standard deviation of gaussian.
        width - maximum absolute width on either sides of the mean.
    Returns:
        Sampled noise scalar from truncated gaussian with mean=0, sigma=eta,
        and width=width.
    """
    mu = 0
    sigma = eta
    lower = mu - width
    upper = mu + width
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs()


def compute_updated_pose(
    position: np.array, rotation: np.quaternion, delta_rpt: np.array, delta_y: float,
) -> Tuple[np.array, np.quaternion]:
    r"""Given initial position, rotation and an egocentric delta,
    compute the updated pose by linear transformations.

    Args:
        position - (x, y, z) position.
        rotation - represents a counter-clockwise rotation about Y-axis.
        delta_rpt - egocentric delta in polar coordinates.
        delta_y - change in Y position.
    Returns:
        New position, rotation with deltas applied to old position, rotation.
    """
    x, y, z = position
    theta = compute_heading_from_quaternion(rotation)
    D_rho, D_phi, D_theta = delta_rpt

    x_new = x + D_rho * math.sin(theta + D_phi)
    y_new = y + delta_y
    z_new = z - D_rho * math.cos(theta + D_phi)
    theta_new = theta + D_theta

    position_new = np.array([x_new, y_new, z_new])
    rotation_new = compute_quaternion_from_heading(theta_new)

    return (position_new, rotation_new)
