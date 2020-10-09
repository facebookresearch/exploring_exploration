#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import cv2
import math
import h5py
import logging
import numpy as np
import torch

from exploring_exploration.utils.storage import RolloutStoragePoseEstimation
from exploring_exploration.utils.geometry import (
    process_odometer,
    compute_egocentric_coors,
    process_pose,
)
from exploring_exploration.utils.visualization import (
    torch_to_np,
    torch_to_np_depth,
)
from exploring_exploration.utils.common import (
    process_image,
    resize_image,
    flatten_two,
    unflatten_two,
)
from exploring_exploration.utils.pose_estimation import (
    compute_pose_sptm,
    compute_pose_sptm_ransac,
    RansacPoseEstimator,
)
from exploring_exploration.models.frontier_agent import FrontierAgent
from exploring_exploration.utils.visualization import TensorboardWriter
from exploring_exploration.utils.median_pooling import MedianPool1d
from exploring_exploration.utils.metrics import compute_pose_metrics
from einops import rearrange, asnumpy


def evaluate_pose(
    models,
    envs,
    config,
    device,
    multi_step=False,
    interval_steps=None,
    visualize_policy=True,
    visualize_size=200,
    visualize_batches=8,
    visualize_n_per_batch=1,
):
    # =============== Evaluation configs ======================
    num_steps = config["num_steps"]
    num_processes = config["num_processes"]
    obs_shape = config["obs_shape"]
    feat_shape_sim = config["feat_shape_sim"]
    feat_shape_pose = config["feat_shape_pose"]
    odometer_shape = config["odometer_shape"]
    lab_shape = config["lab_shape"]
    map_shape = config["map_shape"]
    map_scale = config["map_scale"]
    angles = config["angles"]
    bin_size = config["bin_size"]
    gaussian_kernel = config["gaussian_kernel"]
    match_thresh = config["match_thresh"]
    pose_loss_fn = config["pose_loss_fn"]
    num_eval_episodes = config["num_eval_episodes"]
    num_pose_refs = config["num_pose_refs"]
    median_filter_size = config["median_filter_size"]
    vote_kernel_size = config["vote_kernel_size"]
    env_name = config["env_name"]
    actor_type = config["actor_type"]
    encoder_type = config["encoder_type"]
    pose_predictor_type = config["pose_predictor_type"]
    ransac_n = config["ransac_n"]
    ransac_niter = config["ransac_niter"]
    ransac_batch = config["ransac_batch"]
    use_action_embedding = config["use_action_embedding"]
    use_collision_embedding = config["use_collision_embedding"]
    vis_save_dir = config["vis_save_dir"]
    if "final_topdown_save_path" in config:
        final_topdown_save_path = config["final_topdown_save_path"]
    else:
        final_topdown_save_path = None
    rescale_image_flag = config.get("input_highres", False)

    if actor_type == "forward":
        forward_action_id = config["forward_action_id"]
    elif actor_type == "forward-plus":
        forward_action_id = config["forward_action_id"]
        turn_action_id = config["turn_action_id"]
    elif actor_type == "frontier":
        assert num_processes == 1
        if "avd" in env_name:
            action_space = {"forward": 2, "left": 0, "right": 1, "stop": 3}
        else:
            action_space = {"forward": 0, "left": 1, "right": 2, "stop": 3}
        occ_map_scale = config["occ_map_scale"]
        frontier_dilate_occ = config["frontier_dilate_occ"]
        max_time_per_target = config["max_time_per_target"]

        frontier_agent = FrontierAgent(
            action_space,
            env_name,
            occ_map_scale,
            show_animation=False,
            dilate_occupancy=frontier_dilate_occ,
            max_time_per_target=max_time_per_target,
        )

    use_policy = (
        actor_type != "random"
        and actor_type != "oracle"
        and actor_type != "forward"
        and actor_type != "forward-plus"
        and actor_type != "frontier"
    )

    # =============== Models ======================
    rnet = models["rnet"]
    posenet = models["posenet"]
    pose_head = models["pose_head"]
    if use_policy:
        encoder = models["encoder"]
        actor_critic = models["actor_critic"]

    # Set to evaluation mode
    rnet.eval()
    posenet.eval()
    pose_head.eval()
    if use_policy:
        encoder.eval()
        actor_critic.eval()

    # ============ Create median filter ================
    median_filter = MedianPool1d(median_filter_size, 1, median_filter_size // 2)
    median_filter.to(device)

    # =============== Create rollouts  ======================
    rollouts = RolloutStoragePoseEstimation(
        num_steps,
        num_processes,
        feat_shape_sim,
        feat_shape_pose,
        odometer_shape,
        lab_shape,
        envs.action_space,
        map_shape,
        num_pose_refs,
    )
    rollouts.to(device)

    tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    per_episode_statistics = []
    # =============== Gather evaluation info  ======================
    episode_environment_statistics = []
    if multi_step:
        episode_losses_all = {interval: [] for interval in interval_steps}
        true_poses_all = {interval: [] for interval in interval_steps}
        pred_poses_all = {interval: [] for interval in interval_steps}
        true_pose_angles_all = {interval: [] for interval in interval_steps}
        pred_pose_angles_all = {interval: [] for interval in interval_steps}
    else:
        episode_losses_all = []
        true_poses_all = []
        pred_poses_all = []
        true_pose_angles_all = []
        pred_pose_angles_all = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if rescale_image_flag:
            obs_im = resize_image(obs_im)
        if encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
            if rescale_image_flag:
                obs_lm = resize_image(obs_lm)
                obs_sm = resize_image(obs_sm)
        else:
            obs_lm = None
            obs_sm = None

        return obs_im, obs_sm, obs_lm

    num_eval_batches = (num_eval_episodes // num_processes) + 1
    times_per_episode = []
    for neval in range(num_eval_batches):
        ep_start_time = time.time()
        obs = envs.reset()
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)
        # Convert reference poses to map coordinates
        obs_collns = obs["collisions"].long()  # (num_processes, 1)
        obs_odometer = process_odometer(obs["delta"])
        pose_refs = process_image(
            flatten_two(obs["pose_refs"])
        )  # (num_processes * num_pose_refs, ...)
        if rescale_image_flag:
            pose_refs = resize_image(pose_refs)
        pose_refs = unflatten_two(pose_refs, num_processes, num_pose_refs)
        poses = process_pose(
            flatten_two(obs["pose_regress"])
        )  # (num_processes * num_pose_refs, ...)
        poses = unflatten_two(poses, num_processes, num_pose_refs)
        if actor_type == "frontier":
            delta_ego = torch.zeros((num_processes, 3)).to(device)
            frontier_agent.reset()

        if use_policy:
            recurrent_hidden_states = torch.zeros(num_processes, feat_shape_sim[0]).to(
                device
            )
            masks = torch.zeros(num_processes, 1).to(device)

        with torch.no_grad():
            obs_feat_sim = rnet.get_feats(obs_im)
            obs_feat_pose = posenet.get_feats(obs_im)
            pose_refs_feat_sim = rnet.get_feats(
                flatten_two(pose_refs)
            )  # (N*nRef, feat_size_sim)
            pose_refs_feat_pose = posenet.get_feats(
                flatten_two(pose_refs)
            )  # (N*nRef, feat_size_pose)
            pose_refs_feat_sim = unflatten_two(
                pose_refs_feat_sim, num_processes, num_pose_refs
            )
            pose_refs_feat_pose = unflatten_two(
                pose_refs_feat_pose, num_processes, num_pose_refs
            )

        # Initialize the memory of rollouts
        rollouts.reset()
        rollouts.obs_feat_sim[0].copy_(obs_feat_sim)
        rollouts.obs_feat_pose[0].copy_(obs_feat_pose)
        rollouts.obs_odometer[0].copy_(obs_odometer)
        rollouts.pose_refs_feat_sim[0].copy_(pose_refs_feat_sim)
        rollouts.pose_refs_feat_pose[0].copy_(pose_refs_feat_pose)
        rollouts.poses[0].copy_(poses)
        true_heading_angles = obs["pose_regress"][:, :, 2]  # (N, nRef)

        prev_action = torch.zeros(num_processes, 1).long().to(device)
        prev_collision = obs_collns

        for step in range(num_steps):
            if use_policy:
                encoder_inputs = [obs_im]
                if encoder_type == "rgb+map":
                    encoder_inputs += [obs_sm, obs_lm]
                with torch.no_grad():
                    obs_feats = encoder(*encoder_inputs)
                    policy_inputs = {"features": obs_feats}
                    if use_action_embedding:
                        policy_inputs["actions"] = prev_action.long()
                    if use_collision_embedding:
                        policy_inputs["collisions"] = prev_collision.long()

                    policy_outputs = actor_critic.act(
                        policy_inputs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=False,
                    )
                    _, action, _, recurrent_hidden_states = policy_outputs
            elif actor_type == "oracle":
                action = obs["oracle_action"].long()
            elif actor_type == "random":
                action = torch.Tensor(
                    np.random.randint(0, envs.action_space.n, size=(num_processes, 1))
                ).long()
            elif actor_type == "forward":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                action = action.long()

            elif actor_type == "forward-plus":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                collision_mask = prev_collision > 0
                action[collision_mask] = turn_action_id
                action = action.long()

            elif actor_type == "frontier":
                # This assumes that num_processes = 1
                occ_map = obs["highres_coarse_occupancy"][0].cpu().numpy()
                occ_map = occ_map.transpose(1, 2, 0)
                occ_map = np.ascontiguousarray(occ_map)
                occ_map = occ_map.astype(np.uint8)
                action = frontier_agent.act(
                    occ_map, delta_ego[0].cpu().numpy(), prev_collision[0].item()
                )
                action = torch.Tensor([[action]]).long()

            obs, reward, done, infos = envs.step(action)

            for pr in range(num_processes):
                if step == 0:
                    episode_environment_statistics.append(
                        infos[pr]["environment_statistics"]
                    )

            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)
            obs_odometer = process_odometer(obs["delta"])
            obs_collns = obs["collisions"]  # (N, 1)
            if actor_type == "frontier":
                delta_ego = compute_egocentric_coors(
                    obs_odometer, rollouts.obs_odometer[step], occ_map_scale,
                )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            with torch.no_grad():
                obs_feat_sim = rnet.get_feats(obs_im)
                obs_feat_pose = posenet.get_feats(obs_im)
                refs_feat_sim = rnet.get_feats(
                    flatten_two(pose_refs)
                )  # (N*nRef, feat_size_sim)
                refs_feat_pose = posenet.get_feats(
                    flatten_two(pose_refs)
                )  # (N*nRef, feat_size_pose)
                refs_feat_sim = unflatten_two(
                    refs_feat_sim, num_processes, num_pose_refs
                )  # (N, nRef, feat_size_sim)
                refs_feat_pose = unflatten_two(
                    refs_feat_pose, num_processes, num_pose_refs
                )  # (N, nRef, feat_size_pose)

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(num_processes)]).to(device)
            # Accumulate odometer readings to give relative pose from the starting point
            obs_odometer = rollouts.obs_odometer[step] * masks + obs_odometer

            # This must not reach done = True
            assert done[0] == False
            # Update rollouts
            rollouts.insert(
                obs_feat_sim,
                obs_feat_pose,
                obs_odometer,
                action,
                masks,
                poses,
                refs_feat_sim,
                refs_feat_pose,
            )
            # Update prev values
            prev_collision = obs_collns
            prev_action = action

        obs_feat_sim = rollouts.obs_feat_sim  # (T, N, feat_size_sim)
        obs_feat_pose = rollouts.obs_feat_pose  # (T, N, feat_size_pose)
        obs_odometer = rollouts.obs_odometer
        refs_feat_sim = rollouts.pose_refs_feat_sim[-1]  # (N, nRef, feat_size_sim)
        refs_feat_pose = rollouts.pose_refs_feat_pose[-1]  # (N, nRef, feat_size_pose)

        pose_config = {}
        pose_config["map_shape"] = map_shape
        pose_config["map_scale"] = map_scale
        pose_config["bin_size"] = bin_size
        pose_config["angles"] = angles
        pose_config["median_filter_size"] = median_filter_size
        pose_config["vote_kernel_size"] = vote_kernel_size
        pose_config["match_thresh"] = match_thresh
        if pose_predictor_type == "ransac":
            ransac_config = {}
            ransac_config["ransac_n"] = ransac_n  # 5
            ransac_config["ransac_theta_1"] = 1000.0 if "avd" in env_name else 1.0
            ransac_config["ransac_theta_2"] = math.radians(30)
            ransac_config["ransac_niter"] = ransac_niter
            ransac_config["ransac_batch"] = ransac_batch  # 5
            ransac_config["map_shape"] = map_shape
            ransac_config["map_scale"] = map_scale
            ransac_config["bin_size"] = bin_size
            ransac_config["angles"] = angles
            ransac_config["vote_kernel_size"] = vote_kernel_size
            ransac_estimator = RansacPoseEstimator(ransac_config, pose_head, device)
            pose_predictor = compute_pose_sptm_ransac
        else:
            ransac_estimator = None
            pose_predictor = compute_pose_sptm

        pose_models = {}
        pose_models["rnet"] = rnet
        pose_models["posenet"] = posenet
        pose_models["pose_head"] = pose_head
        pose_models["ransac_estimator"] = ransac_estimator

        # ========== Predict pose ============
        if multi_step:
            for interval in interval_steps:
                with torch.no_grad():
                    predicted_all = pose_predictor(
                        obs_feat_sim[:interval],
                        obs_feat_pose[:interval],
                        obs_odometer[:interval],
                        refs_feat_sim,
                        refs_feat_pose,
                        pose_config,
                        pose_models,
                        device,
                        env_name,
                    )
                    predicted_poses = predicted_all[
                        "predicted_poses"
                    ]  # (N, nRef, num_poses)
                    predicted_heading = flatten_two(
                        predicted_all["predicted_positions"][:, :, 2]
                    )  # (N, nRef)
                    predicted_poses = flatten_two(
                        predicted_poses
                    )  # (N*nRef, num_poses)

                    true_poses = rollouts.poses[0]  # (N, nRef, *lab_shape)
                    true_poses = flatten_two(true_poses)  # (N*nRef, *lab_shape)
                    true_heading = flatten_two(true_heading_angles)  # (N*nRef, )

                    pose_loss = pose_loss_fn(predicted_poses, true_poses).cpu()

                episode_losses_all[interval].append(pose_loss)
                true_poses_all[interval].append(true_poses.cpu())
                pred_poses_all[interval].append(predicted_poses.cpu())
                true_pose_angles_all[interval].append(true_heading.cpu())
                pred_pose_angles_all[interval].append(predicted_heading.cpu())
        else:
            with torch.no_grad():
                predicted_all = pose_predictor(
                    obs_feat_sim,
                    obs_feat_pose,
                    obs_odometer,
                    refs_feat_sim,
                    refs_feat_pose,
                    pose_config,
                    pose_models,
                    device,
                    env_name,
                )
                predicted_poses = predicted_all[
                    "predicted_poses"
                ]  # (N, nRef, num_poses)
                predicted_poses = flatten_two(predicted_poses)  # (N*nRef, num_poses)
                predicted_heading = flatten_two(
                    predicted_all["predicted_positions"][:, :, 2]
                )  # (N, nRef)

                true_poses = rollouts.poses[0]  # (N, nRef, *lab_shape)
                true_poses = flatten_two(true_poses)  # (N*nRef, *lab_shape)
                true_heading = flatten_two(true_heading_angles)  # (N*nRef, )
                pose_loss = pose_loss_fn(predicted_poses, true_poses).cpu()

            episode_losses_all.append(pose_loss)
            true_poses_all.append(true_poses.cpu())
            pred_poses_all.append(predicted_poses.cpu())
            true_pose_angles_all.append(true_heading.cpu())
            pred_pose_angles_all.append(predicted_heading.cpu())

        # End of episode
        times_per_episode.append(time.time() - ep_start_time)
        mins_per_episode = np.mean(times_per_episode).item() / 60.0
        eta_completion = mins_per_episode * (num_eval_batches - neval)
        neps_done = (neval + 1) * num_processes
        neps_total = num_eval_batches * num_processes
        logging.info(
            f"=====> Episodes done: {neps_done}/{neps_total}, Time per episode: {mins_per_episode:.3f} mins, ETA completion: {eta_completion:.3f} mins"
        )

    envs.close()

    if multi_step:
        metrics = {interval: {} for interval in interval_steps}

        # Fill in per-episode statistics
        total_episodes = (
            torch.cat(episode_losses_all[interval_steps[-1]], dim=0).shape[0]
            // num_pose_refs
        )
        per_episode_statistics = [
            [] for _ in range(total_episodes)
        ]  # Each episode can have results over multiple intervals
        for interval in interval_steps:
            episode_losses_all[interval] = torch.cat(
                episode_losses_all[interval], dim=0
            ).numpy()
            true_poses_all[interval] = torch.cat(
                true_poses_all[interval], dim=0
            ).numpy()
            pred_poses_all[interval] = torch.cat(
                pred_poses_all[interval], dim=0
            ).numpy()
            true_pose_angles_all[interval] = torch.cat(
                true_pose_angles_all[interval], dim=0
            ).numpy()
            pred_pose_angles_all[interval] = torch.cat(
                pred_pose_angles_all[interval], dim=0
            ).numpy()

            logging.info(
                "======= Evaluating at {} steps for {} episodes ========".format(
                    interval, num_eval_batches * num_processes
                )
            )
            metrics[interval]["pose_loss"] = np.mean(episode_losses_all[interval])

            pose_metrics, per_episode_pose_metrics = compute_pose_metrics(
                true_poses_all[interval],
                pred_poses_all[interval],
                true_pose_angles_all[interval],
                pred_pose_angles_all[interval],
                env_name,
            )

            for k, v in pose_metrics.items():
                metrics[interval][k] = v

            for k, v in metrics[interval].items():
                logging.info("{}: {:.3f}".format(k, v))

            # Average over the pose references
            episode_losses_all[interval] = (
                episode_losses_all[interval].reshape(-1, num_pose_refs).mean(axis=1)
            )
            for k in per_episode_pose_metrics.keys():
                per_episode_pose_metrics[k] = (
                    per_episode_pose_metrics[k].reshape(-1, num_pose_refs).mean(axis=1)
                )

            for nep in range(total_episodes):
                per_episode_metrics = {
                    "time_step": interval,
                    "pose_loss": episode_losses_all[interval][nep].item(),
                    "environment_statistics": episode_environment_statistics[nep],
                }

                for k, v in per_episode_pose_metrics.items():
                    per_episode_metrics[k] = v[nep].item()
                per_episode_statistics[nep].append(per_episode_metrics)
    else:
        episode_losses_all = torch.cat(episode_losses_all, dim=0).numpy()
        true_poses_all = torch.cat(true_poses_all, dim=0).numpy()
        pred_poses_all = torch.cat(pred_poses_all, dim=0).numpy()
        true_pose_angles_all = torch.cat(true_pose_angles_all, dim=0).numpy()
        pred_pose_angles_all = torch.cat(pred_pose_angles_all, dim=0).numpy()

        metrics = {}
        metrics["pose_loss"] = np.mean(episode_losses_all)

        pose_metrics, per_episode_pose_metrics = compute_pose_metrics(
            true_poses_all,
            pred_poses_all,
            true_pose_angles_all,
            pred_pose_angles_all,
            env_name,
        )
        for k, v in pose_metrics.items():
            metrics[k] = v

        # Fill in per-episode statistics
        total_episodes = episode_losses_all.shape[0] // num_pose_refs
        per_episode_statistics = [
            [] for _ in range(total_episodes)
        ]  # Each episode can have results over multiple intervals
        # Average over the pose references
        episode_losses_all = episode_losses_all.reshape(-1, num_pose_refs).mean(axis=1)
        for k in per_episode_pose_metrics.keys():
            per_episode_pose_metrics[k] = (
                per_episode_pose_metrics[k].reshape(-1, num_pose_refs).mean(axis=1)
            )
        for nep in range(total_episodes):
            per_episode_metrics = {
                "time_step": num_steps,
                "pose_loss": episode_losses_all[nep].item(),
                "environment_statistics": episode_environment_statistics[nep],
            }
            for k, v in per_episode_pose_metrics.items():
                per_episode_metrics[k] = v[nep].item()
            per_episode_statistics[nep].append(per_episode_metrics)

        logging.info(
            "======= Evaluating for {} episodes ========".format(
                num_eval_batches * num_processes
            )
        )
        for k, v in metrics.items():
            logging.info("{}: {:.3f}".format(k, v))

    return metrics, per_episode_statistics


def evaluate_tdn_astar(models, envs, config, device, visualize_policy=False):
    # =============== Evaluation configs ======================
    num_steps_exp = config["num_steps_exp"]
    num_steps_nav = config["num_steps_nav"]
    num_processes = 1
    num_eval_episodes = config["num_eval_episodes"]
    env_name = config["env_name"]
    actor_type = config["actor_type"]
    encoder_type = config["encoder_type"]
    feat_shape_sim = config["feat_shape_sim"]
    use_action_embedding = config["use_action_embedding"]
    use_collision_embedding = config["use_collision_embedding"]
    vis_save_dir = config["vis_save_dir"]
    if actor_type == "forward":
        forward_action_id = config["forward_action_id"]
    elif actor_type == "forward-plus":
        forward_action_id = config["forward_action_id"]
        turn_action_id = config["turn_action_id"]
    elif actor_type == "frontier":
        assert num_processes == 1
        if "avd" in env_name:
            action_space = {"forward": 2, "left": 0, "right": 1, "stop": 3}
        else:
            action_space = {"forward": 0, "left": 1, "right": 2, "stop": 3}
        occ_map_scale = config["occ_map_scale"]
        max_time_per_target = config["max_time_per_target"]
        frontier_agent = FrontierAgent(
            action_space,
            env_name,
            occ_map_scale,
            show_animation=False,
            max_time_per_target=max_time_per_target,
        )

    use_policy = (
        actor_type != "random"
        and actor_type != "oracle"
        and actor_type != "forward"
        and actor_type != "forward-plus"
        and actor_type != "frontier"
    )

    # =============== Models ======================
    nav_policy = models["nav_policy"]
    if use_policy:
        encoder = models["encoder"]
        actor_critic = models["actor_critic"]

    # Set to evaluation mode
    if use_policy:
        encoder.eval()
        actor_critic.eval()

    tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    # =============== Gather evaluation info  ======================
    episode_environment_statistics = []
    exp_area_covered = []
    exp_collisions = []
    nav_error_all = []
    s_score_all = []
    spl_score_all = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
        else:
            obs_lm = None
            obs_sm = None
        return obs_im, obs_sm, obs_lm

    # =============== Evaluate over predefined number of episodes  ======================
    obs = envs.reset()
    num_eval_batches = (num_eval_episodes // num_processes) + 1
    for neval in range(num_eval_batches):
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)
        obs_collns = obs["collisions"]
        if actor_type == "frontier":
            delta_ego = torch.zeros((num_processes, 3)).to(device)
            frontier_agent.reset()

        if use_policy:
            recurrent_hidden_states = torch.zeros(num_processes, feat_shape_sim[0]).to(
                device
            )
            masks = torch.zeros(num_processes, 1).to(device)

        nav_policy.reset()

        prev_action = torch.zeros(num_processes, 1).long().to(device)
        prev_collision = obs_collns
        obs_odometer = torch.zeros(num_processes, 4).to(device)
        per_proc_collisions = [0.0 for _ in range(num_processes)]

        # =================================================================
        # ==================== Perform exploration ========================
        # =================================================================
        for step in range(num_steps_exp):
            if use_policy:
                encoder_inputs = [obs_im]
                if encoder_type == "rgb+map":
                    encoder_inputs += [obs_sm, obs_lm]
                with torch.no_grad():
                    policy_feats = encoder(*encoder_inputs)
                    policy_inputs = {"features": policy_feats}
                    if use_action_embedding:
                        policy_inputs["actions"] = prev_action
                    if use_collision_embedding:
                        policy_inputs["collisions"] = prev_collision.long()

                    policy_outputs = actor_critic.act(
                        policy_inputs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=False,
                    )
                    _, action, _, recurrent_hidden_states = policy_outputs
            elif actor_type == "oracle":
                action = obs["oracle_action"].long()
            elif actor_type == "random":
                action = torch.randint(
                    0, envs.action_space.n, (num_processes, 1)
                ).long()
            elif actor_type == "forward":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                action = action.long()
            elif actor_type == "forward-plus":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                collision_mask = prev_collision > 0
                action[collision_mask] = turn_action_id
                action = action.long()
            elif actor_type == "frontier":
                # This assumes that num_processes = 1
                occ_map = obs["highres_coarse_occupancy"][0].cpu().numpy()
                occ_map = occ_map.transpose(1, 2, 0)
                occ_map = np.ascontiguousarray(occ_map)
                occ_map = occ_map.astype(np.uint8)
                action = frontier_agent.act(
                    occ_map, delta_ego[0].cpu().numpy(), prev_collision[0].item()
                )
                action = torch.Tensor([[action]]).long()

            obs, reward, done, infos = envs.step(action)
            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)
            obs_collns = obs["collisions"]

            obs_odometer_curr = process_odometer(obs["delta"])
            if actor_type == "frontier":
                delta_ego = compute_egocentric_coors(
                    obs_odometer_curr, obs_odometer, occ_map_scale,
                )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(num_processes)]).to(device)
            obs_odometer = obs_odometer + obs_odometer_curr

            # This must not reach done = True
            assert done[0] == False

            # Update collisions metric
            for pr in range(num_processes):
                per_proc_collisions[pr] += obs_collns[pr, 0].item()

            prev_collision = obs_collns
            prev_action = action

            # Verifying correctness
            if step == num_steps_exp - 1:
                assert infos[0]["finished_exploration"]
            elif step < num_steps_exp - 1:
                assert not infos[0]["finished_exploration"]
                exploration_topdown_map = infos[0]["topdown_map"]
        # Update Exploration statistics
        for pr in range(num_processes):
            episode_environment_statistics.append(infos[pr]["environment_statistics"])
            exp_area_covered.append(infos[pr]["seen_area"])
            exp_collisions.append(per_proc_collisions[pr])

        # =================================================================
        # ===================== Navigation evaluation =====================
        # =================================================================
        # gather statistics for visualization
        per_proc_rgb = [[] for pr in range(num_processes)]
        per_proc_depth = [[] for pr in range(num_processes)]
        per_proc_fine_occ = [[] for pr in range(num_processes)]
        per_proc_coarse_occ = [[] for pr in range(num_processes)]
        per_proc_topdown_map = [[] for pr in range(num_processes)]
        per_proc_planner_vis = [[] for pr in range(num_processes)]
        per_proc_gt_topdown_map = [[] for pr in range(num_processes)]
        per_proc_initial_planner_vis = [[] for pr in range(num_processes)]
        per_proc_exploration_topdown_map = [[] for pr in range(num_processes)]

        WIDTH, HEIGHT = 300, 300

        nav_policy.reset()

        initial_planning_vis = None
        for t in range(num_steps_nav):
            # Processing environment inputs
            obs_highres_coarse_occ = torch_to_np(obs["highres_coarse_occupancy"][0])
            if t == 0:
                topdown_map = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            else:
                topdown_map = infos[0]["topdown_map"]
            goal_x = int(obs["target_grid_loc"][0, 0].item())
            goal_y = int(obs["target_grid_loc"][0, 1].item())
            coarse_occ_orig = np.flip(obs_highres_coarse_occ, axis=2)
            coarse_occ_orig = np.ascontiguousarray(coarse_occ_orig)

            action = nav_policy.act(
                coarse_occ_orig, (goal_x, goal_y), obs["collisions"][0, 0].item()
            )
            if action == 3:
                logging.info("=====> STOP action called!")
            actions = torch.Tensor([[action]])

            obs, reward, done, infos = envs.step(actions)

            if visualize_policy:
                if t == 0:
                    initial_planning_vis = np.flip(
                        nav_policy.planning_visualization, axis=2
                    )
                for pr in range(num_processes):
                    per_proc_rgb[pr].append(torch_to_np(obs["im"][pr]))
                    if "habitat" in env_name:
                        per_proc_depth[pr].append(
                            torch_to_np_depth(obs["depth"][pr] * 10000.0)
                        )
                    else:
                        per_proc_depth[pr].append(torch_to_np_depth(obs["depth"][pr]))
                    per_proc_fine_occ[pr].append(torch_to_np(obs["fine_occupancy"][pr]))
                    per_proc_coarse_occ[pr].append(
                        torch_to_np(obs["highres_coarse_occupancy"][pr])
                    )
                    per_proc_topdown_map[pr].append(
                        np.flip(infos[pr]["topdown_map"], axis=2)
                    )
                    per_proc_planner_vis[pr].append(
                        np.flip(nav_policy.planning_visualization, axis=2)
                    )
                    per_proc_initial_planner_vis[pr].append(initial_planning_vis)
                    per_proc_exploration_topdown_map[pr].append(
                        np.flip(exploration_topdown_map, axis=2)
                    )

            if done[0] or action == 3:
                nav_error_all.append(infos[0]["nav_error"])
                spl_score_all.append(infos[0]["spl"])
                s_score_all.append(infos[0]["success_rate"])
                break

            if t == num_steps_nav - 1 and not done[0]:
                raise AssertionError("done not being called at end of episode!")

        # Write the episode data to tensorboard
        if visualize_policy:
            proc_fn = lambda x: np.ascontiguousarray(
                np.flip(np.concatenate(x, axis=1), axis=2)
            )
            for pr in range(num_processes):
                rgb_data = per_proc_rgb[pr]
                depth_data = per_proc_depth[pr]
                fine_occ_data = per_proc_fine_occ[pr]
                coarse_occ_data = per_proc_coarse_occ[pr]
                topdown_map_data = per_proc_topdown_map[pr]
                planner_vis_data = per_proc_planner_vis[pr]
                final_topdown_map_data = [
                    topdown_map_data[-1] for _ in range(len(topdown_map_data))
                ]
                initial_planner_vis_data = per_proc_initial_planner_vis[pr]
                exploration_topdown_map_data = per_proc_exploration_topdown_map[pr]

                per_frame_data_proc = zip(
                    rgb_data,
                    coarse_occ_data,
                    topdown_map_data,
                    planner_vis_data,
                    final_topdown_map_data,
                    initial_planner_vis_data,
                    exploration_topdown_map_data,
                )

                video_frames = [
                    proc_fn([cv2.resize(d, (WIDTH, HEIGHT)) for d in per_frame_data])
                    for per_frame_data in per_frame_data_proc
                ]
                tbwriter.add_video_from_np_images(
                    "Episode_{:05d}".format(neval), 0, video_frames, fps=4
                )

        logging.info(
            "===========> Episode done: SPL: {:.3f}, SR: {:.3f}, Nav Err: {:.3f}, Neval: {}".format(
                spl_score_all[-1], s_score_all[-1], nav_error_all[-1], neval
            )
        )

    envs.close()

    # Fill in per-episode statistics
    total_episodes = len(nav_error_all)
    per_episode_statistics = []
    for nep in range(total_episodes):
        per_episode_metrics = {
            "time_step": num_steps_exp,
            "nav_error": nav_error_all[nep],
            "success_rate": s_score_all[nep],
            "spl": spl_score_all[nep],
            "exploration_area_covered": exp_area_covered[nep],
            "exploration_collisions": exp_collisions[nep],
            "environment_statistics": episode_environment_statistics[nep],
        }
        per_episode_statistics.append(per_episode_metrics)

    metrics = {}
    metrics["nav_error"] = np.mean(nav_error_all)
    metrics["spl"] = np.mean(spl_score_all)
    metrics["success_rate"] = np.mean(s_score_all)

    logging.info(
        "======= Evaluating for {} episodes ========".format(
            num_eval_batches * num_processes
        )
    )
    for k, v in metrics.items():
        logging.info("{}: {:.3f}".format(k, v))

    return metrics, per_episode_statistics


def evaluate_tdn_hierarchical_astar(
    models, envs, config, device, visualize_policy=False
):
    # =============== Evaluation configs ======================
    num_steps_exp = config["num_steps_exp"]
    num_steps_nav = config["num_steps_nav"]
    num_processes = 1
    num_eval_episodes = config["num_eval_episodes"]
    env_name = config["env_name"]
    actor_type = config["actor_type"]
    encoder_type = config["encoder_type"]
    feat_shape_sim = config["feat_shape_sim"]
    use_action_embedding = config["use_action_embedding"]
    use_collision_embedding = config["use_collision_embedding"]
    vis_save_dir = config["vis_save_dir"]
    occ_map_scale = config["occ_map_scale"]
    if actor_type == "forward":
        forward_action_id = config["forward_action_id"]
    elif actor_type == "forward-plus":
        forward_action_id = config["forward_action_id"]
        turn_action_id = config["turn_action_id"]
    elif actor_type == "frontier":
        assert num_processes == 1
        if "avd" in env_name:
            action_space = {"forward": 2, "left": 0, "right": 1, "stop": 3}
        else:
            action_space = {"forward": 0, "left": 1, "right": 2, "stop": 3}
        frontier_dilate_occ = config["frontier_dilate_occ"]
        max_time_per_target = config["max_time_per_target"]

        frontier_agent = FrontierAgent(
            action_space,
            env_name,
            occ_map_scale,
            show_animation=False,
            dilate_occupancy=frontier_dilate_occ,
            max_time_per_target=max_time_per_target,
        )

    use_policy = (
        actor_type != "random"
        and actor_type != "oracle"
        and actor_type != "forward"
        and actor_type != "forward-plus"
        and actor_type != "frontier"
    )

    # =============== Models ======================
    nav_policy = models["nav_policy"]
    if use_policy:
        encoder = models["encoder"]
        actor_critic = models["actor_critic"]

    # Set to evaluation mode
    if use_policy:
        encoder.eval()
        actor_critic.eval()

    tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    per_episode_statistics = []
    # =============== Gather evaluation info  ======================
    episode_environment_statistics = []
    exp_area_covered = []
    exp_collisions = []
    nav_error_all = []
    s_score_all = []
    spl_score_all = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
        else:
            obs_lm = None
            obs_sm = None
        return obs_im, obs_sm, obs_lm

    # =============== Evaluate over predefined number of episodes  ======================
    num_eval_batches = (num_eval_episodes // num_processes) + 1
    times_per_episode = []
    for neval in range(num_eval_batches):
        ep_start_time = time.time()
        # =================== Start a new episode ====================
        if "habitat" not in env_name or neval == 0:
            # Habitat calls reset after done automatically. Account for that.
            obs = envs.reset()
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)
        obs_collns = obs["collisions"]
        if actor_type == "frontier":
            delta_ego = torch.zeros((num_processes, 3)).to(device)
            frontier_agent.reset()

        if use_policy:
            recurrent_hidden_states = torch.zeros(num_processes, feat_shape_sim[0]).to(
                device
            )
            masks = torch.zeros(num_processes, 1).to(device)

        nav_policy.reset()

        prev_action = torch.zeros(num_processes, 1).long().to(device)
        prev_collision = obs_collns
        obs_odometer = torch.zeros(num_processes, 4).to(device)
        per_proc_collisions = [0.0 for _ in range(num_processes)]

        # =================== Perform exploration ========================
        for step in range(num_steps_exp):
            if use_policy:
                encoder_inputs = [obs_im]
                if encoder_type == "rgb+map":
                    encoder_inputs += [obs_sm, obs_lm]
                with torch.no_grad():
                    policy_feats = encoder(*encoder_inputs)
                    policy_inputs = {"features": policy_feats}
                    if use_action_embedding:
                        policy_inputs["actions"] = prev_action
                    if use_collision_embedding:
                        policy_inputs["collisions"] = prev_collision.long()

                    policy_outputs = actor_critic.act(
                        policy_inputs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=False,
                    )
                    _, action, _, recurrent_hidden_states = policy_outputs
            elif actor_type == "oracle":
                action = obs["oracle_action"].long()
            elif actor_type == "random":
                action = torch.randint(
                    0, envs.action_space.n, (num_processes, 1)
                ).long()
            elif actor_type == "forward":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                action = action.long()
            elif actor_type == "forward-plus":
                action = torch.Tensor(np.ones((num_processes, 1)) * forward_action_id)
                collision_mask = prev_collision > 0
                action[collision_mask] = turn_action_id
                action = action.long()
            elif actor_type == "frontier":
                # This assumes that num_processes = 1
                occ_map = obs["highres_coarse_occupancy"][0].cpu().numpy()
                occ_map = occ_map.transpose(1, 2, 0)
                occ_map = np.ascontiguousarray(occ_map)
                occ_map = occ_map.astype(np.uint8)
                action = frontier_agent.act(
                    occ_map, delta_ego[0].cpu().numpy(), prev_collision[0].item()
                )
                action = torch.Tensor([[action]]).long()

            obs, reward, done, infos = envs.step(action)
            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)
            obs_collns = obs["collisions"]

            obs_odometer_curr = process_odometer(obs["delta"])
            if actor_type == "frontier":
                delta_ego = compute_egocentric_coors(
                    obs_odometer_curr, obs_odometer, occ_map_scale,
                )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(num_processes)]).to(device)
            obs_odometer = obs_odometer + obs_odometer_curr

            # This must not reach done = True
            assert done[0] == False

            # Update collisions metric
            for pr in range(num_processes):
                per_proc_collisions[pr] += obs_collns[pr, 0].item()

            prev_collision = obs_collns
            prev_action = action

            # Debug stuff
            if step == num_steps_exp - 1:
                assert infos[0]["finished_exploration"]
            elif step < num_steps_exp - 1:
                assert not infos[0]["finished_exploration"]
                exploration_topdown_map = infos[0]["topdown_map"]

        # =================================================================
        # ===================== Navigation evaluation =====================
        # =================================================================

        # Exploration statistics
        for pr in range(num_processes):
            episode_environment_statistics.append(infos[pr]["environment_statistics"])
            exp_area_covered.append(infos[pr]["seen_area"])
            exp_collisions.append(per_proc_collisions[pr])

        # gather statistics for visualization
        per_proc_rgb = [[] for pr in range(num_processes)]
        per_proc_depth = [[] for pr in range(num_processes)]
        per_proc_fine_occ = [[] for pr in range(num_processes)]
        per_proc_coarse_occ = [[] for pr in range(num_processes)]
        per_proc_topdown_map = [[] for pr in range(num_processes)]
        per_proc_planner_vis = [[] for pr in range(num_processes)]
        per_proc_gt_topdown_map = [[] for pr in range(num_processes)]
        per_proc_initial_planner_vis = [[] for pr in range(num_processes)]
        per_proc_exploration_topdown_map = [[] for pr in range(num_processes)]

        WIDTH, HEIGHT = 300, 300

        nav_policy.reset()
        delta_ego = torch.zeros((num_processes, 3)).to(device)
        obs_odometer = torch.zeros(num_processes, 4).to(device)

        # planned_trajectory_lengths = []
        initial_planning_vis = None
        for t in range(num_steps_nav):
            # Processing environment inputs
            obs_rgb = torch_to_np(obs["im"][0])
            obs_depth = torch_to_np_depth(obs["depth"][0])
            obs_fine_occ = torch_to_np(obs["fine_occupancy"][0])
            obs_coarse_occ = torch_to_np(obs["coarse_occupancy"][0])
            obs_highres_coarse_occ = torch_to_np(obs["highres_coarse_occupancy"][0])
            obs_gt_highres_coarse_occ = (
                obs["gt_highres_coarse_occupancy"][0].cpu().numpy()
            )
            # Select portions of obs_gt_highres_coarse_occ that are visible in obs_highres_coarse_occ
            occ_mask = np.all(
                obs_highres_coarse_occ == np.array([255, 0, 0]), axis=-1
            )  # BGR
            free_mask = np.all(obs_highres_coarse_occ == np.array([0, 255, 0]), axis=-1)
            known_mask = (occ_mask | free_mask).astype(np.uint8) * 255
            # Dilate known_mask to extract avoid missing walls due to minor errors
            dkernel = np.ones((9, 9), np.uint8)
            known_mask = cv2.dilate(known_mask, dkernel, iterations=2) > 0
            gt_free_mask = obs_gt_highres_coarse_occ == 0.0
            gt_occ_mask = obs_gt_highres_coarse_occ == 1.0
            new_highres_coarse_occ = np.ones_like(obs_highres_coarse_occ) * 255  # BGR
            np.putmask(new_highres_coarse_occ[..., 0], gt_free_mask & known_mask, 0)
            np.putmask(new_highres_coarse_occ[..., 1], gt_free_mask & known_mask, 255)
            np.putmask(new_highres_coarse_occ[..., 2], gt_free_mask & known_mask, 0)
            np.putmask(new_highres_coarse_occ[..., 0], gt_occ_mask & known_mask, 255)
            np.putmask(new_highres_coarse_occ[..., 1], gt_occ_mask & known_mask, 0)
            np.putmask(new_highres_coarse_occ[..., 2], gt_occ_mask & known_mask, 0)

            if t == 0:
                topdown_map = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            else:
                topdown_map = infos[0]["topdown_map"]
            goal_x = int(obs["target_grid_loc"][0, 0].item())
            goal_y = int(obs["target_grid_loc"][0, 1].item())
            coarse_occ_orig = np.flip(new_highres_coarse_occ, axis=2)
            coarse_occ_orig = np.ascontiguousarray(coarse_occ_orig)

            action = nav_policy.act(
                coarse_occ_orig,
                (goal_x, goal_y),
                delta_ego[0].cpu().numpy(),
                obs["collisions"][0, 0].item(),
            )
            # if nav_policy._prev_path_length is not None:
            #    planned_trajectory_lengths.append(nav_policy._prev_path_length)

            if action == 3:
                logging.info("=====> STOP action called!")
            actions = torch.Tensor([[action]])

            obs, reward, done, infos = envs.step(actions)
            obs_odometer_curr = process_odometer(obs["delta"])
            delta_ego = compute_egocentric_coors(
                obs_odometer_curr, obs_odometer, occ_map_scale,
            )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            obs_odometer = obs_odometer + obs_odometer_curr

            if action == 3:
                assert done[0]

            if visualize_policy:
                if t == 0:
                    initial_planning_vis = np.flip(
                        nav_policy.planning_visualization, axis=2
                    )
                for pr in range(num_processes):
                    per_proc_rgb[pr].append(torch_to_np(obs["im"][pr]))
                    if "habitat" in env_name:
                        per_proc_depth[pr].append(
                            torch_to_np_depth(obs["depth"][pr] * 10000.0)
                        )
                    else:
                        per_proc_depth[pr].append(torch_to_np_depth(obs["depth"][pr]))
                    per_proc_fine_occ[pr].append(torch_to_np(obs["fine_occupancy"][pr]))
                    per_proc_coarse_occ[pr].append(
                        torch_to_np(obs["highres_coarse_occupancy"][pr])
                    )
                    per_proc_topdown_map[pr].append(
                        np.flip(infos[pr]["topdown_map"], axis=2)
                    )
                    per_proc_planner_vis[pr].append(
                        np.flip(nav_policy.planning_visualization, axis=2)
                    )
                    per_proc_initial_planner_vis[pr].append(initial_planning_vis)
                    per_proc_exploration_topdown_map[pr].append(
                        np.flip(exploration_topdown_map, axis=2)
                    )

            if done[0] or action == 3:
                nav_error_all.append(infos[0]["nav_error"])
                spl_score_all.append(infos[0]["spl"])
                s_score_all.append(infos[0]["success_rate"])
                break
        # Write the episode data to tensorboard
        if visualize_policy:
            proc_fn = lambda x: np.ascontiguousarray(
                np.flip(np.concatenate(x, axis=1), axis=2)
            )
            for pr in range(num_processes):
                rgb_data = per_proc_rgb[pr]
                depth_data = per_proc_depth[pr]
                fine_occ_data = per_proc_fine_occ[pr]
                coarse_occ_data = per_proc_coarse_occ[pr]
                topdown_map_data = per_proc_topdown_map[pr]
                planner_vis_data = per_proc_planner_vis[pr]
                final_topdown_map_data = [
                    topdown_map_data[-1] for _ in range(len(topdown_map_data))
                ]
                initial_planner_vis_data = per_proc_initial_planner_vis[pr]
                exploration_topdown_map_data = per_proc_exploration_topdown_map[pr]

                per_frame_data_proc = zip(
                    rgb_data,
                    coarse_occ_data,
                    topdown_map_data,
                    planner_vis_data,
                    final_topdown_map_data,
                    initial_planner_vis_data,
                    exploration_topdown_map_data,
                )

                video_frames = [
                    proc_fn([cv2.resize(d, (WIDTH, HEIGHT)) for d in per_frame_data])
                    for per_frame_data in per_frame_data_proc
                ]
                tbwriter.add_video_from_np_images(
                    "Episode_{:05d}".format(neval), 0, video_frames, fps=4
                )

        # End of episode
        times_per_episode.append(time.time() - ep_start_time)
        mins_per_episode = np.mean(times_per_episode).item() / 60.0
        eta_completion = mins_per_episode * (num_eval_batches - neval)
        logging.info(
            f"============> Num episodes done: {neval+1}/{num_eval_batches}, Avg time per episode: {mins_per_episode:.3f} mins, ETA completion: {eta_completion:.3f} mins"
        )
        logging.info(
            "===========> Episode done: SPL: {:.3f}, SR: {:.3f}, Nav Err: {:.3f}, Area covered: {:.3f}, Neval: {}".format(
                spl_score_all[-1],
                s_score_all[-1],
                nav_error_all[-1],
                exp_area_covered[-1],
                neval,
            )
        )

    envs.close()

    # Fill in per-episode statistics
    total_episodes = len(nav_error_all)
    per_episode_statistics = []
    for nep in range(total_episodes):
        per_episode_metrics = {
            "time_step": num_steps_exp,
            "nav_error": nav_error_all[nep],
            "success_rate": s_score_all[nep],
            "spl": spl_score_all[nep],
            "exploration_area_covered": exp_area_covered[nep],
            "exploration_collisions": exp_collisions[nep],
            "environment_statistics": episode_environment_statistics[nep],
        }
        per_episode_statistics.append(per_episode_metrics)

    metrics = {}
    metrics["nav_error"] = np.mean(nav_error_all)
    metrics["spl"] = np.mean(spl_score_all)
    metrics["success_rate"] = np.mean(s_score_all)

    logging.info(
        "======= Evaluating for {} episodes ========".format(
            num_eval_batches * num_processes
        )
    )
    for k, v in metrics.items():
        logging.info("{}: {:.3f}".format(k, v))

    return metrics, per_episode_statistics


def evaluate_visitation(
    models,
    envs,
    config,
    device,
    multi_step=False,
    interval_steps=None,
    visualize_policy=True,
    visualize_size=200,
    visualize_batches=8,
    visualize_n_per_batch=1,
):
    # =============== Evaluation configs ======================
    num_steps = config["num_steps"]
    feat_shape_sim = config["feat_shape_sim"]
    NPROC = config["num_processes"]
    NREF = config["num_pose_refs"]
    num_eval_episodes = config["num_eval_episodes"]
    env_name = config["env_name"]
    actor_type = config["actor_type"]
    encoder_type = config["encoder_type"]
    use_action_embedding = config["use_action_embedding"]
    use_collision_embedding = config["use_collision_embedding"]
    vis_save_dir = config["vis_save_dir"]
    if "final_topdown_save_path" in config:
        final_topdown_save_path = config["final_topdown_save_path"]
    else:
        final_topdown_save_path = None
    rescale_image_flag = config.get("input_highres", False)

    if actor_type == "forward":
        forward_action_id = config["forward_action_id"]
    elif actor_type == "forward-plus":
        forward_action_id = config["forward_action_id"]
        turn_action_id = config["turn_action_id"]
    elif actor_type == "frontier":
        assert NPROC == 1
        if "avd" in env_name:
            action_space = {"forward": 2, "left": 0, "right": 1, "stop": 3}
        else:
            action_space = {"forward": 0, "left": 1, "right": 2, "stop": 3}
        occ_map_scale = config["occ_map_scale"]
        frontier_dilate_occ = config["frontier_dilate_occ"]
        max_time_per_target = config["max_time_per_target"]

        frontier_agent = FrontierAgent(
            action_space,
            env_name,
            occ_map_scale,
            show_animation=False,
            dilate_occupancy=frontier_dilate_occ,
            max_time_per_target=max_time_per_target,
        )

    use_policy = (
        actor_type != "random"
        and actor_type != "oracle"
        and actor_type != "forward"
        and actor_type != "forward-plus"
        and actor_type != "frontier"
    )

    # =============== Models ======================
    if use_policy:
        encoder = models["encoder"]
        actor_critic = models["actor_critic"]

    # Set to evaluation mode
    if use_policy:
        encoder.eval()
        actor_critic.eval()

    tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    per_episode_statistics = []
    # =============== Gather evaluation info  ======================
    episode_environment_statistics = []
    if multi_step:
        episode_osr_all = {interval: [] for interval in interval_steps}
        episode_collns_all = {interval: [] for interval in interval_steps}
        episode_area_all = {interval: [] for interval in interval_steps}
        episode_objects_all = {interval: [] for interval in interval_steps}
        episode_objects_small_all = {interval: [] for interval in interval_steps}
        episode_objects_medium_all = {interval: [] for interval in interval_steps}
        episode_objects_large_all = {interval: [] for interval in interval_steps}
        episode_categories_all = {interval: [] for interval in interval_steps}
    else:
        episode_osr_all = []
        episode_collns_all = []
        episode_area_all = []
        episode_objects_all = []
        episode_objects_small_all = []
        episode_objects_medium_all = []
        episode_objects_large_all = []
        episode_categories_all = []
    episode_final_topdown_map_all = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if rescale_image_flag:
            obs_im = resize_image(obs_im)
        if encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
            if rescale_image_flag:
                obs_lm = resize_image(obs_lm)
                obs_sm = resize_image(obs_sm)
        else:
            obs_lm = None
            obs_sm = None

        return obs_im, obs_sm, obs_lm

    num_eval_batches = (num_eval_episodes // NPROC) + 1
    times_per_episode = []
    for neval in range(num_eval_batches):
        ep_start_time = time.time()
        obs = envs.reset()
        # Processing observations
        obs_im, obs_sm, obs_lm = get_obs(obs)
        obs_collns = obs["collisions"].long()  # (NPROC, 1)
        obs_odometer = process_odometer(obs["delta"])
        if actor_type == "frontier":
            delta_ego = torch.zeros((NPROC, 3)).to(device)
            frontier_agent.reset()
        if use_policy:
            recurrent_hidden_states = torch.zeros(NPROC, feat_shape_sim[0]).to(device)
            masks = torch.zeros(NPROC, 1).to(device)
        stepwise_osr = [[] for pr in range(NPROC)]
        stepwise_collisions = [[] for pr in range(NPROC)]
        stepwise_objects = [[] for pr in range(NPROC)]
        stepwise_objects_small = [[] for pr in range(NPROC)]
        stepwise_objects_medium = [[] for pr in range(NPROC)]
        stepwise_objects_large = [[] for pr in range(NPROC)]
        stepwise_categories = [[] for pr in range(NPROC)]
        stepwise_area = [[] for pr in range(NPROC)]
        for pr in range(NPROC):
            stepwise_collisions[pr].append(obs_collns[pr, 0].item())
        # gather statistics for visualization
        per_proc_rgb = [[] for pr in range(NPROC)]
        per_proc_depth = [[] for pr in range(NPROC)]
        per_proc_fine_occ = [[] for pr in range(NPROC)]
        per_proc_coarse_occ = [[] for pr in range(NPROC)]
        per_proc_topdown_map = [[] for pr in range(NPROC)]

        prev_action = torch.zeros(NPROC, 1).long().to(device)
        prev_collision = obs_collns

        for step in range(num_steps):
            if use_policy:
                encoder_inputs = [obs_im]
                if encoder_type == "rgb+map":
                    encoder_inputs += [obs_sm, obs_lm]
                with torch.no_grad():
                    obs_feats = encoder(*encoder_inputs)
                    policy_inputs = {"features": obs_feats}
                    if use_action_embedding:
                        policy_inputs["actions"] = prev_action.long()
                    if use_collision_embedding:
                        policy_inputs["collisions"] = prev_collision.long()

                    policy_outputs = actor_critic.act(
                        policy_inputs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=False,
                    )
                    _, action, _, recurrent_hidden_states = policy_outputs
            elif actor_type == "oracle":
                action = obs["oracle_action"].long()
            elif actor_type == "random":
                action = torch.Tensor(
                    np.random.randint(0, envs.action_space.n, size=(NPROC, 1))
                ).long()
            elif actor_type == "forward":
                action = torch.Tensor(np.ones((NPROC, 1)) * forward_action_id)
                action = action.long()

            elif actor_type == "forward-plus":
                action = torch.Tensor(np.ones((NPROC, 1)) * forward_action_id)
                collision_mask = prev_collision > 0
                action[collision_mask] = turn_action_id
                action = action.long()

            elif actor_type == "frontier":
                # This assumes that NPROC = 1
                occ_map = asnumpy(obs["highres_coarse_occupancy"][0])
                occ_map = rearrange(occ_map, "c h w -> h w c").astype(np.uint8)
                action = frontier_agent.act(
                    occ_map, asnumpy(delta_ego[0]), prev_collision[0].item()
                )
                action = torch.Tensor([[action]]).long()

            obs, reward, done, infos = envs.step(action)

            for pr in range(NPROC):
                if visualize_policy:
                    per_proc_rgb[pr].append(torch_to_np(obs["im"][pr]))
                    per_proc_depth[pr].append(torch_to_np_depth(obs["depth"][pr]))
                    per_proc_fine_occ[pr].append(torch_to_np(obs["fine_occupancy"][pr]))
                    per_proc_coarse_occ[pr].append(
                        torch_to_np(obs["coarse_occupancy"][pr])
                    )
                    per_proc_topdown_map[pr].append(
                        np.flip(infos[pr]["topdown_map"], axis=2)
                    )
                if step == 0:
                    episode_environment_statistics.append(
                        infos[pr]["environment_statistics"]
                    )
                if step == num_steps - 1:
                    episode_final_topdown_map_all.append(
                        cv2.resize(
                            np.flip(infos[pr]["topdown_map"], axis=2), (200, 200)
                        )
                    )

            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)
            obs_odometer_curr = process_odometer(obs["delta"])
            obs_collns = obs["collisions"]  # (N, 1)
            if actor_type == "frontier":
                delta_ego = compute_egocentric_coors(
                    obs_odometer_curr, obs_odometer, occ_map_scale,
                )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            # Update metrics
            for pr in range(NPROC):
                stepwise_osr[pr].append(infos[pr]["oracle_pose_success"])
                stepwise_collisions[pr].append(obs_collns[pr, 0].item())
                if "num_objects_visited" in infos[pr]:
                    stepwise_objects[pr].append(infos[pr]["num_objects_visited"])
                    stepwise_objects_small[pr].append(
                        infos[pr]["num_small_objects_visited"]
                    )
                    stepwise_objects_medium[pr].append(
                        infos[pr]["num_medium_objects_visited"]
                    )
                    stepwise_objects_large[pr].append(
                        infos[pr]["num_large_objects_visited"]
                    )
                else:
                    stepwise_objects[pr].append(0.0)
                    stepwise_objects_small[pr].append(0.0)
                    stepwise_objects_medium[pr].append(0.0)
                    stepwise_objects_large[pr].append(0.0)
                if "num_categories_visited" in infos[pr]:
                    stepwise_categories[pr].append(infos[pr]["num_categories_visited"])
                else:
                    stepwise_categories[pr].append(0.0)
                stepwise_area[pr].append(infos[pr]["seen_area"])

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(NPROC)]).to(device)
            # Accumulate odometer readings to give relative pose
            # from the starting point
            obs_odometer = obs_odometer * masks + obs_odometer_curr

            # This must not reach done = True
            assert done[0] == False
            # Update prev values
            prev_collision = obs_collns
            prev_action = action

        # Write the episode data to tensorboard
        if neval < visualize_batches and visualize_policy:
            WIDTH = visualize_size
            HEIGHT = visualize_size
            proc_fn = lambda x: np.ascontiguousarray(
                np.flip(np.concatenate(x, axis=1), axis=2)
            )
            num_to_save_per_batch = visualize_n_per_batch
            for pr in range(num_to_save_per_batch):
                rgb_data = per_proc_rgb[pr]
                depth_data = per_proc_depth[pr]
                fine_occ_data = per_proc_fine_occ[pr]
                coarse_occ_data = per_proc_coarse_occ[pr]
                topdown_map_data = per_proc_topdown_map[pr]
                final_topdown_map_data = [
                    topdown_map_data[-1] for _ in range(len(topdown_map_data))
                ]
                per_frame_data_proc = zip(
                    rgb_data,
                    depth_data,
                    fine_occ_data,
                    coarse_occ_data,
                    topdown_map_data,
                    final_topdown_map_data,
                )
                video_frames = [
                    proc_fn([cv2.resize(d, (WIDTH, HEIGHT)) for d in per_frame_data])
                    for per_frame_data in per_frame_data_proc
                ]
                tbwriter.add_video_from_np_images(
                    "Episode_{:05d}".format(neval * num_to_save_per_batch + pr),
                    0,
                    video_frames,
                    fps=4,
                )

        stepwise_collisions = np.array(stepwise_collisions)  # (N, T)
        # ========== Update metrics ============
        if multi_step:
            for interval in interval_steps:
                for pr in range(NPROC):
                    # Landmarks covered
                    episode_osr_all[interval].append(stepwise_osr[pr][interval - 1])
                    # Collisions per episode
                    episode_collns_all[interval].append(
                        stepwise_collisions[pr][:interval].sum()
                    )
                    # Area covered
                    episode_area_all[interval].append(stepwise_area[pr][interval - 1])
                    # Objects covered
                    episode_objects_all[interval].append(
                        stepwise_objects[pr][interval - 1]
                    )
                    # Small objects covered
                    episode_objects_small_all[interval].append(
                        stepwise_objects_small[pr][interval - 1]
                    )
                    # Medium objects covered
                    episode_objects_medium_all[interval].append(
                        stepwise_objects_medium[pr][interval - 1]
                    )
                    # Large objects covered
                    episode_objects_large_all[interval].append(
                        stepwise_objects_large[pr][interval - 1]
                    )
                    # Categories covered
                    episode_categories_all[interval].append(
                        stepwise_categories[pr][interval - 1]
                    )
        else:
            for pr in range(NPROC):
                # Landmarks covered
                episode_osr_all.append(stepwise_osr[pr][-1])
                # Collisions per episode
                episode_collns_all.append(stepwise_collisions[pr].sum())
                # Area covered
                episode_area_all.append(stepwise_area[pr][-1])
                # Objects covered
                episode_objects_all.append(stepwise_objects[pr][-1])
                # Small objects covered
                episode_objects_small_all.append(stepwise_objects_small[pr][-1])
                # Medium objects covered
                episode_objects_medium_all.append(stepwise_objects_medium[pr][-1])
                # Large objects covered
                episode_objects_large_all.append(stepwise_objects_large[pr][-1])
                # Categories covered
                episode_categories_all.append(stepwise_categories[pr][-1])

        # End of episode
        times_per_episode.append(time.time() - ep_start_time)
        mins_per_episode = np.mean(times_per_episode).item() / 60.0
        eta_completion = mins_per_episode * (num_eval_batches - neval)
        neps_done = (neval + 1) * NPROC
        neps_total = num_eval_batches * NPROC

        logging.info(
            f"=====> Episodes done: {neps_done}/{neps_total}, Time per episode: {mins_per_episode:.3f} mins, ETA completion: {eta_completion:.3f} mins"
        )

    envs.close()

    if multi_step:
        metrics = {interval: {} for interval in interval_steps}

        # Fill in per-episode statistics
        total_episodes = np.array(episode_collns_all[interval_steps[0]]).shape[0]
        per_episode_statistics = [
            [] for _ in range(total_episodes)
        ]  # Each episode can have results over multiple intervals
        for interval in interval_steps:
            episode_osr_all[interval] = np.array(episode_osr_all[interval])
            episode_collns_all[interval] = np.array(episode_collns_all[interval])
            episode_area_all[interval] = np.array(episode_area_all[interval])
            episode_objects_all[interval] = np.array(episode_objects_all[interval])
            episode_objects_small_all[interval] = np.array(
                episode_objects_small_all[interval]
            )
            episode_objects_medium_all[interval] = np.array(
                episode_objects_medium_all[interval]
            )
            episode_objects_large_all[interval] = np.array(
                episode_objects_large_all[interval]
            )
            episode_categories_all[interval] = np.array(
                episode_categories_all[interval]
            )

            logging.info(
                "======= Evaluating at {} steps for {} episodes ========".format(
                    interval, num_eval_batches * NPROC
                )
            )
            metrics[interval]["landmarks_covered"] = np.mean(episode_osr_all[interval])
            metrics[interval]["collisions"] = np.mean(episode_collns_all[interval])
            metrics[interval]["area_covered"] = np.mean(episode_area_all[interval])
            metrics[interval]["objects_covered"] = np.mean(
                episode_objects_all[interval]
            )
            metrics[interval]["small_objects_covered"] = np.mean(
                episode_objects_small_all[interval]
            )
            metrics[interval]["medium_objects_covered"] = np.mean(
                episode_objects_medium_all[interval]
            )
            metrics[interval]["large_objects_covered"] = np.mean(
                episode_objects_large_all[interval]
            )
            metrics[interval]["categories_covered"] = np.mean(
                episode_categories_all[interval]
            )

            for k, v in metrics[interval].items():
                logging.info("{}: {:.3f}".format(k, v))

            for nep in range(total_episodes):
                per_episode_metrics = {
                    "time_step": interval,
                    "landmarks_covered": episode_osr_all[interval][nep].item(),
                    "collisions": episode_collns_all[interval][nep].item(),
                    "area_covered": episode_area_all[interval][nep].item(),
                    "objects_covered": episode_objects_all[interval][nep].item(),
                    "small_objects_covered": episode_objects_small_all[interval][
                        nep
                    ].item(),
                    "medium_objects_covered": episode_objects_medium_all[interval][
                        nep
                    ].item(),
                    "large_objects_covered": episode_objects_large_all[interval][
                        nep
                    ].item(),
                    "categories_covered": episode_categories_all[interval][nep].item(),
                    "environment_statistics": episode_environment_statistics[nep],
                }

                per_episode_statistics[nep].append(per_episode_metrics)
    else:
        episode_osr_all = np.array(episode_osr_all)
        episode_collns_all = np.array(episode_collns_all)
        episode_area_all = np.array(episode_area_all)
        episode_objects_all = np.array(episode_objects_all)
        episode_objects_small_all = np.array(episode_objects_small_all)
        episode_objects_medium_all = np.array(episode_objects_medium_all)
        episode_objects_large_all = np.array(episode_objects_large_all)
        episode_categories_all = np.array(episode_categories_all)

        metrics = {}
        metrics["osr"] = np.mean(episode_osr_all)
        metrics["collisions"] = np.mean(episode_collns_all)
        metrics["area_covered"] = np.mean(episode_area_all)
        metrics["objects_covered"] = np.mean(episode_objects_all)
        metrics["small_objects_covered"] = np.mean(episode_objects_small_all)
        metrics["medium_objects_covered"] = np.mean(episode_objects_medium_all)
        metrics["large_objects_covered"] = np.mean(episode_objects_large_all)
        metrics["categories_covered"] = np.mean(episode_categories_all)

        # Fill in per-episode statistics
        total_episodes = episode_collns_all.shape[0]
        per_episode_statistics = [
            [] for _ in range(total_episodes)
        ]  # Each episode can have results over multiple intervals
        for nep in range(total_episodes):
            per_episode_metrics = {
                "time_step": num_steps,
                "osr": episode_osr_all[nep].item(),
                "collisions": episode_collns_all[nep].item(),
                "area_covered": episode_area_all[nep].item(),
                "objects_covered": episode_objects_all[nep].item(),
                "small_objects_covered": episode_objects_small_all[nep].item(),
                "medium_objects_covered": episode_objects_medium_all[nep].item(),
                "large_objects_covered": episode_objects_large_all[nep].item(),
                "categories_covered": episode_categories_all[nep].item(),
                "environment_statistics": episode_environment_statistics[nep],
            }
            per_episode_statistics[nep].append(per_episode_metrics)

        logging.info(
            "======= Evaluating for {} episodes ========".format(
                num_eval_batches * NPROC
            )
        )
        for k, v in metrics.items():
            logging.info("{}: {:.3f}".format(k, v))

    if final_topdown_save_path is not None:
        episode_final_topdown_map_all = np.stack(
            episode_final_topdown_map_all, axis=0
        )  # (nepisodes, H, W, 3)

        topdown_h5_file = h5py.File(final_topdown_save_path, "w")
        topdown_h5_file.create_dataset(
            "final_topdown_map", data=episode_final_topdown_map_all
        )
        topdown_h5_file.close()

    return metrics, per_episode_statistics
