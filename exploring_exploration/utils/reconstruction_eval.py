#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import cv2
import torch
import logging
import numpy as np
import torch.nn.functional as F

from exploring_exploration.utils.storage import RolloutStorageReconstruction
from exploring_exploration.utils.geometry import (
    subtract_pose,
    process_odometer,
    compute_egocentric_coors,
)
from exploring_exploration.utils.visualization import (
    draw_border,
    torch_to_np,
    torch_to_np_depth,
)
from exploring_exploration.utils.common import (
    process_image,
    unprocess_image,
    flatten_two,
    unflatten_two,
    unsq_exp,
)
from exploring_exploration.models.frontier_agent import FrontierAgent
from exploring_exploration.utils.visualization import TensorboardWriter
from exploring_exploration.utils.metrics import precision_at_k
from exploring_exploration.utils.reconstruction import masked_mean


def compute_reconstruction_metrics(pred_scores, gt_scores, masks=None):
    """
    Inputs:
        pred_scores - (N, nclasses) logits
        gt_scores   - (N, nclasses) similarity scores
        masks       - (N, ) if available
    """
    metrics = {}
    per_sample_metrics = {}
    for k in [1, 2, 5, 10]:
        per_sample_prec_at_k = precision_at_k(pred_scores, gt_scores, k=k)  # (N, )
        if masks is None:
            metrics[f"precision@{k}"] = per_sample_prec_at_k.mean().item()
            per_sample_metrics[f"precision@{k}"] = per_sample_prec_at_k.cpu().numpy()
        else:
            metrics[f"precision@{k}"] = (per_sample_prec_at_k * masks).sum() / (
                masks.sum() + 1e-10
            )
            per_sample_metrics[f"precision@{k}"] = (
                (per_sample_prec_at_k * masks).cpu().numpy()
            )
    return metrics, per_sample_metrics


def evaluate_reconstruction_oracle(
    models,
    envs,
    config,
    device,
    multi_step=False,
    interval_steps=None,
    visualize_policy=True,
):
    """
    Evaluates the reconstruction performance of an oracle agent.
    """
    # =============== Evaluation configs ======================
    num_steps = config["num_steps"]
    num_processes = config["num_processes"]
    num_eval_episodes = config["num_eval_episodes"]
    num_pose_refs = config["num_pose_refs"]
    cluster_centroids = config["cluster_centroids"]
    clusters2images = config["clusters2images"]
    rec_loss_fn = config["rec_loss_fn"]
    rec_loss_fn_J = config["rec_loss_fn_J"]
    vis_save_dir = config["vis_save_dir"]
    odometer_shape = config["odometer_shape"]
    env_name = config["env_name"]

    # =============== Models ======================
    decoder = models["decoder"]
    pose_encoder = models["pose_encoder"]
    feature_network = models["feature_network"]

    # Set to evaluation mode
    decoder.eval()
    pose_encoder.eval()
    feature_network.eval()

    tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    # =============== Gather evaluation info  ======================
    if multi_step:
        episode_losses = {interval: [] for interval in interval_steps}
        episode_preds = {interval: [] for interval in interval_steps}
        episode_ref_ims = {interval: [] for interval in interval_steps}
        episode_gts = {interval: [] for interval in interval_steps}
        episode_masks = {interval: [] for interval in interval_steps}
    else:
        episode_losses = []
        episode_preds = []
        episode_ref_ims = []
        episode_gts = []
        episode_masks = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        return obs_im

    num_eval_batches = (num_eval_episodes // num_processes) + 1
    nclusters = cluster_centroids.shape[0]

    rollouts = RolloutStorageReconstruction(
        num_steps, num_processes, (nclusters,), odometer_shape, num_pose_refs,
    )

    rollouts.to(device)
    # =========== Evaluate over predefined number of episodes  ==============
    for neval in range(num_eval_batches):
        # =================== Start a new episode ====================
        obs = envs.reset()
        # Processing environment inputs
        obs_im = get_obs(obs)
        obs_odometer = process_odometer(obs["delta"])
        # Convert mm to m for AVD
        if "avd" in env_name:
            obs_odometer[:, :2] /= 1000.0
        # This remains constant throughout the episode
        tgt_poses = process_odometer(flatten_two(obs["pose_regress"]))[:, :3]
        tgt_poses = unflatten_two(
            tgt_poses, num_processes, num_pose_refs
        )  # (N, nRef, 3)
        tgt_masks = obs["valid_masks"].unsqueeze(2)  # (N, nRef, 1)
        tgt_ims = process_image(
            flatten_two(obs["pose_refs"])
        )  # (num_processes * num_pose_refs, ...)
        # Convert mm to m for AVD
        if "avd" in env_name:
            tgt_poses[:, :, :2] /= 1000.0
        with torch.no_grad():
            obs_feat = feature_network(obs_im)
            tgt_feat = feature_network(tgt_ims)
            # Compute similarity scores with all other clusters
            obs_feat = torch.matmul(obs_feat, cluster_centroids.t())  # (N, nclusters)
            tgt_feat = torch.matmul(
                tgt_feat, cluster_centroids.t()
            )  # (N*nRef, nclusters)
        tgt_feat = unflatten_two(tgt_feat, num_processes, num_pose_refs)

        # Initialize the memory of rollouts
        rollouts.reset()
        rollouts.obs_feats[0].copy_(obs_feat)
        rollouts.obs_odometer[0].copy_(obs_odometer)
        rollouts.tgt_poses.copy_(tgt_poses)
        rollouts.tgt_feats.copy_(tgt_feat)
        rollouts.tgt_masks.copy_(tgt_masks)

        for step in range(num_steps):
            action = obs["oracle_action"].long()

            if (multi_step and ((step + 1) in interval_steps)) or (
                (not multi_step) and (step == num_steps - 1)
            ):
                L = step + 1
                obs_feats_curr = rollouts.obs_feats[:L]  # (L, N, feat_dim)
                obs_odometer_curr = rollouts.obs_odometer[
                    :L, :, :3
                ]  # (L, N, 3) --- (y, x, phi)
                # Convert odometer readings tgt_pose's frame of reference
                obs_odometer_exp = unsq_exp(
                    obs_odometer_curr, num_pose_refs, dim=2
                )  # (L, N, nRef, 3)
                obs_odometer_exp = obs_odometer_exp.view(-1, 3)  # (L*N*nRef, 3)
                tgt_poses_exp = unsq_exp(tgt_poses, L, dim=0)  # (L, N, nRef, 3)
                tgt_poses_exp = tgt_poses_exp.view(-1, 3)  # (L*N*nRef, 3)
                obs_relpose = subtract_pose(
                    obs_odometer_exp, tgt_poses_exp
                )  # (L*N*nRef, 3) --- (x, y, phi)

                obs_relpose_enc = pose_encoder(obs_relpose)  # (L*N*nRef, 16)
                obs_relpose_enc = obs_relpose_enc.view(
                    L, num_processes, num_pose_refs, 16
                )  # (L, N, nRef, 16)
                tgt_relpose_enc = torch.zeros(1, *obs_relpose_enc.shape[1:]).to(
                    device
                )  # (1, N, nRef, 16)

                obs_feats_exp = unsq_exp(
                    obs_feats_curr, num_pose_refs, dim=2
                )  # (L, N, nRef, feat_dim)
                obs_feats_exp = obs_feats_exp.view(L, num_processes * num_pose_refs, -1)
                obs_relpose_enc = obs_relpose_enc.view(
                    L, num_processes * num_pose_refs, -1
                )
                tgt_relpose_enc = tgt_relpose_enc.view(
                    1, num_processes * num_pose_refs, -1
                )

                rec_inputs = {
                    "history_image_features": obs_feats_exp,
                    "history_pose_features": obs_relpose_enc,
                    "target_pose_features": tgt_relpose_enc,
                }

                with torch.no_grad():
                    pred_logits = decoder(rec_inputs)  # (1, N*nRef, nclasses)
                    pred_logits = pred_logits.squeeze(0)
                    pred_outputs = unflatten_two(
                        pred_logits, num_processes, num_pose_refs
                    )  # (N, nRef, nclasses)
                    loss = rec_loss_fn(
                        pred_logits,
                        flatten_two(tgt_feat),
                        cluster_centroids,
                        reduction="none",
                        K=rec_loss_fn_J,
                    ).sum(dim=1)

                if multi_step:
                    episode_losses[step + 1].append(loss.cpu())
                    episode_preds[step + 1].append(
                        pred_outputs.cpu().numpy()
                    )  # (N, nRef, nclasses)
                    episode_masks[step + 1].append(
                        tgt_masks.cpu().numpy()
                    )  # (N, nRef, 1)
                    episode_ref_ims[step + 1].append(
                        unflatten_two(tgt_ims, num_processes, num_pose_refs)
                        .cpu()
                        .numpy()
                    )  # (N, nRef, C, H, W)
                    episode_gts[step + 1].append(
                        tgt_feat.cpu().numpy()
                    )  # (N, nRef, feat_dim)
                else:
                    episode_losses.append(loss.cpu())
                    episode_preds.append(pred_outputs.cpu().numpy())
                    episode_masks.append(tgt_masks.cpu().numpy())  # (N, nRef, 1)
                    episode_ref_ims.append(
                        unflatten_two(tgt_ims, num_processes, num_pose_refs)
                        .cpu()
                        .numpy()
                    )  # (N, nRef, C, H, W)
                    episode_gts.append(tgt_feat.cpu().numpy())  # (N, nRef, 2048)

            obs, reward, done, infos = envs.step(action)

            # Processing environment inputs
            obs_im = get_obs(obs)
            obs_odometer_ = process_odometer(obs["delta"])
            # Convert mm to m for AVD
            if "avd" in env_name:
                obs_odometer_[:, :2] /= 1000.0

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(num_processes)]).to(device)
            # Accumulate odometer readings to give relative pose from the starting point
            obs_odometer = obs_odometer * masks + obs_odometer_

            # This must not reach done = True
            # assert(done[0] == False)
            with torch.no_grad():
                obs_feat = feature_network(obs_im)
                # Compute similarity scores with all other clusters
                obs_feat = torch.matmul(
                    obs_feat, cluster_centroids.t()
                )  # (N, nclusters)

            # Update rollouts
            rollouts.insert(obs_feat, obs_odometer)

    envs.close()

    if multi_step:
        metrics = {interval: {} for interval in interval_steps}
        cluster_centroids_np_T = cluster_centroids.cpu().numpy().T
        for interval in interval_steps:
            episode_losses[interval] = torch.cat(
                episode_losses[interval], dim=0
            ).numpy()
            episode_masks[interval] = np.concatenate(
                episode_masks[interval], axis=0
            )  # (N, nRef, 1)
            episode_preds[interval] = np.concatenate(
                episode_preds[interval], axis=0
            )  # (N, nRef, 2048 / nclasses)
            episode_ref_ims[interval] = np.concatenate(
                episode_ref_ims[interval], axis=0
            )  # (N, nRef, C, H, W)
            temp_image = unprocess_image(
                episode_ref_ims[interval].reshape(
                    -1, *episode_ref_ims[interval].shape[2:]
                )
            )  # (N*nRef, H, W, C)
            episode_ref_ims[interval] = temp_image.reshape(
                *episode_ref_ims[interval].shape[:2], *temp_image.shape[1:]
            )  # (N, nRef, H, W, C)
            episode_gts[interval] = np.concatenate(
                episode_gts[interval], axis=0
            )  # (N, nRef, 2048)

            # Retrieve the top_K predicted clusters
            topk_matches = np.argpartition(episode_preds[interval], -5, axis=2)[
                :, :, -5:
            ]  # (N, nRef, 5)
            pred_feat_score = torch.Tensor(
                episode_preds[interval]
            )  # (N, nRef, nclusters)
            gt_feat_score = torch.Tensor(episode_gts[interval])  # (N, nRef, nclusters)
            logging.info(
                "======= Evaluating at {} steps for {} episodes ========".format(
                    interval, num_eval_batches * num_processes
                )
            )
            episode_losses[interval] = episode_losses[interval].reshape(
                -1, num_pose_refs
            )  # (N, nRef)
            metrics[interval]["rec_loss"] = np.sum(
                episode_losses[interval] * episode_masks[interval]
            ) / np.sum(episode_masks[interval])
            episode_masks[interval] = episode_masks[interval][:, :, 0]  # (N, nRef)
            gt_masks = flatten_two(torch.Tensor(episode_masks[interval]))
            pred_feat_score_flattened = flatten_two(pred_feat_score)[gt_masks == 1.0]
            gt_feat_score_flattened = flatten_two(gt_feat_score)[gt_masks == 1.0]
            rec_metrics, _ = compute_reconstruction_metrics(
                pred_feat_score_flattened, gt_feat_score_flattened
            )

            for k, v in rec_metrics.items():
                metrics[interval][k] = v

            for k, v in metrics[interval].items():
                logging.info("{}: {:.3f}".format(k, v))

            # ======== Generate visualizations  ===========
            if interval == interval_steps[-1]:
                # Sample the first reference from each process to visualize
                vis_gt_images = episode_ref_ims[:, 0]  # (N, H, W, C)
                vis_retrieved_clusters = []
                vis_pred_logits = episode_preds[interval][:, 0]  # (N, nclasses)
                vis_pred_scores = F.softmax(
                    torch.Tensor(vis_pred_logits), dim=1
                ).numpy()  # (N, nclasses)
                vis_gt_feats = episode_gts[interval][:, 0]  # (N, feat_dim)
                # Compute similarity between GT features and all clusters
                vis_gt_sim = vis_gt_feats  # (N, nclasses)
                # Top 5 sim scores for GT features
                vis_gt_topk_idxes = np.argpartition(vis_gt_sim, -5, axis=1)[
                    :, -5:
                ]  # (N, 5)

                # Sample clusters predicted by the network
                for j in range(min(vis_gt_sim.shape[0], 12)):
                    vis_gt_image_j = cv2.resize(vis_gt_images[j], (300, 300))
                    # Add zero padding on top for text
                    vis_gt_image_j = np.pad(
                        vis_gt_image_j, ((100, 0), (0, 0), (0, 0)), mode="constant"
                    )
                    gt_sim_text = ",".join(
                        "{:.2f}".format(vis_gt_sim[j, v.item()])
                        for v in vis_gt_topk_idxes[j]
                    )
                    vis_gt_image_j = cv2.putText(
                        vis_gt_image_j,
                        "Best GT sim",
                        (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        thickness=2,
                    )
                    vis_gt_image_j = cv2.putText(
                        vis_gt_image_j,
                        gt_sim_text,
                        (5, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        thickness=2,
                    )

                    proc_retrieved_clusters = [vis_gt_image_j]
                    for k in topk_matches[j][0]:
                        if clusters2images[k].shape[0] == 0:
                            continue
                        # Pick a random set of 9 cluster images
                        random_idxes = np.random.randint(
                            0, clusters2images[k].shape[0], (9,)
                        )
                        ret_images = clusters2images[k][random_idxes]  # (9, H, W, C)
                        H, W = ret_images.shape[1:3]
                        ret_images = ret_images.reshape(
                            3, 3, *ret_images.shape[1:]
                        )  # (3, 3, H, W, C)
                        ret_images = np.ascontiguousarray(
                            ret_images.transpose(0, 2, 1, 3, 4)
                        )
                        ret_images = ret_images.reshape(3 * H, 3 * W, -1)
                        ret_images = draw_border(ret_images[np.newaxis, ...])[0]
                        ret_images = cv2.resize(ret_images, (300, 300))
                        # The similarity score of GT features with the retrieved clusters
                        proc_retrieved_clusters.append(ret_images)
                        gt_sim_score = vis_gt_sim[j][k].item()
                        pred_sim_score = vis_pred_scores[j][k].item()
                        gt_text = f"GT sim: {gt_sim_score:.2f}"
                        pred_text = f"Pred prob: {pred_sim_score:.2f}"
                        # Add zero padding on top for text
                        ret_images = np.pad(
                            ret_images, ((100, 0), (0, 0), (0, 0)), mode="constant"
                        )
                        # Add the text
                        ret_images = cv2.putText(
                            ret_images,
                            gt_text,
                            (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            thickness=2,
                        )
                        ret_images = cv2.putText(
                            ret_images,
                            pred_text,
                            (5, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            thickness=2,
                        )
                        proc_retrieved_clusters.append(ret_images)

                    proc_retrieved_clusters = np.concatenate(
                        proc_retrieved_clusters, axis=1
                    )
                    proc_retrieved_clusters = (
                        proc_retrieved_clusters.astype(np.float32) / 255.0
                    )
                    proc_retrieved_clusters = torch.Tensor(
                        proc_retrieved_clusters
                    ).permute(2, 0, 1)
                    proc_retrieved_clusters = proc_retrieved_clusters.contiguous()
                    vis_retrieved_clusters.append(proc_retrieved_clusters)

                vis_retrieved_clusters = torch.stack(vis_retrieved_clusters, axis=0)
                tbwriter.add_images(
                    f"Reconstructed images @ interval : {interval:04d}",
                    vis_retrieved_clusters,
                    0,
                )

    else:
        metrics = {}
        episode_losses = torch.cat(episode_losses, dim=0).numpy()
        episode_preds = np.concatenate(episode_preds, axis=0)  # (N, nRef, feat_dim)
        episode_ref_ims = np.concatenate(episode_ref_ims, axis=0)  # (N, nRef, C, H, W)
        episode_masks = np.concatenate(episode_masks, axis=0)  # (N, nRef, 1)
        temp_image = unprocess_image(
            episode_ref_ims.reshape(-1, *episode_ref_ims.shape[2:])
        )  # (N*nRef, H, W, C)
        episode_ref_ims = temp_image.reshape(
            *episode_ref_ims.shape[:2], *temp_image.shape[1:]
        )  # (N, nRef, H, W, C)
        episode_gts = np.concatenate(episode_gts, axis=0)  # (N, nRef, feat_dim)
        cluster_centroids_np_T = cluster_centroids.cpu().numpy().T

        # Retrieve the top_K predicted clusters
        topk_matches = np.argpartition(episode_preds, -5, axis=2)[
            :, :, -5:
        ]  # (N, nRef, 5)
        pred_feat_score = torch.Tensor(episode_preds)  # (N, nRef, nclusters)

        gt_feat_score = torch.Tensor(episode_gts)  # (N, nRef, nclusters)

        # ======== Generate visualizations  ===========
        # Sample the first reference from each process to visualize
        vis_gt_images = episode_ref_ims[:, 0]  # (N, H, W, C)
        vis_retrieved_clusters = []
        vis_pred_logits = episode_preds[:, 0]  # (N, nclasses)
        vis_pred_scores = F.softmax(
            torch.Tensor(vis_pred_logits), dim=1
        ).numpy()  # (N, nclasses)
        vis_gt_feats = episode_gts[:, 0]  # (N, feat_dim)
        # Compute similarity between GT features and all clusters
        vis_gt_sim = vis_gt_feats  # (N, nclasses)
        # Top 5 sim scores for GT features
        vis_gt_topk_idxes = np.argpartition(vis_gt_sim, -5, axis=1)[:, -5:]  # (N, 5)

        # Sample clusters predicted by the network
        for j in range(min(vis_gt_sim.shape[0], 12)):
            vis_gt_image_j = cv2.resize(vis_gt_images[j], (300, 300))
            # Add zero padding on top for text
            vis_gt_image_j = np.pad(
                vis_gt_image_j, ((100, 0), (0, 0), (0, 0)), mode="constant"
            )
            gt_sim_text = ",".join(
                "{:.2f}".format(vis_gt_sim[j, v.item()]) for v in vis_gt_topk_idxes[j]
            )
            vis_gt_image_j = cv2.putText(
                vis_gt_image_j,
                "Best GT sim",
                (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                thickness=2,
            )
            vis_gt_image_j = cv2.putText(
                vis_gt_image_j,
                gt_sim_text,
                (5, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                thickness=2,
            )

            proc_retrieved_clusters = [vis_gt_image_j]
            for k in topk_matches[j][0]:
                if clusters2images[k].shape[0] == 0:
                    continue
                # Pick a random set of 9 cluster images
                random_idxes = np.random.randint(0, clusters2images[k].shape[0], (9,))
                ret_images = clusters2images[k][random_idxes]  # (9, H, W, C)
                H, W = ret_images.shape[1:3]
                ret_images = ret_images.reshape(
                    3, 3, *ret_images.shape[1:]
                )  # (3, 3, H, W, C)
                ret_images = np.ascontiguousarray(ret_images.transpose(0, 2, 1, 3, 4))
                ret_images = ret_images.reshape(3 * H, 3 * W, -1)
                ret_images = draw_border(ret_images[np.newaxis, ...])[0]
                ret_images = cv2.resize(ret_images, (300, 300))
                # The similarity score of GT features with the retrieved clusters
                gt_sim_score = vis_gt_sim[j][k].item()
                pred_sim_score = vis_pred_scores[j][k].item()
                gt_text = f"GT sim: {gt_sim_score:.2f}"
                pred_text = f"Pred prob: {pred_sim_score:.2f}"
                # Add zero padding on top for text
                ret_images = np.pad(
                    ret_images, ((100, 0), (0, 0), (0, 0)), mode="constant"
                )
                # Add the text
                ret_images = cv2.putText(
                    ret_images,
                    gt_text,
                    (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                )
                ret_images = cv2.putText(
                    ret_images,
                    pred_text,
                    (5, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                )
                proc_retrieved_clusters.append(ret_images)

            proc_retrieved_clusters = np.concatenate(proc_retrieved_clusters, axis=1)
            proc_retrieved_clusters = proc_retrieved_clusters.astype(np.float32) / 255.0
            proc_retrieved_clusters = torch.Tensor(proc_retrieved_clusters).permute(
                2, 0, 1
            )
            proc_retrieved_clusters = proc_retrieved_clusters.contiguous()
            vis_retrieved_clusters.append(proc_retrieved_clusters)

        vis_retrieved_clusters = torch.stack(vis_retrieved_clusters, axis=0)
        tbwriter.add_images("Reconstructed images", vis_retrieved_clusters, 0)

        logging.info(
            "======= Evaluating for {} episodes ========".format(
                num_eval_batches * num_processes
            )
        )
        # =================== Update metrics =====================
        metrics["rec_loss"] = np.mean(episode_losses).item()
        episode_masks = episode_masks[:, :, 0]  # (N, nRef)
        gt_masks = flatten_two(torch.Tensor(episode_masks))
        pred_feat_score_flattened = flatten_two(pred_feat_score)[gt_masks == 1.0]
        gt_feat_score_flattened = flatten_two(gt_feat_score)[gt_masks == 1.0]
        rec_metrics, _ = compute_reconstruction_metrics(
            pred_feat_score_flattened, gt_feat_score_flattened
        )
        for k, v in rec_metrics.items():
            metrics[k] = v

        for k, v in metrics.items():
            logging.info("{}: {:.3f}".format(k, v))

    tbwriter.close()
    return metrics


def evaluate_reconstruction(
    models,
    envs,
    config,
    device,
    multi_step=False,
    interval_steps=None,
    visualize_policy=True,
):
    # =============== Evaluation configs ======================
    num_steps = config["num_steps"]
    N = config["num_processes"]
    feat_shape_sim = config["feat_shape_sim"]
    odometer_shape = config["odometer_shape"]
    num_eval_episodes = config["num_eval_episodes"]
    nRef = config["num_pose_refs"]
    env_name = config["env_name"]
    actor_type = config["actor_type"]
    encoder_type = config["encoder_type"]
    use_action_embedding = config["use_action_embedding"]
    use_collision_embedding = config["use_collision_embedding"]
    cluster_centroids = config["cluster_centroids"]
    clusters2images = config["clusters2images"]
    rec_loss_fn = config["rec_loss_fn"]
    vis_save_dir = config["vis_save_dir"]
    if actor_type == "forward":
        forward_action_id = config["forward_action_id"]
    elif actor_type == "forward-plus":
        forward_action_id = config["forward_action_id"]
        turn_action_id = config["turn_action_id"]
    elif actor_type == "frontier":
        assert N == 1
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

    nclusters = cluster_centroids.shape[0]
    use_policy = (
        actor_type != "random"
        and actor_type != "oracle"
        and actor_type != "forward"
        and actor_type != "forward-plus"
        and actor_type != "frontier"
    )

    # =============== Models ======================
    decoder = models["decoder"]
    pose_encoder = models["pose_encoder"]
    feature_network = models["feature_network"]
    if use_policy:
        encoder = models["encoder"]
        actor_critic = models["actor_critic"]

    # Set to evaluation mode
    decoder.eval()
    pose_encoder.eval()
    feature_network.eval()
    if use_policy:
        encoder.eval()
        actor_critic.eval()

    # =============== Create rollouts  ======================
    rollouts = RolloutStorageReconstruction(
        num_steps, N, (nclusters,), odometer_shape, nRef,
    )

    rollouts.to(device)

    if visualize_policy:
        tbwriter = TensorboardWriter(log_dir=vis_save_dir)

    per_episode_statistics = []
    # =============== Gather evaluation info  ======================
    episode_environment_statistics = []
    if multi_step:
        episode_losses = {interval: [] for interval in interval_steps}
        episode_preds = {interval: [] for interval in interval_steps}
        if visualize_policy:
            episode_ref_ims = {interval: [] for interval in interval_steps}
        episode_gts = {interval: [] for interval in interval_steps}
        episode_masks = {interval: [] for interval in interval_steps}
        episode_area_all = {interval: [] for interval in interval_steps}
    else:
        episode_losses = []
        episode_preds = []
        if visualize_policy:
            episode_ref_ims = []
        episode_gts = []
        episode_masks = []
        episode_area_all = []

    episode_rgb_all = []
    episode_depth_all = []
    episode_fine_occ_all = []
    episode_coarse_occ_all = []
    episode_topdown_map_all = []

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
        else:
            obs_lm = None
            obs_sm = None
        return obs_im, obs_sm, obs_lm

    num_eval_batches = (num_eval_episodes // N) + 1
    # =============== Evaluate over predefined number of episodes  ======================
    times_per_episode = []
    for neval in range(num_eval_batches):
        ep_start_time = time.time()
        # =================== Start a new episode ====================
        obs = envs.reset()
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)
        obs_odometer = process_odometer(obs["delta"])
        # Convert mm to m for AVD
        if "avd" in env_name:
            obs_odometer[:, :2] /= 1000.0
        obs_collns = obs["collisions"]  # (N, 1)

        # This remains constant throughout the episode
        tgt_poses = process_odometer(flatten_two(obs["pose_regress"]))[:, :3]
        tgt_poses = unflatten_two(tgt_poses, N, nRef)  # (N, nRef, 3)
        tgt_masks = obs["valid_masks"].unsqueeze(2)  # (N, nRef, 1)
        tgt_ims = process_image(flatten_two(obs["pose_refs"]))  # (N * nRef, ...)
        # Convert mm to m for AVD
        if "avd" in env_name:
            tgt_poses[:, :, :2] /= 1000.0

        if actor_type == "frontier":
            delta_ego = torch.zeros((N, 3)).to(device)
            frontier_agent.reset()

        if use_policy:
            recurrent_hidden_states = torch.zeros(N, feat_shape_sim[0]).to(device)
            masks = torch.zeros(N, 1).to(device)

        with torch.no_grad():
            obs_feat = feature_network(obs_im)
            tgt_feat = feature_network(tgt_ims)
            # Compute similarity scores with all other clusters
            obs_feat = torch.matmul(obs_feat, cluster_centroids.t())  # (N, nclusters)
            tgt_feat = torch.matmul(
                tgt_feat, cluster_centroids.t()
            )  # (N*nRef, nclusters)
        tgt_feat = unflatten_two(tgt_feat, N, nRef)

        # Initialize the memory of rollouts
        rollouts.reset()
        rollouts.obs_feats[0].copy_(obs_feat)
        rollouts.obs_odometer[0].copy_(obs_odometer)
        rollouts.tgt_poses.copy_(tgt_poses)
        rollouts.tgt_feats.copy_(tgt_feat)
        rollouts.tgt_masks.copy_(tgt_masks)

        stepwise_area = [[] for pr in range(N)]

        # gather statistics for visualization
        if visualize_policy:
            per_proc_rgb = [[] for pr in range(N)]
            per_proc_depth = [[] for pr in range(N)]
            per_proc_fine_occ = [[] for pr in range(N)]
            per_proc_coarse_occ = [[] for pr in range(N)]
            per_proc_topdown_map = [[] for pr in range(N)]

        prev_action = torch.zeros(N, 1).long().to(device)
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
                    np.random.randint(0, envs.action_space.n, size=(N, 1))
                ).long()
            elif actor_type == "forward":
                action = torch.Tensor(np.ones((N, 1)) * forward_action_id)
                action = action.long()
            elif actor_type == "forward-plus":
                action = torch.Tensor(np.ones((N, 1)) * forward_action_id)
                collision_mask = prev_collision > 0
                action[collision_mask] = turn_action_id
                action = action.long()
            elif actor_type == "frontier":
                # This assumes that N = 1
                occ_map = obs["highres_coarse_occupancy"][0].cpu().numpy()
                occ_map = occ_map.transpose(1, 2, 0)
                occ_map = np.ascontiguousarray(occ_map)
                occ_map = occ_map.astype(np.uint8)
                action = frontier_agent.act(
                    occ_map, delta_ego[0].cpu().numpy(), prev_collision[0].item()
                )
                action = torch.Tensor([[action]]).long()

            if (multi_step and ((step + 1) in interval_steps)) or (
                (not multi_step) and (step == num_steps - 1)
            ):
                L = step + 1
                obs_feats_curr = rollouts.obs_feats[:L]  # (L, N, feat_dim)
                obs_odometer_curr = rollouts.obs_odometer[
                    :L, :, :3
                ]  # (L, N, 3) --- (y, x, phi)
                # Convert odometer readings tgt_pose's frame of reference
                obs_odometer_exp = unsq_exp(
                    obs_odometer_curr, nRef, dim=2
                )  # (L, N, nRef, 3)
                obs_odometer_exp = obs_odometer_exp.view(-1, 3)  # (L*N*nRef, 3)
                tgt_poses_exp = unsq_exp(tgt_poses, L, dim=0)  # (L, N, nRef, 3)
                tgt_poses_exp = tgt_poses_exp.view(-1, 3)  # (L*N*nRef, 3)
                obs_relpose = subtract_pose(
                    obs_odometer_exp, tgt_poses_exp
                )  # (L*N*nRef, 3) --- (x, y, phi)

                obs_relpose_enc = pose_encoder(obs_relpose)  # (L*N*nRef, 16)
                obs_relpose_enc = obs_relpose_enc.view(
                    L, N, nRef, 16
                )  # (L, N, nRef, 16)
                tgt_relpose_enc = torch.zeros(1, *obs_relpose_enc.shape[1:]).to(
                    device
                )  # (1, N, nRef, 16)

                obs_feats_exp = unsq_exp(
                    obs_feats_curr, nRef, dim=2
                )  # (L, N, nRef, feat_dim)
                obs_feats_exp = obs_feats_exp.view(L, N * nRef, -1)
                obs_relpose_enc = obs_relpose_enc.view(L, N * nRef, -1)
                tgt_relpose_enc = tgt_relpose_enc.view(1, N * nRef, -1)

                rec_inputs = {
                    "history_image_features": obs_feats_exp,
                    "history_pose_features": obs_relpose_enc,
                    "target_pose_features": tgt_relpose_enc,
                }

                with torch.no_grad():
                    pred_logits = decoder(rec_inputs)  # (1, N*nRef, nclasses)
                    pred_logits = pred_logits.squeeze(0)
                    pred_outputs = unflatten_two(
                        pred_logits, N, nRef
                    )  # (N, nRef, nclasses)
                    loss = rec_loss_fn(
                        pred_logits,
                        flatten_two(tgt_feat),
                        cluster_centroids,
                        reduction="none",
                    ).sum(
                        dim=1
                    )  # (N*nRef, )

                if multi_step:
                    episode_losses[step + 1].append(loss.cpu())
                    episode_preds[step + 1].append(
                        pred_outputs.cpu().numpy()
                    )  # (N, nRef, nclasses)
                    episode_masks[step + 1].append(
                        tgt_masks.cpu().numpy()
                    )  # (N, nRef, 1)
                    if visualize_policy:
                        episode_ref_ims[step + 1].append(
                            unflatten_two(tgt_ims, N, nRef).cpu().numpy()
                        )  # (N, nRef, C, H, W)
                    episode_gts[step + 1].append(
                        tgt_feat.cpu().numpy()
                    )  # (N, nRef, feat_dim)
                else:
                    episode_losses.append(loss.cpu())
                    episode_preds.append(pred_outputs.cpu().numpy())  # (N, nRef, 1)
                    episode_masks.append(tgt_masks.cpu().numpy())  # (N, nRef, 1)
                    if visualize_policy:
                        episode_ref_ims.append(
                            unflatten_two(tgt_ims, N, nRef).cpu().numpy()
                        )  # (N, nRef, C, H, W)
                    episode_gts.append(tgt_feat.cpu().numpy())  # (N, nRef, 2048)

            obs, reward, done, infos = envs.step(action)

            if visualize_policy:
                for pr in range(N):
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
                for pr in range(N):
                    episode_environment_statistics.append(
                        infos[pr]["environment_statistics"]
                    )

            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)
            obs_odometer_ = process_odometer(obs["delta"])
            # Convert mm to m for AVD
            if "avd" in env_name:
                obs_odometer_[:, :2] /= 1000.0
            obs_collns = obs["collisions"]  # (N, 1)

            # Always set masks to 1 (does not matter for now)
            masks = torch.FloatTensor([[1.0] for _ in range(N)]).to(device)
            # Accumulate odometer readings to give relative pose from the starting point
            obs_odometer = obs_odometer * masks + obs_odometer_
            if actor_type == "frontier":
                delta_ego = compute_egocentric_coors(
                    obs_odometer_, rollouts.obs_odometer[step], occ_map_scale,
                )  # (N, 3) --- (dx_ego, dy_ego, dt_ego)

            # Update metrics
            for pr in range(N):
                stepwise_area[pr].append(infos[pr]["seen_area"])

            # This must not reach done = True
            # assert(done[0] == False)
            with torch.no_grad():
                obs_feat = feature_network(obs_im)
                # Compute similarity scores with all other clusters
                obs_feat = torch.matmul(
                    obs_feat, cluster_centroids.t()
                )  # (N, nclusters)

            # Update rollouts
            rollouts.insert(obs_feat, obs_odometer)
            # Update prev values
            prev_collision = obs_collns
            prev_action = action
        # endfor

        if multi_step:
            for interval in interval_steps:
                for pr in range(N):
                    # Area covered
                    episode_area_all[interval].append(stepwise_area[pr][interval - 1])
                if interval == interval_steps[-1]:
                    if "episode_id" in episode_environment_statistics[-pr - 1]:
                        logging.info(
                            "Episode id: {}, Area covered: {:.1f}".format(
                                episode_environment_statistics[-pr - 1]["episode_id"],
                                episode_area_all[interval][-pr - 1],
                            )
                        )
        else:
            for pr in range(N):
                episode_area_all.append(stepwise_area[pr][-1])
                if "episode_id" in episode_environment_statistics[-pr - 1]:
                    logging.info(
                        "Episode id: {}, Area covered: {:.1f}".format(
                            episode_environment_statistics[-pr - 1]["episode_id"],
                            episode_area_all[-pr - 1],
                        )
                    )

        # Write the episode data to tensorboard (only for the first episode of first 8 batches to save time)
        if neval < 8 and visualize_policy:
            WIDTH = 200
            HEIGHT = 200
            proc_fn = lambda x: np.ascontiguousarray(
                np.flip(np.concatenate(x, axis=1), axis=2)
            )
            num_to_save_per_batch = 1
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

        # End of episode
        times_per_episode.append(time.time() - ep_start_time)
        mins_per_episode = np.mean(times_per_episode).item() / 60.0
        eta_completion = mins_per_episode * (num_eval_batches - neval)
        num_episodes_done = (neval + 1) * N
        total_episodes = num_eval_batches * N
        logging.info(
            f"============> Num episodes done: {num_episodes_done}/{total_episodes}, Avg time per episode: {mins_per_episode:.3f} mins, ETA completion: {eta_completion:.3f} mins"
        )

    envs.close()

    if multi_step:
        metrics = {interval: {} for interval in interval_steps}
        cluster_centroids_np_T = cluster_centroids.cpu().numpy().T

        # Fill in per-episode statistics
        total_episodes = len(np.concatenate(episode_gts[interval_steps[0]], axis=0))
        per_episode_statistics = [[] for _ in range(total_episodes)]

        for interval in interval_steps:

            episode_losses[interval] = torch.cat(
                episode_losses[interval], dim=0
            ).numpy()
            episode_masks[interval] = np.concatenate(
                episode_masks[interval], axis=0
            )  # (N, nRef, 1)
            episode_preds[interval] = np.concatenate(
                episode_preds[interval], axis=0
            )  # (N, nRef, 2048 / nclasses)
            if visualize_policy:
                episode_ref_ims[interval] = np.concatenate(
                    episode_ref_ims[interval], axis=0
                )  # (N, nRef, C, H, W)
                epshape = episode_ref_ims[interval].shape
                temp_image = unprocess_image(
                    episode_ref_ims[interval].reshape(-1, *epshape[2:])
                )  # (N*nRef, H, W, C)
                episode_ref_ims[interval] = temp_image.reshape(
                    *epshape[:2], *temp_image.shape[1:]
                )  # (N, nRef, H, W, C)
            episode_gts[interval] = np.concatenate(
                episode_gts[interval], axis=0
            )  # (N, nRef, 2048)
            episode_area_all[interval] = np.array(episode_area_all[interval])

            # Retrieve the top_K predicted clusters
            topk_matches = np.argpartition(episode_preds[interval], -5, axis=2)[
                :, :, -5:
            ]  # (N, nRef, 5)
            pred_feat_score = torch.Tensor(
                episode_preds[interval]
            )  # (N, nRef, nclusters)

            gt_feat_score = torch.Tensor(episode_gts[interval])  # (N, nRef, nclusters)
            episode_losses[interval] = episode_losses[interval].reshape(
                -1, nRef
            )  # (N, nRef)
            episode_masks[interval] = episode_masks[interval][:, :, 0]  # (N, nRef)

            logging.info(
                "======= Evaluating at {} steps for {} episodes ========".format(
                    interval, num_eval_batches * N
                )
            )
            metrics[interval]["rec_loss"] = masked_mean(
                episode_losses[interval], episode_masks[interval]
            )
            metrics[interval]["area_covered"] = episode_area_all[interval].mean().item()
            gt_masks = flatten_two(torch.Tensor(episode_masks[interval]))
            # pred_feat_score_flattened = flatten_two(pred_feat_score)[gt_masks == 1.0]
            # gt_feat_score_flattened = flatten_two(gt_feat_score)[gt_masks == 1.0] # (N*nRef, nclusters)
            pred_feat_score_flattened = flatten_two(pred_feat_score)
            gt_feat_score_flattened = flatten_two(gt_feat_score)  # (N*nRef, nclusters)
            rec_metrics, per_episode_rec_metrics = compute_reconstruction_metrics(
                pred_feat_score_flattened, gt_feat_score_flattened, gt_masks,
            )

            for k, v in per_episode_rec_metrics.items():
                per_episode_rec_metrics[k] = masked_mean(
                    v.reshape(-1, nRef), episode_masks[interval], axis=1
                )  # (N, )

            # Fill in per-episode statistics
            for nep in range(total_episodes):
                eploss = (
                    episode_losses[interval][nep] * episode_masks[interval][nep]
                ).sum() / (episode_masks[interval][nep].sum() + 1e-8)
                eploss = eploss.item()

                curr_episode_metrics = {
                    "time_step": interval,
                    "rec_loss": eploss,
                    "environment_statistics": episode_environment_statistics[nep],
                    "area_covered": episode_area_all[interval][nep].item(),
                }
                for k, v in per_episode_rec_metrics.items():
                    curr_episode_metrics[k] = v[nep].item()
                per_episode_statistics[nep].append(curr_episode_metrics)

            for k, v in rec_metrics.items():
                metrics[interval][k] = v

            for k, v in metrics[interval].items():
                logging.info("{}: {:.3f}".format(k, v))

            # ======== Generate visualizations  ===========
            if interval == interval_steps[-1] and visualize_policy:
                # Sample the first reference from each process to visualize
                vis_gt_images = episode_ref_ims[interval][:, 0]  # (N, H, W, C)
                vis_retrieved_clusters = []
                vis_pred_logits = episode_preds[interval][:, 0]  # (N, nclasses)
                vis_pred_scores = F.softmax(
                    torch.Tensor(vis_pred_logits), dim=1
                ).numpy()  # (N, nclasses)
                vis_gt_feats = episode_gts[interval][:, 0]  # (N, feat_dim)
                # Compute similarity between GT features and all clusters
                # vis_gt_sim = np.matmul(vis_gt_feats, cluster_centroids_np_T) # (N, nclasses)
                vis_gt_sim = vis_gt_feats  # (N, nclasses)
                # Top 5 sim scores for GT features
                vis_gt_topk_idxes = np.argpartition(vis_gt_sim, -5, axis=1)[
                    :, -5:
                ]  # (N, 5)

                # Sample clusters predicted by the network
                for j in range(min(vis_gt_sim.shape[0], 12)):
                    vis_gt_image_j = cv2.resize(vis_gt_images[j], (300, 300))
                    # Add zero padding on top for text
                    vis_gt_image_j = np.pad(
                        vis_gt_image_j, ((100, 0), (0, 0), (0, 0)), mode="constant"
                    )
                    gt_sim_text = ",".join(
                        "{:.2f}".format(vis_gt_sim[j, v.item()])
                        for v in vis_gt_topk_idxes[j]
                    )
                    vis_gt_image_j = cv2.putText(
                        vis_gt_image_j,
                        "Best GT sim",
                        (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        thickness=2,
                    )
                    vis_gt_image_j = cv2.putText(
                        vis_gt_image_j,
                        gt_sim_text,
                        (5, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        thickness=2,
                    )

                    proc_retrieved_clusters = [vis_gt_image_j]
                    for k in topk_matches[j][0]:
                        if clusters2images[k].shape[0] == 0:
                            continue
                        # Pick a random set of 9 cluster images
                        random_idxes = np.random.randint(
                            0, clusters2images[k].shape[0], (9,)
                        )
                        ret_images = clusters2images[k][random_idxes]  # (9, H, W, C)
                        H, W = ret_images.shape[1:3]
                        ret_images = ret_images.reshape(
                            3, 3, *ret_images.shape[1:]
                        )  # (3, 3, H, W, C)
                        ret_images = np.ascontiguousarray(
                            ret_images.transpose(0, 2, 1, 3, 4)
                        )
                        ret_images = ret_images.reshape(3 * H, 3 * W, -1)
                        ret_images = draw_border(ret_images[np.newaxis, ...])[0]
                        ret_images = cv2.resize(ret_images, (300, 300))
                        # The similarity score of GT features with the retrieved clusters
                        gt_sim_score = vis_gt_sim[j][k].item()
                        pred_sim_score = vis_pred_scores[j][k].item()
                        gt_text = f"GT sim: {gt_sim_score:.2f}"
                        pred_text = f"Pred prob: {pred_sim_score:.2f}"
                        # Add zero padding on top for text
                        ret_images = np.pad(
                            ret_images, ((100, 0), (0, 0), (0, 0)), mode="constant"
                        )
                        # Add the text
                        ret_images = cv2.putText(
                            ret_images,
                            gt_text,
                            (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            thickness=2,
                        )
                        ret_images = cv2.putText(
                            ret_images,
                            pred_text,
                            (5, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            thickness=2,
                        )
                        proc_retrieved_clusters.append(ret_images)

                    proc_retrieved_clusters = np.concatenate(
                        proc_retrieved_clusters, axis=1
                    )
                    proc_retrieved_clusters = (
                        proc_retrieved_clusters.astype(np.float32) / 255.0
                    )
                    proc_retrieved_clusters = torch.Tensor(
                        proc_retrieved_clusters
                    ).permute(2, 0, 1)
                    proc_retrieved_clusters = proc_retrieved_clusters.contiguous()
                    vis_retrieved_clusters.append(proc_retrieved_clusters)

                vis_retrieved_clusters = torch.stack(vis_retrieved_clusters, axis=0)
                tbwriter.add_images(
                    f"Reconstructed images @ interval : {interval:04d}",
                    vis_retrieved_clusters,
                    0,
                )

    else:
        metrics = {}
        episode_losses = torch.cat(episode_losses, dim=0).numpy()
        episode_preds = np.concatenate(episode_preds, axis=0)  # (N, nRef, feat_dim)
        episode_masks = np.concatenate(episode_masks, axis=0)  # (N, nRef, 1)
        if visualize_policy:
            episode_ref_ims = np.concatenate(
                episode_ref_ims, axis=0
            )  # (N, nRef, C, H, W)
            temp_image = unprocess_image(
                episode_ref_ims.reshape(-1, *episode_ref_ims.shape[2:])
            )  # (N*nRef, H, W, C)
            episode_ref_ims = temp_image.reshape(
                *episode_ref_ims.shape[:2], *temp_image.shape[1:]
            )  # (N, nRef, H, W, C)
        episode_gts = np.concatenate(episode_gts, axis=0)  # (N, nRef, feat_dim)
        cluster_centroids_np_T = cluster_centroids.cpu().numpy().T
        episode_area_all = np.array(episode_area_all)

        # Retrieve the top_K predicted clusters
        topk_matches = np.argpartition(episode_preds, -5, axis=2)[
            :, :, -5:
        ]  # (N, nRef, 5)
        pred_feat_score = torch.Tensor(episode_preds)  # (N, nRef, nclusters)
        gt_feat_score = torch.Tensor(episode_gts)  # (N, nRef, nclusters)

        if visualize_policy:
            # ======== Generate visualizations  ===========
            # Sample the first reference from each process to visualize
            vis_gt_images = episode_ref_ims[:, 0]  # (N, H, W, C)
            vis_retrieved_clusters = []
            vis_pred_logits = episode_preds[:, 0]  # (N, nclasses)
            vis_pred_scores = F.softmax(
                torch.Tensor(vis_pred_logits), dim=1
            ).numpy()  # (N, nclasses)
            vis_gt_feats = episode_gts[:, 0]  # (N, feat_dim)
            # Compute similarity between GT features and all clusters
            # vis_gt_sim = np.matmul(vis_gt_feats, cluster_centroids_np_T) # (N, nclasses)
            vis_gt_sim = vis_gt_feats  # (N, nclasses)
            # Top 5 sim scores for GT features
            vis_gt_topk_idxes = np.argpartition(vis_gt_sim, -5, axis=1)[
                :, -5:
            ]  # (N, 5)

            # Sample clusters predicted by the network
            for j in range(min(vis_gt_sim.shape[0], 12)):
                vis_gt_image_j = cv2.resize(vis_gt_images[j], (300, 300))
                # Add zero padding on top for text
                vis_gt_image_j = np.pad(
                    vis_gt_image_j, ((100, 0), (0, 0), (0, 0)), mode="constant"
                )
                gt_sim_text = ",".join(
                    "{:.2f}".format(vis_gt_sim[j, v.item()])
                    for v in vis_gt_topk_idxes[j]
                )
                vis_gt_image_j = cv2.putText(
                    vis_gt_image_j,
                    "Best GT sim",
                    (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                )
                vis_gt_image_j = cv2.putText(
                    vis_gt_image_j,
                    gt_sim_text,
                    (5, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    thickness=2,
                )

                proc_retrieved_clusters = [vis_gt_image_j]
                for k in topk_matches[j][0]:
                    if clusters2images[k].shape[0] == 0:
                        continue
                    # Pick a random set of 9 cluster images
                    random_idxes = np.random.randint(
                        0, clusters2images[k].shape[0], (9,)
                    )
                    ret_images = clusters2images[k][random_idxes]  # (9, H, W, C)
                    H, W = ret_images.shape[1:3]
                    ret_images = ret_images.reshape(
                        3, 3, *ret_images.shape[1:]
                    )  # (3, 3, H, W, C)
                    ret_images = np.ascontiguousarray(
                        ret_images.transpose(0, 2, 1, 3, 4)
                    )
                    ret_images = ret_images.reshape(3 * H, 3 * W, -1)
                    ret_images = draw_border(ret_images[np.newaxis, ...])[0]
                    ret_images = cv2.resize(ret_images, (300, 300))
                    # The similarity score of GT features with the retrieved clusters
                    gt_sim_score = vis_gt_sim[j][k].item()
                    pred_sim_score = vis_pred_scores[j][k].item()
                    gt_text = f"GT sim: {gt_sim_score:.2f}"
                    pred_text = f"Pred prob: {pred_sim_score:.2f}"
                    # Add zero padding on top for text
                    ret_images = np.pad(
                        ret_images, ((100, 0), (0, 0), (0, 0)), mode="constant"
                    )
                    # Add the text
                    ret_images = cv2.putText(
                        ret_images,
                        gt_text,
                        (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        thickness=2,
                    )
                    ret_images = cv2.putText(
                        ret_images,
                        pred_text,
                        (5, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        thickness=2,
                    )
                    proc_retrieved_clusters.append(ret_images)

                proc_retrieved_clusters = np.concatenate(
                    proc_retrieved_clusters, axis=1
                )
                proc_retrieved_clusters = (
                    proc_retrieved_clusters.astype(np.float32) / 255.0
                )
                proc_retrieved_clusters = torch.Tensor(proc_retrieved_clusters).permute(
                    2, 0, 1
                )
                proc_retrieved_clusters = proc_retrieved_clusters.contiguous()
                vis_retrieved_clusters.append(proc_retrieved_clusters)

            vis_retrieved_clusters = torch.stack(vis_retrieved_clusters, axis=0)
            tbwriter.add_images("Reconstructed images", vis_retrieved_clusters, 0)

        total_episodes = episode_gts.shape[0]
        episode_losses = episode_losses.reshape(total_episodes, nRef)
        episode_masks = episode_masks[:, :, 0]  # (N, nRef)
        # =================== Update metrics =====================
        logging.info(
            "======= Evaluating at {} steps for {} episodes ========".format(
                num_steps, num_eval_batches * N
            )
        )
        metrics["rec_loss"] = masked_mean(episode_losses, episode_masks)
        metrics["area_covered"] = np.mean(episode_area_all).item()

        gt_masks = flatten_two(torch.Tensor(episode_masks))
        # pred_feat_score_flattened = flatten_two(pred_feat_score)[gt_masks == 1.0]
        # gt_feat_score_flattened = flatten_two(gt_feat_score)[gt_masks == 1.0]
        pred_feat_score_flattened = flatten_two(pred_feat_score)
        gt_feat_score_flattened = flatten_two(gt_feat_score)
        rec_metrics, per_episode_rec_metrics = compute_reconstruction_metrics(
            pred_feat_score_flattened, gt_feat_score_flattened, gt_masks,
        )
        for k, v in per_episode_rec_metrics.items():
            per_episode_rec_metrics[k] = masked_mean(
                v.reshape(-1, nRef), episode_masks, axis=1
            )  # (N, )

        # Fill in per-episode statistics
        per_episode_statistics = [
            [] for _ in range(total_episodes)
        ]  # Each episode can have results over multiple intervals
        for nep in range(total_episodes):
            eploss = masked_mean(episode_losses[nep], episode_masks[nep]).item()
            curr_episode_metrics = {
                "time_step": num_steps,
                "rec_loss": eploss,
                "environment_statistics": episode_environment_statistics[nep],
                "area_covered": episode_area_all[nep],
            }
            for k, v in per_episode_rec_metrics.items():
                curr_episode_metrics[k] = v[nep].item()
            per_episode_statistics.append(curr_episode_metrics)

        for k, v in rec_metrics.items():
            metrics[k] = v

        for k, v in metrics.items():
            logging.info("{}: {:.3f}".format(k, v))

    if visualize_policy:
        tbwriter.close()

    return metrics, per_episode_statistics
