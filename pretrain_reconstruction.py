#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import h5py
import torch
import logging
import numpy as np
import torch.nn as nn

from exploring_exploration.arguments import get_args
from exploring_exploration.envs import (
    make_vec_envs_avd,
    make_vec_envs_habitat,
)
from exploring_exploration.models.reconstruction import (
    FeatureReconstructionModule,
    FeatureNetwork,
    PoseEncoder,
)
from exploring_exploration.utils.common import (
    process_image,
    flatten_two,
    unflatten_two,
)
from exploring_exploration.utils.reconstruction import rec_loss_fn_classify
from exploring_exploration.algo import SupervisedReconstruction
from exploring_exploration.utils.storage import RolloutStorageReconstruction
from exploring_exploration.utils.geometry import process_odometer
from exploring_exploration.utils.reconstruction_eval import (
    evaluate_reconstruction_oracle,
)
from einops import rearrange
from tensorboardX import SummaryWriter

args = get_args()

num_updates = (args.num_episodes // args.num_processes) + 1

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    pass

eval_log_dir = os.path.join(args.log_dir, "eval_monitor")

try:
    os.makedirs(eval_log_dir)
except OSError:
    pass


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    ndevices = torch.cuda.device_count()
    # Setup loggers
    tbwriter = SummaryWriter(log_dir=args.log_dir)
    logging.basicConfig(filename=f"{args.log_dir}/train_log.txt", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)
    if "habitat" in args.env_name:
        devices = [int(dev) for dev in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        # Devices need to be indexed between 0 to N-1
        devices = [dev for dev in range(len(devices))]
        envs = make_vec_envs_habitat(
            args.habitat_config_file, device, devices, seed=args.seed
        )
    else:
        train_log_dir = os.path.join(args.log_dir, "train_monitor")
        try:
            os.makedirs(train_log_dir)
        except OSError:
            pass
        envs = make_vec_envs_avd(
            args.env_name,
            args.seed,
            args.num_processes,
            train_log_dir,
            device,
            True,
            num_frame_stack=1,
            split="train",
            nRef=args.num_pose_refs,
            ref_dist_thresh=args.ref_dist_thresh,
        )

    args.feat_shape_sim = (512,)
    args.obs_shape = envs.observation_space.spaces["im"].shape

    # =================== Load concept clusters =================
    clusters_h5 = h5py.File(args.clusters_path, "r")
    cluster_centroids = torch.Tensor(np.array(clusters_h5["cluster_centroids"])).to(
        device
    )
    args.nclusters = cluster_centroids.shape[0]
    clusters2images = {}
    for i in range(args.nclusters):
            clusters_h5[f"cluster_{i}/images"]
        )  # (K, C, H, W) torch Tensor
        cluster_images = rearrange(cluster_images, "k c h w -> k h w c")
        cluster_images = (cluster_images * 255.0).astype(np.uint8)
        clusters2images[i] = cluster_images  # (K, H, W, C)
    clusters_h5.close()

    # =================== Create models ====================
    decoder = FeatureReconstructionModule(
        args.nclusters, args.nclusters, nlayers=args.n_transformer_layers,
    )
    feature_network = FeatureNetwork()
    feature_network = nn.DataParallel(feature_network, dim=0)
    pose_encoder = PoseEncoder()
    if args.use_multi_gpu:
        decoder = nn.DataParallel(decoder, dim=1)
        pose_encoder = nn.DataParallel(pose_encoder, dim=0)

    # =================== Load models ====================
    save_path = os.path.join(args.save_dir, "checkpoints")
    checkpoint_path = os.path.join(save_path, "ckpt.latest.pth")
    if os.path.isfile(checkpoint_path):
        logging.info("Resuming from old model!")
        decoder_state, pose_encoder_state, j_start = torch.load(checkpoint_path)
        decoder.load_state_dict(decoder_state)
        pose_encoder.load_state_dict(pose_encoder_state)
    else:
        j_start = -1
    decoder.to(device)
    pose_encoder.to(device)
    feature_network.to(device)
    decoder.eval()
    pose_encoder.eval()
    feature_network.eval()  # Feature network is frozen

    # =================== Define decoder training algorithm ====================
    algo_config = {}
    algo_config["lr"] = args.lr
    algo_config["eps"] = args.eps
    algo_config["rec_loss_fn"] = rec_loss_fn_classify
    algo_config["rec_loss_fn_J"] = args.rec_loss_fn_J
    algo_config["max_grad_norm"] = args.max_grad_norm
    algo_config["cluster_centroids"] = cluster_centroids
    algo_config["prediction_interval"] = 20 if "avd" in args.env_name else 100

    algo_config["decoder"] = decoder
    algo_config["pose_encoder"] = pose_encoder

    reconstruction_algo = SupervisedReconstruction(algo_config)

    # =================== Define rollouts ====================
    odometer_shape = (4,)
    rollouts = RolloutStorageReconstruction(
        args.num_rl_steps,
        args.num_processes,
        (args.nclusters,),
        odometer_shape,
        args.num_pose_refs,
    )
    rollouts.to(device)

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        return obs_im

    start = time.time()
    NPROC = args.num_processes
    NREF = args.num_pose_refs
    for j in range(j_start + 1, num_updates):
        # =================== Start a new episode ====================
        obs = envs.reset()
        # Processing environment inputs
        obs_im = get_obs(obs)  # (num_processes, 3, 84, 84)
        obs_odometer = process_odometer(obs["delta"])  # (num_processes, 4)
        # Convert mm to m for AVD
        if "avd" in args.env_name:
            obs_odometer[:, :2] /= 1000.0
        # ============== Target poses and corresponding images ================
        # NOTE - these are constant throughout the episode.
        # (num_processes * num_pose_refs, 3) --- (y, x, t)
        tgt_poses = process_odometer(flatten_two(obs["pose_regress"]))[:, :3]
        tgt_poses = unflatten_two(tgt_poses, NPROC, NREF)  # (N, nRef, 3)
        tgt_masks = obs["valid_masks"].unsqueeze(2)  # (N, nRef, 1)
        # Convert mm to m for AVD
        if "avd" in args.env_name:
            tgt_poses[:, :, :2] /= 1000.0
        tgt_ims = process_image(flatten_two(obs["pose_refs"]))  # (N*nRef, C, H, W)
        # Initialize the memory of rollouts
        rollouts.reset()
        with torch.no_grad():
            obs_feat = feature_network(obs_im)  # (N, 2048)
            tgt_feat = feature_network(tgt_ims)  # (N*nRef, 2048)
            # Compute similarity scores with all other clusters
            obs_feat = torch.matmul(obs_feat, cluster_centroids.t())  # (N, nclusters)
            tgt_feat = torch.matmul(
                tgt_feat, cluster_centroids.t()
            )  # (N*nRef, nclusters)
        tgt_feat = unflatten_two(tgt_feat, NPROC, NREF)  # (N, nRef, nclusters)
        rollouts.obs_feats[0].copy_(obs_feat)
        rollouts.obs_odometer[0].copy_(obs_odometer)
        rollouts.tgt_poses.copy_(tgt_poses)
        rollouts.tgt_feats.copy_(tgt_feat)
        rollouts.tgt_masks.copy_(tgt_masks)
        # =============== Update over a full batch of episodes ================
        # num_steps must be total number of steps in each episode
        for step in range(args.num_steps):
            pstep = rollouts.step
            action = obs["oracle_action"]
            # Act, get reward and next obs
            obs, reward, done, infos = envs.step(action)
            # Processing environment inputs
            obs_im = get_obs(obs)  # (num_processes, 3, 84, 84)
            obs_odometer = process_odometer(obs["delta"])  # (num_processes, 4)
            if "avd" in args.env_name:
                obs_odometer[:, :2] /= 1000.0
            with torch.no_grad():
                obs_feat = feature_network(obs_im)
                # Compute similarity scores with all other clusters
                obs_feat = torch.matmul(
                    obs_feat, cluster_centroids.t()
                )  # (N, nclusters)
            # Always set masks to 1 (since this loop happens within one episode)
            masks = torch.FloatTensor([[1.0] for _ in range(NPROC)]).to(device)
            # Accumulate odometer readings to give relative pose
            # from the starting point
            obs_odometer = rollouts.obs_odometer[pstep] * masks + obs_odometer
            # Update rollouts
            rollouts.insert(obs_feat, obs_odometer)
            if (step + 1) % args.num_rl_steps == 0:
                decoder.train()
                pose_encoder.train()
                # Update decoder
                losses = reconstruction_algo.update(rollouts)
                # Refresh rollouts
                rollouts.after_update()
                decoder.eval()
                pose_encoder.eval()

        # =================== Save model ====================
        if (j + 1) % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "checkpoints")
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            decoder_state = decoder.state_dict()
            pose_encoder_state = pose_encoder.state_dict()
            torch.save(
                [decoder_state, pose_encoder_state, j],
                os.path.join(save_path, "ckpt.latest.pth"),
            )
            if args.save_unique:
                torch.save(
                    [decoder_state, pose_encoder_state, j],
                    os.path.join(save_path, f"{save_path}/ckpt.{(j+1):07d}.pth"),
                )

        # =================== Logging data ====================
        total_num_steps = (j + 1 - j_start) * NPROC * args.num_steps
        if j % args.log_interval == 0:
            end = time.time()
            fps = int(total_num_steps / (end - start))
            logging.info(f"===> Updates {j}, #steps {total_num_steps}, FPS {fps}")
            train_metrics = losses
            for k, v in train_metrics.items():
                logging.info("{}: {:.3f}".format(k, v))
                tbwriter.add_scalar("train_metrics/{}".format(k), v, j)

        # =================== Evaluate models ====================
        if args.eval_interval is not None and (j + 1) % args.eval_interval == 0:
            if "habitat" in args.env_name:
                devices = [
                    int(dev) for dev in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                ]
                # Devices need to be indexed between 0 to N-1
                devices = [dev for dev in range(len(devices))]
                eval_envs = make_vec_envs_habitat(
                    args.eval_habitat_config_file, device, devices
                )
            else:
                eval_envs = make_vec_envs_avd(
                    args.env_name,
                    args.seed + 12,
                    12,
                    eval_log_dir,
                    device,
                    True,
                    split="val",
                    nRef=NREF,
                    ref_dist_thresh=args.ref_dist_thresh,
                    set_return_topdown_map=True,
                )

            num_eval_episodes = 16 if "habitat" in args.env_name else 30

            eval_config = {}
            eval_config["num_steps"] = args.num_steps
            eval_config["num_processes"] = 1 if "habitat" in args.env_name else 12
            eval_config["num_eval_episodes"] = num_eval_episodes
            eval_config["num_pose_refs"] = NREF
            eval_config["cluster_centroids"] = cluster_centroids
            eval_config["clusters2images"] = clusters2images
            eval_config["odometer_shape"] = odometer_shape
            eval_config["rec_loss_fn"] = rec_loss_fn_classify
            eval_config["rec_loss_fn_J"] = args.rec_loss_fn_J
            eval_config["vis_save_dir"] = os.path.join(
                args.save_dir, "policy_vis", "update_{:05d}".format(j + 1)
            )
            eval_config["env_name"] = args.env_name

            models = {}
            models["decoder"] = decoder
            models["pose_encoder"] = pose_encoder
            models["feature_network"] = feature_network

            val_metrics = evaluate_reconstruction_oracle(
                models, eval_envs, eval_config, device
            )

            for k, v in val_metrics.items():
                tbwriter.add_scalar("val_metrics/{}".format(k), v, j)

    tbwriter.close()


if __name__ == "__main__":
    main()
