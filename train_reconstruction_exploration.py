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
from exploring_exploration.utils.reconstruction import (
    rec_loss_fn_classify,
    compute_reconstruction_rewards,
)
from exploring_exploration.utils.storage import (
    RolloutStorageReconstruction,
    RolloutStoragePPO,
)
from exploring_exploration.algo import PPO
from exploring_exploration.models import RGBEncoder, MapRGBEncoder, Policy
from exploring_exploration.utils.geometry import process_odometer
from exploring_exploration.utils.reconstruction_eval import evaluate_reconstruction
from einops import rearrange
from tensorboardX import SummaryWriter

from collections import defaultdict, deque

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
        if len(devices) > 2:
            devices = devices[1:]
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
        )

    args.feat_shape_sim = (512,)
    args.obs_shape = envs.observation_space.spaces["im"].shape
    args.odometer_shape = (4,)  # (delta_y, delta_x, delta_head, delta_elev)

    # =================== Load clusters =================
    clusters_h5 = h5py.File(args.clusters_path, "r")
    cluster_centroids = torch.Tensor(np.array(clusters_h5["cluster_centroids"])).to(
        device
    )
    cluster_centroids_t = cluster_centroids.t()
    args.nclusters = cluster_centroids.shape[0]
    clusters2images = {}
    for i in range(args.nclusters):
        cluster_images = np.array(
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
    pose_encoder = PoseEncoder()
    if args.encoder_type == "rgb":
        encoder = RGBEncoder(fix_cnn=args.fix_cnn)
    elif args.encoder_type == "rgb+map":
        encoder = MapRGBEncoder(fix_cnn=args.fix_cnn)
    else:
        raise ValueError(f"encoder_type {args.encoder_type} not defined!")
    action_config = (
        {"nactions": envs.action_space.n, "embedding_size": args.action_embedding_size}
        if args.use_action_embedding
        else None
    )
    collision_config = (
        {"collision_dim": 2, "embedding_size": args.collision_embedding_size}
        if args.use_collision_embedding
        else None
    )
    actor_critic = Policy(
        envs.action_space,
        base_kwargs={
            "feat_dim": args.feat_shape_sim[0],
            "recurrent": True,
            "hidden_size": args.feat_shape_sim[0],
            "action_config": action_config,
            "collision_config": collision_config,
        },
    )

    # =================== Load models ====================
    decoder_state, pose_encoder_state = torch.load(args.load_path_rec)[:2]
    # Remove DataParallel related strings
    new_decoder_state, new_pose_encoder_state = {}, {}
    for k, v in decoder_state.items():
        new_decoder_state[k.replace("module.", "")] = v
    for k, v in pose_encoder_state.items():
        new_pose_encoder_state[k.replace("module.", "")] = v
    decoder.load_state_dict(new_decoder_state)
    pose_encoder.load_state_dict(new_pose_encoder_state)
    decoder = nn.DataParallel(decoder, dim=1)
    pose_encoder = nn.DataParallel(pose_encoder, dim=0)
    save_path = os.path.join(args.save_dir, "checkpoints")
    checkpoint_path = os.path.join(save_path, "ckpt.latest.pth")
    if os.path.isfile(checkpoint_path):
        logging.info("Resuming from old model!")
        loaded_states = torch.load(checkpoint_path)
        encoder_state, actor_critic_state, j_start = loaded_states
        encoder.load_state_dict(encoder_state)
        actor_critic.load_state_dict(actor_critic_state)
    elif args.pretrained_il_model != "":
        logging.info("Initializing with pre-trained model!")
        encoder_state, actor_critic_state, _ = torch.load(args.pretrained_il_model)
        actor_critic.load_state_dict(actor_critic_state)
        encoder.load_state_dict(encoder_state)
        j_start = -1
    else:
        j_start = -1
    encoder.to(device)
    actor_critic.to(device)
    decoder.to(device)
    feature_network.to(device)
    pose_encoder.to(device)
    encoder.eval()
    actor_critic.eval()
    # decoder, feature_network, pose_encoder are frozen during policy training
    decoder.eval()
    feature_network.eval()
    pose_encoder.eval()

    # =================== Define RL training algorithm ====================
    rl_algo_config = {}
    rl_algo_config["lr"] = args.lr
    rl_algo_config["eps"] = args.eps
    rl_algo_config["encoder_type"] = args.encoder_type
    rl_algo_config["max_grad_norm"] = args.max_grad_norm
    rl_algo_config["clip_param"] = args.clip_param
    rl_algo_config["ppo_epoch"] = args.ppo_epoch
    rl_algo_config["entropy_coef"] = args.entropy_coef
    rl_algo_config["num_mini_batch"] = args.num_mini_batch
    rl_algo_config["value_loss_coef"] = args.value_loss_coef
    rl_algo_config["use_clipped_value_loss"] = False
    rl_algo_config["nactions"] = envs.action_space.n

    rl_algo_config["encoder"] = encoder
    rl_algo_config["actor_critic"] = actor_critic
    rl_algo_config["use_action_embedding"] = args.use_action_embedding
    rl_algo_config["use_collision_embedding"] = args.use_collision_embedding

    rl_agent = PPO(rl_algo_config)

    # =================== Define stats buffer ====================
    train_metrics_tracker = defaultdict(lambda: deque(maxlen=10))

    # =================== Define rollouts ====================
    rollouts_recon = RolloutStorageReconstruction(
        args.num_steps,
        args.num_processes,
        (args.nclusters,),
        args.odometer_shape,
        args.num_pose_refs,
    )
    rollouts_policy = RolloutStoragePPO(
        args.num_rl_steps,
        args.num_processes,
        args.obs_shape,
        envs.action_space,
        args.feat_shape_sim[0],
        encoder_type=args.encoder_type,
    )
    rollouts_recon.to(device)
    rollouts_policy.to(device)

    def get_obs(obs):
        obs_im = process_image(obs["im"])
        if args.encoder_type == "rgb+map":
            obs_lm = process_image(obs["coarse_occupancy"])
            obs_sm = process_image(obs["fine_occupancy"])
        else:
            obs_lm = None
            obs_sm = None
        return obs_im, obs_sm, obs_lm

    start = time.time()
    NPROC = args.num_processes
    NREF = args.num_pose_refs
    for j in range(j_start + 1, num_updates):
        # =================== Start a new episode ====================
        obs = envs.reset()
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)  # (num_processes, 3, 84, 84)
        obs_odometer = process_odometer(obs["delta"])  # (num_processes, 4)
        # Convert mm to m for AVD
        if "avd" in args.env_name:
            obs_odometer[:, :2] /= 1000.0
        obs_collns = obs["collisions"].long()  # (num_processes, 1)
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
        # Initialize the memory of rollouts for reconstruction
        rollouts_recon.reset()
        with torch.no_grad():
            obs_feat = feature_network(obs_im)  # (N, 2048)
            tgt_feat = feature_network(tgt_ims)  # (N*nRef, 2048)
            # Compute similarity scores with all other clusters
            obs_feat = torch.matmul(obs_feat, cluster_centroids_t)  # (N, nclusters)
            tgt_feat = torch.matmul(
                tgt_feat, cluster_centroids_t
            )  # (N*nRef, nclusters)
        tgt_feat = unflatten_two(tgt_feat, NPROC, NREF)  # (N, nRef, nclusters)
        rollouts_recon.obs_feats[0].copy_(obs_feat)
        rollouts_recon.obs_odometer[0].copy_(obs_odometer)
        rollouts_recon.tgt_poses.copy_(tgt_poses)
        rollouts_recon.tgt_feats.copy_(tgt_feat)
        rollouts_recon.tgt_masks.copy_(tgt_masks)
        # Initialize the memory of rollouts for policy
        rollouts_policy.reset()
        rollouts_policy.obs_im[0].copy_(obs_im)
        if args.encoder_type == "rgb+map":
            rollouts_policy.obs_sm[0].copy_(obs_sm)
            rollouts_policy.obs_lm[0].copy_(obs_lm)
        rollouts_policy.collisions[0].copy_(obs_collns)
        # Episode statistics
        episode_expl_rewards = np.zeros((NPROC, 1))
        episode_collisions = np.zeros((NPROC, 1))
        episode_rec_rewards = np.zeros((NPROC, 1))
        episode_collisions += obs_collns.cpu().numpy()
        # Metrics
        osr_tracker = [0.0 for _ in range(NPROC)]
        objects_tracker = [0.0 for _ in range(NPROC)]
        area_tracker = [0.0 for _ in range(NPROC)]
        novelty_tracker = [0.0 for _ in range(NPROC)]
        smooth_coverage_tracker = [0.0 for _ in range(NPROC)]
        per_proc_area = [0.0 for _ in range(NPROC)]
        # Other states
        prev_action = torch.zeros(NPROC, 1).long().to(device)
        prev_collision = rollouts_policy.collisions[0]
        rec_reward_interval = args.rec_reward_interval
        prev_rec_rewards = torch.zeros(NPROC, 1)  # (N, 1)
        prev_rec_rewards = prev_rec_rewards.to(device)
        rec_rewards_at_t0 = None
        # ================= Update over a full batch of episodes =================
        # num_steps must be total number of steps in each episode
        for step in range(args.num_steps):
            pstep = rollouts_policy.step
            with torch.no_grad():
                encoder_inputs = [rollouts_policy.obs_im[pstep]]
                if args.encoder_type == "rgb+map":
                    encoder_inputs.append(rollouts_policy.obs_sm[pstep])
                    encoder_inputs.append(rollouts_policy.obs_lm[pstep])
                obs_feats = encoder(*encoder_inputs)
                policy_inputs = {"features": obs_feats}
                if args.use_action_embedding:
                    policy_inputs["actions"] = prev_action.long()
                if args.use_collision_embedding:
                    policy_inputs["collisions"] = prev_collision.long()

                policy_outputs = actor_critic.act(
                    policy_inputs,
                    rollouts_policy.recurrent_hidden_states[pstep],
                    rollouts_policy.masks[pstep],
                )
                (
                    value,
                    action,
                    action_log_probs,
                    recurrent_hidden_states,
                ) = policy_outputs

            # Act, get reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)  # (num_processes, 3, 84, 84)
            obs_odometer = process_odometer(obs["delta"])  # (num_processes, 4)
            if "avd" in args.env_name:
                obs_odometer[:, :2] /= 1000.0
            obs_collns = obs["collisions"]  # (N, 1)
            with torch.no_grad():
                obs_feat = feature_network(obs_im)
                # Compute similarity scores with all other clusters
                obs_feat = torch.matmul(obs_feat, cluster_centroids_t)  # (N, nclusters)

            # Always set masks to 1 (since this loop happens within one episode)
            masks = torch.FloatTensor([[1.0] for _ in range(NPROC)]).to(device)

            # Accumulate odometer readings to give relative pose from the starting point
            obs_odometer = rollouts_recon.obs_odometer[step] * masks + obs_odometer

            # Update rollouts_recon
            rollouts_recon.insert(obs_feat, obs_odometer)

            # Compute the exploration rewards
            reward_exploration = torch.zeros(NPROC, 1)  # (N, 1)
            for proc in range(NPROC):
                seen_area = float(infos[proc]["seen_area"])
                objects_visited = infos[proc].get("num_objects_visited", 0.0)
                oracle_success = float(infos[proc]["oracle_pose_success"])
                novelty_reward = infos[proc].get("count_based_reward", 0.0)
                smooth_coverage_reward = infos[proc].get("coverage_novelty_reward", 0.0)
                area_reward = seen_area - area_tracker[proc]
                objects_reward = objects_visited - objects_tracker[proc]
                landmarks_reward = oracle_success - osr_tracker[proc]
                collision_reward = -obs_collns[proc, 0].item()

                area_tracker[proc] = seen_area
                objects_tracker[proc] = objects_visited
                osr_tracker[proc] = oracle_success
                per_proc_area[proc] = seen_area
                novelty_tracker[proc] += novelty_reward
                smooth_coverage_tracker[proc] += smooth_coverage_reward

            # Compute reconstruction rewards
            if (step + 1) % rec_reward_interval == 0 or step == 0:
                rec_rewards = compute_reconstruction_rewards(
                    rollouts_recon.obs_feats[: (step + 1)],
                    rollouts_recon.obs_odometer[: (step + 1), :, :3],
                    rollouts_recon.tgt_feats,
                    rollouts_recon.tgt_poses,
                    cluster_centroids_t,
                    decoder,
                    pose_encoder,
                ).detach()  # (N, nRef)
                rec_rewards = rec_rewards * tgt_masks.squeeze(2)  # (N, nRef)
                rec_rewards = rec_rewards.sum(dim=1).unsqueeze(
                    1
                )  # / (tgt_masks.sum(dim=1) + 1e-8)
                final_rec_rewards = rec_rewards - prev_rec_rewards
                # Ignore the exploration reward at T=0 since it will be a huge spike
                if (("avd" in args.env_name) and (step != 0)) or (
                    ("habitat" in args.env_name) and (step > 20)
                ):
                    reward_exploration += (
                        final_rec_rewards.cpu() * args.rec_reward_scale
                    )
                    episode_rec_rewards += final_rec_rewards.cpu().numpy()
                prev_rec_rewards = rec_rewards

            overall_reward = (
                reward * (1 - args.reward_scale)
                + reward_exploration * args.reward_scale
            )

            # Update statistics
            episode_expl_rewards += reward_exploration.numpy() * args.reward_scale

            # Update rollouts_policy
            rollouts_policy.insert(
                obs_im,
                obs_sm,
                obs_lm,
                recurrent_hidden_states,
                action,
                action_log_probs,
                value,
                overall_reward,
                masks,
                obs_collns,
            )

            # Update prev values
            prev_collision = obs_collns
            prev_action = action
            episode_collisions += obs_collns.cpu().numpy()

            # Update RL policy
            if (step + 1) % args.num_rl_steps == 0:
                # Update value function for last step
                with torch.no_grad():
                    encoder_inputs = [rollouts_policy.obs_im[-1]]
                    if args.encoder_type == "rgb+map":
                        encoder_inputs.append(rollouts_policy.obs_sm[-1])
                        encoder_inputs.append(rollouts_policy.obs_lm[-1])
                    obs_feats = encoder(*encoder_inputs)
                    policy_inputs = {"features": obs_feats}
                    if args.use_action_embedding:
                        policy_inputs["actions"] = prev_action.long()
                    if args.use_collision_embedding:
                        policy_inputs["collisions"] = prev_collision.long()
                    next_value = actor_critic.get_value(
                        policy_inputs,
                        rollouts_policy.recurrent_hidden_states[-1],
                        rollouts_policy.masks[-1],
                    ).detach()
                # Compute returns
                rollouts_policy.compute_returns(
                    next_value, args.use_gae, args.gamma, args.tau
                )

                encoder.train()
                actor_critic.train()
                # Update model
                rl_losses = rl_agent.update(rollouts_policy)
                # Refresh rollouts_policy
                rollouts_policy.after_update()
                encoder.eval()
                actor_critic.eval()

        # =================== Save model ====================
        if (j + 1) % args.save_interval == 0 and args.save_dir != "":
            save_path = f"{args.save_dir}/checkpoints"
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            encoder_state = encoder.state_dict()
            actor_critic_state = actor_critic.state_dict()
            torch.save(
                [encoder_state, actor_critic_state, j], f"{save_path}/ckpt.latest.pth",
            )
            if args.save_unique:
                torch.save(
                    [encoder_state, actor_critic_state, j],
                    f"{save_path}/ckpt.{(j+1):07d}.pth",
                )

        # =================== Logging data ====================
        total_num_steps = (j + 1 - j_start) * NPROC * args.num_steps
        if j % args.log_interval == 0:
            end = time.time()
            fps = int(total_num_steps / (end - start))
            logging.info(f"===> Updates {j}, #steps {total_num_steps}, FPS {fps}")
            train_metrics = rl_losses
            train_metrics["exploration_rewards"] = (
                np.mean(episode_expl_rewards) * rec_reward_interval / args.num_steps
            )
            train_metrics["rec_rewards"] = (
                np.mean(episode_rec_rewards) * rec_reward_interval / args.num_steps
            )
            train_metrics["area_covered"] = np.mean(per_proc_area)
            train_metrics["objects_covered"] = np.mean(objects_tracker)
            train_metrics["landmarks_covered"] = np.mean(osr_tracker)
            train_metrics["collisions"] = np.mean(episode_collisions)
            train_metrics["novelty_rewards"] = np.mean(novelty_tracker)
            train_metrics["smooth_coverage_rewards"] = np.mean(smooth_coverage_tracker)

            # Update statistics
            for k, v in train_metrics.items():
                train_metrics_tracker[k].append(v)

            for k, v in train_metrics_tracker.items():
                logging.info(f"{k}: {np.mean(v).item():.3f}")
                tbwriter.add_scalar(f"train_metrics/{k}", np.mean(v).item(), j)

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
                    set_return_topdown_map=True,
                )

            num_eval_episodes = 16 if "habitat" in args.env_name else 30

            eval_config = {}
            eval_config["num_steps"] = args.num_steps
            eval_config["feat_shape_sim"] = args.feat_shape_sim
            eval_config["num_processes"] = 1 if "habitat" in args.env_name else 12
            eval_config["odometer_shape"] = args.odometer_shape
            eval_config["num_eval_episodes"] = num_eval_episodes
            eval_config["num_pose_refs"] = NREF
            eval_config["env_name"] = args.env_name
            eval_config["actor_type"] = "learned_policy"
            eval_config["encoder_type"] = args.encoder_type
            eval_config["use_action_embedding"] = args.use_action_embedding
            eval_config["use_collision_embedding"] = args.use_collision_embedding
            eval_config["cluster_centroids"] = cluster_centroids
            eval_config["clusters2images"] = clusters2images
            eval_config["rec_loss_fn"] = rec_loss_fn_classify
            eval_config[
                "vis_save_dir"
            ] = f"{args.save_dir}/policy_vis/update_{(j+1):05d}"
            models = {}
            models["decoder"] = decoder
            models["pose_encoder"] = pose_encoder
            models["feature_network"] = feature_network
            models["encoder"] = encoder
            models["actor_critic"] = actor_critic
            val_metrics, _ = evaluate_reconstruction(
                models, eval_envs, eval_config, device
            )
            for k, v in val_metrics.items():
                tbwriter.add_scalar(f"val_metrics/{k}", v, j)

    tbwriter.close()


if __name__ == "__main__":
    main()
