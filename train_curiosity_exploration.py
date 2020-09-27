#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import torch
import logging
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from exploring_exploration.arguments import get_args
from exploring_exploration.envs import (
    make_vec_envs_avd,
    make_vec_envs_habitat,
)
from exploring_exploration.utils.common import (
    process_image,
    random_range,
)
from exploring_exploration.models import RGBEncoder, MapRGBEncoder, Policy
from exploring_exploration.models.curiosity import (
    ForwardDynamics,
    Phi,
    RunningMeanStd,
)
from exploring_exploration.utils.eval import evaluate_visitation
from exploring_exploration.utils.storage import RolloutStoragePPO
from exploring_exploration.algo import PPO
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
    args.feat_shape_pose = (512 * 9,)
    args.obs_shape = envs.observation_space.spaces["im"].shape

    # =================== Create models ====================
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
    icm_phi = Phi() if args.icm_embedding_type == "imagenet" else None
    icm_fd = ForwardDynamics(envs.action_space.n)
    # =================== Load models ====================
    save_path = os.path.join(args.save_dir, "checkpoints")
    checkpoint_path = os.path.join(save_path, "ckpt.latest.pth")
    if os.path.isfile(checkpoint_path):
        logging.info("Resuming from old model!")
        loaded_states = torch.load(checkpoint_path)
        encoder_state, actor_critic_state, icm_fd_state, j_start = loaded_states
        encoder.load_state_dict(encoder_state)
        actor_critic.load_state_dict(actor_critic_state)
        icm_fd.load_state_dict(icm_fd_state)
    elif args.pretrained_il_model != "":
        logging.info("Initializing with pre-trained model!")
        encoder_state, actor_critic_state, _ = torch.load(args.pretrained_il_model)
        encoder.load_state_dict(encoder_state)
        actor_critic.load_state_dict(actor_critic_state)
        j_start = -1
    else:
        j_start = -1
    encoder.to(device)
    actor_critic.to(device)
    if args.icm_embedding_type == "imagenet":
        icm_phi.to(device)
    icm_fd.to(device)
    encoder.train()
    actor_critic.train()
    if args.icm_embedding_type == "imagenet":
        icm_phi.eval()  # Do not train/the feature model for ICM
    icm_fd.train()
    # =================== Define ICM training algorithm ====================
    icm_optimizer = optim.Adam(icm_fd.parameters(), lr=args.lr)
    # Maintain a running mean of the variance of returns after every
    # num-rl-steps
    if args.normalize_icm_rewards:
        args.returns_rms = RunningMeanStd()
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

    # =================== Define rollouts ====================
    rollouts_policy = RolloutStoragePPO(
        args.num_rl_steps,
        args.num_processes,
        args.obs_shape,
        envs.action_space,
        args.feat_shape_sim[0],
        encoder_type=args.encoder_type,
    )
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
    for j in range(j_start + 1, num_updates):
        # =================== Start a new episode ====================
        obs = envs.reset()
        # Reset ICM data buffer
        all_icm_feats = []
        all_icm_acts = []
        # Set icm models to evaluate mode for data gathering
        if args.icm_embedding_type == "imagenet":
            icm_phi.eval()
        icm_fd.eval()
        # Processing environment inputs
        obs_im, obs_sm, obs_lm = get_obs(obs)  # (num_processes, 3, 84, 84)
        obs_collns = obs["collisions"].long()  # (num_processes, 1)
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
        episode_collisions += obs_collns.cpu().numpy()
        # Metrics
        osr_tracker = [0.0 for _ in range(NPROC)]
        objects_tracker = [0.0 for _ in range(NPROC)]
        area_tracker = [0.0 for _ in range(NPROC)]
        novelty_tracker = [0.0 for _ in range(NPROC)]
        smooth_coverage_tracker = [0.0 for _ in range(NPROC)]
        per_proc_area = [0.0 for _ in range(NPROC)]
        # Other states
        prev_action = torch.zeros(NPROC, 1).to(device)
        prev_collision = obs_collns
        action_onehot = torch.zeros(NPROC, envs.action_space.n).to(
            device
        )  # (N, n_actions)
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

            # Gather curiosity experience. By default, the features are deatached
            # from the forward dynamics loss.
            if args.icm_embedding_type == "imagenet":
                with torch.no_grad():
                    icm_feats = icm_phi(obs_im)
            else:
                icm_feats = recurrent_hidden_states
            all_icm_feats.append(icm_feats)
            all_icm_acts.append(action)

            # Act, get reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Processing environment inputs
            obs_im, obs_sm, obs_lm = get_obs(obs)  # (num_processes, 3, 84, 84)
            obs_collns = obs["collisions"]  # (num_processes, 1)

            # Always set masks to 1 (since this loop happens within one episode)
            masks = torch.FloatTensor([[1.0] for _ in range(NPROC)]).to(device)

            # Compute curiosity rewards for the previous action (not the current)
            reward_exploration = torch.zeros(NPROC, 1)
            if step >= 1:
                phi_st = all_icm_feats[-2]
                phi_st1 = all_icm_feats[-1]
                action_onehot.zero_()
                act = all_icm_acts[-2]
                action_onehot.scatter_(1, act, 1)
                with torch.no_grad():
                    phi_st1_hat = icm_fd(phi_st, action_onehot)
                reward_exploration = (
                    F.mse_loss(phi_st1_hat, phi_st1, reduction="none")
                    .sum(dim=1)
                    .unsqueeze(1)
                    .detach()
                )  # (N, 1)
                # Since this reward corresponds to the previous action, update it
                # accordingly in the rollouts buffer.
                rollouts_policy.update_prev_rewards(
                    reward_exploration * args.reward_scale
                )
                reward_exploration = reward_exploration.cpu()

            for proc in range(NPROC):
                seen_area = float(infos[proc]["seen_area"])
                objects_visited = infos[proc].get("num_objects_visited", 0.0)
                oracle_success = float(infos[proc].get("oracle_pose_success", 0.0))
                novelty_reward = infos[proc].get("count_based_reward", 0.0)
                smooth_coverage_reward = infos[proc].get("coverage_novelty_reward", 0.0)
                area_tracker[proc] = seen_area
                objects_tracker[proc] = objects_visited
                osr_tracker[proc] = oracle_success
                per_proc_area[proc] = seen_area
                novelty_tracker[proc] += novelty_reward
                smooth_coverage_tracker[proc] += smooth_coverage_reward

            # Instrinsic reward is updated separately (delayed by 1 time step)
            overall_reward = reward * (1 - args.reward_scale)

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
                # Normalize the rewards if applicable
                if args.normalize_icm_rewards:
                    current_returns = 0.0
                    for rew in torch.flip(rollouts_policy.rewards, dims=[0]):
                        current_returns = current_returns * args.gamma + rew
                    current_returns = current_returns.squeeze(1).cpu().numpy()
                    args.returns_rms.update(current_returns)
                    rollouts_policy.rewards /= args.returns_rms.var.item()
                # Compute returns
                rollouts_policy.compute_returns(
                    next_value, args.use_gae, args.gamma, args.tau,
                )
                # Update model
                rl_losses = rl_agent.update(rollouts_policy)
                # Refresh rollouts
                rollouts_policy.after_update()

        # ============ Update the ICM dynamics model using past data ===============
        icm_fd.train()
        action_onehot = torch.zeros(NPROC, envs.action_space.n).to(
            device
        )  # (N, n_actions)
        avg_fd_loss = 0
        avg_fd_loss_count = 0
        icm_update_count = 0
        for t in random_range(0, args.num_steps - 1):
            phi_st = all_icm_feats[t]  # (N, 512)
            phi_st1 = all_icm_feats[t + 1]  # (N, 512)
            action_onehot.zero_()
            at = all_icm_acts[t].long()  # (N, 1)
            action_onehot.scatter_(1, at, 1)
            # Forward pass
            phi_st1_hat = icm_fd(phi_st, action_onehot)
            fd_loss = F.mse_loss(phi_st1_hat, phi_st1)
            # Backward pass
            icm_optimizer.zero_grad()
            fd_loss.backward()
            torch.nn.utils.clip_grad_norm_(icm_fd.parameters(), args.max_grad_norm)
            # Update step
            icm_optimizer.step()
            avg_fd_loss += fd_loss.item()
            avg_fd_loss_count += phi_st1_hat.shape[0]
        avg_fd_loss /= avg_fd_loss_count
        all_losses = {"icm_fd_loss": avg_fd_loss}

        # =================== Save model ====================
        if (j + 1) % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "checkpoints")
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            encoder_state = encoder.state_dict()
            actor_critic_state = actor_critic.state_dict()
            icm_fd_state = icm_fd.state_dict()
            torch.save(
                [encoder_state, actor_critic_state, icm_fd_state, j],
                f"{save_path}/ckpt.latest.pth",
            )
            if args.save_unique:
                torch.save(
                    [encoder_state, actor_critic_state, icm_fd_state, j],
                    f"{save_path}/ckpt.{(j+1):07d}.pth",
                )

        # =================== Logging data ====================
        total_num_steps = (j + 1 - j_start) * NPROC * args.num_steps
        if j % args.log_interval == 0:
            end = time.time()
            fps = int(total_num_steps / (end - start))
            logging.info(f"===> Updates {j}, #steps {total_num_steps}, FPS {fps}")
            train_metrics = rl_losses
            train_metrics.update(all_losses)
            train_metrics["exploration_rewards"] = np.mean(episode_expl_rewards)
            train_metrics["area_covered"] = np.mean(per_proc_area)
            train_metrics["objects_covered"] = np.mean(objects_tracker)
            train_metrics["landmarks_covered"] = np.mean(osr_tracker)
            train_metrics["collisions"] = np.mean(episode_collisions)
            train_metrics["novelty_rewards"] = np.mean(novelty_tracker)
            train_metrics["smooth_coverage_rewards"] = np.mean(smooth_coverage_tracker)

            for k, v in train_metrics.items():
                logging.info(f"{k}: {v:.3f}")
                tbwriter.add_scalar(f"train_metrics/{k}", v, j)

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
                    nRef=args.num_pose_refs,
                    set_return_topdown_map=True,
                )

            num_eval_episodes = 16 if "habitat" in args.env_name else 30

            eval_config = {}
            eval_config["num_steps"] = args.num_steps
            eval_config["feat_shape_sim"] = args.feat_shape_sim
            eval_config["num_processes"] = 1 if "habitat" in args.env_name else 12
            eval_config["num_pose_refs"] = args.num_pose_refs
            eval_config["num_eval_episodes"] = num_eval_episodes
            eval_config["env_name"] = args.env_name
            eval_config["actor_type"] = "learned_policy"
            eval_config["encoder_type"] = args.encoder_type
            eval_config["use_action_embedding"] = args.use_action_embedding
            eval_config["use_collision_embedding"] = args.use_collision_embedding
            eval_config[
                "vis_save_dir"
            ] = f"{args.save_dir}/policy_vis/update_{(j+1):05d}"
            models = {}
            models["encoder"] = encoder
            models["actor_critic"] = actor_critic
            val_metrics, _ = evaluate_visitation(
                models, eval_envs, eval_config, device, visualize_policy=False
            )
            for k, v in val_metrics.items():
                tbwriter.add_scalar(f"val_metrics/{k}", v, j)

    tbwriter.close()


if __name__ == "__main__":
    main()
