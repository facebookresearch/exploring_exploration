#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import torch
import random
import logging
import numpy as np

from exploring_exploration.arguments import get_args
from exploring_exploration.envs import (
    make_vec_envs_avd,
    make_vec_envs_habitat,
)
from exploring_exploration.utils.common import process_image
from exploring_exploration.models import RGBEncoder, MapRGBEncoder, Policy
from exploring_exploration.utils.eval import evaluate_visitation
from exploring_exploration.utils.storage import RolloutStorageImitation
from exploring_exploration.algo import Imitation
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
    args.feat_shape_pose = (512 * 9,)
    args.obs_shape = envs.observation_space.spaces["im"].shape

    args.agent_action_prob = args.agent_start_action_prob
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
    # =================== Load models ====================
    save_path = os.path.join(args.save_dir, "checkpoints")
    checkpoint_path = os.path.join(save_path, "ckpt.latest.pth")
    if os.path.isfile(checkpoint_path):
        logging.info("Resuming from old model!")
        loaded_states = torch.load(checkpoint_path)
        encoder_state, actor_critic_state, j_start = loaded_states
        encoder.load_state_dict(encoder_state)
        actor_critic.load_state_dict(actor_critic_state)
    else:
        j_start = -1
    actor_critic.to(device)
    encoder.to(device)
    actor_critic.eval()
    encoder.eval()

    # =================== Define IL training algorithm ====================
    il_algo_config = {}
    il_algo_config["lr"] = args.lr
    il_algo_config["eps"] = args.eps
    il_algo_config["max_grad_norm"] = args.max_grad_norm
    il_algo_config["encoder_type"] = args.encoder_type
    il_algo_config["nactions"] = envs.action_space.n

    il_algo_config["encoder"] = encoder
    il_algo_config["actor_critic"] = actor_critic
    il_algo_config["use_action_embedding"] = args.use_action_embedding
    il_algo_config["use_collision_embedding"] = args.use_collision_embedding
    il_algo_config["use_inflection_weighting"] = args.use_inflection_weighting

    il_agent = Imitation(il_algo_config)

    # =================== Define stats buffer ====================
    train_metrics_tracker = defaultdict(lambda: deque(maxlen=10))

    # =================== Define rollouts ====================
    rollouts_policy = RolloutStorageImitation(
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
        # ======================== Start a new episode ========================
        obs = envs.reset()
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
        per_proc_area = [0.0 for proc in range(NPROC)]
        # Other states
        prev_action = torch.zeros(NPROC, 1).long().to(device)
        prev_collision = rollouts_policy.collisions[0]
        agent_acting_duration = 0
        agent_acting_status = False
        # =============== Update over a full batch of episodes ================
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
                    agent_action,
                    action_log_probs,
                    recurrent_hidden_states,
                ) = policy_outputs

            oracle_action = obs["oracle_action"].long()
            # If action mask is active, then take oracle action.
            # Otherwise, take the agent's action
            if args.agent_action_duration == 1:
                action_masks = (
                    torch.cuda.FloatTensor(NPROC, 1).uniform_()
                    >= args.agent_action_prob
                )
            else:  # agent_action_duration HAS to be atleast 2 to enter this
                # Agent continues acting
                if (
                    agent_acting_status
                    and agent_acting_duration > 0
                    and agent_acting_duration <= args.agent_action_duration
                ):
                    action_masks = torch.zeros(NPROC, 1).to(device)
                    agent_acting_duration = (
                        agent_acting_duration + 1
                    ) % args.agent_action_duration
                    # Agent is done acting
                    if agent_acting_duration == 0:
                        agent_acting_status = False
                # Agent starts acting
                elif random.random() < args.agent_action_prob:
                    action_masks = torch.zeros(NPROC, 1).to(device)
                    agent_acting_status = True
                    agent_acting_duration += 1
                # Agent does not act
                else:
                    action_masks = torch.ones(NPROC, 1).to(device)
            action_masks = action_masks.long()
            action = oracle_action * action_masks + agent_action * (1 - action_masks)

            # Act, get reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Processing observations
            obs_im, obs_sm, obs_lm = get_obs(obs)  # (num_processes, 3, 84, 84)
            obs_collns = obs["collisions"]  # (N, 1)

            # Always set masks to 1 (since this loop happens within one episode)
            masks = torch.FloatTensor([[1.0] for _ in range(NPROC)]).to(device)

            # Compute the exploration rewards
            reward_exploration = torch.zeros(NPROC, 1)  # (N, 1)
            for proc in range(NPROC):
                reward_exploration[proc] += (
                    float(infos[proc]["seen_area"]) - per_proc_area[proc]
                )
                per_proc_area[proc] = float(infos[proc]["seen_area"])

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
                action_masks,
            )

            # Update prev values
            prev_collision = obs_collns
            prev_action = action
            episode_collisions += obs_collns.cpu().numpy()

            # Update IL policy
            if (step + 1) % args.num_rl_steps == 0:
                actor_critic.train()
                encoder.train()
                # Update model
                il_losses = il_agent.update(rollouts_policy)
                # Refresh rollouts
                rollouts_policy.after_update()
                actor_critic.eval()
                encoder.eval()

        # =================== Save model ====================
        if (j + 1) % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "checkpoints")
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            encoder_state = encoder.state_dict()
            actor_critic_state = actor_critic.state_dict()
            torch.save(
                [encoder_state, actor_critic_state, j],
                os.path.join(save_path, "ckpt.latest.pth"),
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
            train_metrics = il_losses
            train_metrics["exploration_rewards"] = np.mean(episode_expl_rewards)
            train_metrics["area_covered"] = np.mean(per_proc_area)
            train_metrics["collisions"] = np.mean(episode_collisions)
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
        # =========== Update agent action schedule ==========
        if (j + 1) % args.agent_action_prob_schedule == 0:
            args.agent_action_prob += args.agent_action_prob_factor
            args.agent_action_prob = min(
                args.agent_action_prob, args.agent_end_action_prob
            )
            logging.info(
                f"=======> Updated action sampling schedule to {args.agent_action_prob:.3f}"
            )

    tbwriter.close()


if __name__ == "__main__":
    main()
