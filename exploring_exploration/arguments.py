#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


def str2bool(v):
    if v.lower() in ["y", "yes", "t", "true"]:
        return True
    return False


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer apha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=500, help="max number of steps in episode"
    )
    parser.add_argument(
        "--num-steps-exp", type=int, default=500, help="max number of steps in episode"
    )
    parser.add_argument(
        "--num-steps-nav", type=int, default=500, help="max number of steps in episode"
    )
    parser.add_argument(
        "--num-rl-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument("--save-unique", type=str2bool, default=False)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10e4,
        help="number of episodes to train (default: 10e4)",
    )
    parser.add_argument(
        "--env-name",
        default="PongNoFrameskip-v4",
        help="environment to train on (default: PongNoFrameskip-v4)",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="directory to save agent logs (default: ./logs/",
    )
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--pretrained-rnet", default="", help="path to pre-trained retrieval network"
    )
    parser.add_argument(
        "--pretrained-posenet",
        default="model.net",
        help="path to pre-trained pairwise pose prediction network",
    )
    parser.add_argument(
        "--pretrained-il-model",
        default="",
        type=str,
        help="path to imitation pretrained policy",
    )
    parser.add_argument(
        "--actor-type",
        default="learned_policy",
        type=str,
        help="can be [ forward | forward-plus | learned_policy | random | oracle ]",
    )
    parser.add_argument(
        "--encoder-type", default="rgb+map", type=str, help="can be [ rgb | rgb+map ]"
    )
    parser.add_argument(
        "--fix-cnn", default=True, type=str2bool, help="Freeze CNN encoders for policy"
    )

    ############################# Pose prediction arguments ###############################
    parser.add_argument("--map-size", type=int, default=21, help="dimension of memory")
    parser.add_argument(
        "--map-scale", type=float, default=1, help="number of pixels per grid length"
    )
    parser.add_argument(
        "--vote-kernel-size", type=int, default=5, help="size of voting kernel"
    )
    parser.add_argument(
        "--num-pose-refs", type=int, default=1, help="number of pose references"
    )
    parser.add_argument("--objects-reward-scale", type=float, default=0.0)
    parser.add_argument("--area-reward-scale", type=float, default=0.0)
    parser.add_argument("--landmarks-reward-scale", type=float, default=0.0)
    parser.add_argument("--novelty-reward-scale", type=float, default=0.0)
    parser.add_argument("--smooth-coverage-reward-scale", type=float, default=0.0)
    parser.add_argument(
        "--pose-predictor-type", default="normal", type=str, help="[ normal | ransac ]"
    )
    parser.add_argument(
        "--match-thresh",
        default=0.95,
        type=float,
        help="minimum score threshold for similarity matches",
    )
    parser.add_argument("--use-classification", type=str2bool, default=False)
    parser.add_argument("--num-classes", type=int, default=15)

    ############################# Imitation Learning arguments ##############################
    parser.add_argument(
        "--agent-start-action-prob",
        type=float,
        default=0.0,
        help="agent will take its own action with this probability",
    )
    parser.add_argument(
        "--agent-end-action-prob",
        type=float,
        default=0.0,
        help="agent will take its own action with this probability",
    )
    parser.add_argument(
        "--agent-action-prob-schedule",
        type=float,
        default=0.0,
        help="the action prob will vary from start to end with a uniform reduction this frequently",
    )
    parser.add_argument(
        "--agent-action-prob-factor",
        type=float,
        default=0.1,
        help="the action prob will vary from start to end with this factor reduction",
    )
    parser.add_argument(
        "--agent-action-duration",
        type=int,
        default=1,
        help="for how long will the agent consecutively act?",
    )
    parser.add_argument(
        "--use-inflection-weighting",
        type=str2bool,
        default=False,
        help="weight actions that differ from previous GT action more in the loss function",
    )

    ############################# Policy Learning arguments ##############################
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--tau", type=float, default=0.95, help="gae parameter (default: 0.95)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="weighting factor between non-exploration and exploration rewards. 1.0 implies pure exploration.",
    )
    parser.add_argument("--collision-penalty-factor", type=float, default=1e-6)
    parser.add_argument(
        "--habitat-config-file",
        type=str,
        default="config.yaml",
        help="path to habitat environment configuration file",
    )
    parser.add_argument(
        "--eval-habitat-config-file",
        type=str,
        default="config.yaml",
        help="path to evaluation habitat environment configuration file",
    )
    parser.add_argument(
        "--use-action-embedding",
        type=str2bool,
        default=True,
        help="use previous actions embedding as input to policy",
    )
    parser.add_argument("--action-embedding-size", type=int, default=32)
    parser.add_argument(
        "--use-collision-embedding",
        type=str2bool,
        default=True,
        help="use collision (or lack of) arising from previous action as input to policy",
    )
    parser.add_argument("--collision-embedding-size", type=int, default=32)
    parser.add_argument("--use-multi-gpu", type=str2bool, default=False)

    ####################### ICM specific arguments ##########################
    parser.add_argument(
        "--icm-embedding-type",
        type=str,
        default="imagenet",
        help="[ imagenet | policy-lstm ]",
    )
    parser.add_argument(
        "--normalize-icm-rewards",
        type=str2bool,
        default=False,
        help="normalize the rewards by standard deviation of returns",
    )

    ####################### Frontier specific arguments ##########################
    parser.add_argument("--frontier-dilate-occ", type=str2bool, default=True)
    parser.add_argument("--max-time-per-target", type=int, default=-1)

    ####################### Reconstruction specific arguments ##########################
    parser.add_argument("--clusters-path", type=str, default="clusters.h5")
    parser.add_argument("--n-transformer-layers", type=int, default=4)
    parser.add_argument(
        "--rec-loss-fn-J",
        type=int,
        default=5,
        help="# of ground truth clusters to select",
    )
    parser.add_argument("--load-path-rec", type=str, default="model.pt")
    parser.add_argument("--rec-reward-scale", type=float, default=0.0)
    parser.add_argument("--rec-reward-interval", type=int, default=20)

    ####################### Evaluation arguments ##########################
    parser.add_argument("--load-path", type=str, default="model.pt")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10000,
        help="number of random episodes to evaluate over",
    )
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--interval_steps", type=int, nargs="+", default=[200])
    parser.add_argument(
        "--ransac-n", default=5, type=int, help="minimum number of ransac matches"
    )
    parser.add_argument(
        "--ransac-niter", default=1000, type=int, help="number of ransac iterations"
    )
    parser.add_argument(
        "--ransac-batch",
        default=5,
        type=int,
        help="number of elements in one ransac batch",
    )
    parser.add_argument("--statistics-save-path", default="results.json", type=str)
    parser.add_argument(
        "--final-topdown-save-dir", type=str, default="eval_topdown_maps.h5"
    )
    parser.add_argument("--enable-odometry-noise", type=str2bool, default=False)
    parser.add_argument("--odometer-noise-scaling", type=float, default=0.0)
    parser.add_argument("--measure-noise-free-area", type=str2bool, default=False)

    ####################### Visualization arguments ##########################
    parser.add_argument(
        "--visualize-batches",
        default=8,
        type=int,
        help="number of test batches to visualize",
    )
    parser.add_argument(
        "--visualize-n-per-batch",
        default=1,
        type=int,
        help="number of elements per batch to visualize",
    )
    parser.add_argument("--visualize-policy", type=str2bool, default=False)
    parser.add_argument("--visualize-save-dir", type=str, default="eval_policy_vis")
    parser.add_argument("--visualize-size", type=int, default=200)
    parser.add_argument("--input-highres", type=str2bool, default=False)

    ####################### Navigation evaluation arguments ##########################
    parser.add_argument("--min-dist", type=float, default=2000.0)
    parser.add_argument("--t-exp", type=int, default=200)
    parser.add_argument("--t-nav", type=int, default=200)
    parser.add_argument("--use-oracle-navigation", type=str2bool, default=False)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.num_refs = 1

    return args
