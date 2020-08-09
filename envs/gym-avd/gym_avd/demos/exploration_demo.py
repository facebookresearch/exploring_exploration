import cv2
import pdb
import gym
import gym_avd
import numpy as np
from utils import *

env = gym.make('avd-pose-landmarks-oracle-v0')
obs = env.reset()
topdown = env.generate_topdown_occupancy()
rgb_im = proc_rgb(obs['im'])
fine_occ_im = proc_rgb(obs['fine_occupancy'])
coarse_occ_im = proc_rgb(obs['coarse_occupancy'])
topdown_im = proc_rgb(topdown)
cv2.imshow('Exploration demo', np.concatenate([rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1))
cv2.waitKey(60)
for i in range(1000):
    # oracle action is generated by sampling shortest paths between random points in the environment.
    action = obs['oracle_action'][0]
    obs, _, done, info = env.step(action)
    if done:
        obs = env.reset()
    topdown = env.generate_topdown_occupancy()
    rgb_im = proc_rgb(obs['im'])
    fine_occ_im = proc_rgb(obs['fine_occupancy'])
    coarse_occ_im = proc_rgb(obs['coarse_occupancy'])
    topdown_im = proc_rgb(topdown)

    metrics_to_print = {
        'Area covered (m^2)': info['seen_area'],
        'Objects covered': info['num_objects_visited'],
        'Landmarks covered': info['oracle_pose_success'],
        'Novelty': info['count_based_reward'],
        'Smooth coverage': info['coverage_novelty_reward'],
    }

    print('===============================================')
    for k, v in metrics_to_print.items():
        print(f'{k:<25s}: {v:6.2f}')

    cv2.imshow('Exploration demo', np.concatenate([rgb_im, fine_occ_im, coarse_occ_im, topdown_im], axis=1))
    cv2.waitKey(60)
