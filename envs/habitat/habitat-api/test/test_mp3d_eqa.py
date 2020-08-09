#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy as np
import pytest

import habitat
import habitat.datasets.eqa.mp3d_eqa_dataset as mp3d_dataset
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset

CFG_TEST = "configs/test/habitat_mp3d_eqa_test.yaml"
CLOSE_STEP_THRESHOLD = 0.028

# List of episodes each from unique house
TEST_EPISODE_SET = [1, 309, 807, 958, 696, 10, 297, 1021, 1307, 1569]

RGB_EPISODE_MEANS = {
    1: 123.1576333222566,
    10: 123.86094605688947,
    297: 122.69351220853402,
    309: 118.95794969775298,
    696: 115.71903709129052,
    807: 143.7834237211494,
    958: 141.97871610030387,
    1021: 119.1051016229882,
    1307: 102.11408987112925,
    1569: 91.01973929495183,
}

EPISODES_LIMIT = 6


def get_minos_for_sim_eqa_config():
    _sim_eqa_c = get_config(CFG_TEST)
    _sim_eqa_c.task_name = "EQA-v0"
    _sim_eqa_c.dataset = mp3d_dataset.get_default_mp3d_v1_config()
    _sim_eqa_c.dataset.split = "val"
    _sim_eqa_c.scene = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    _sim_eqa_c.height = 512
    _sim_eqa_c.width = 512
    _sim_eqa_c.hfov = "45"
    _sim_eqa_c.vfov = "45"
    _sim_eqa_c.sensor_position = [0, 1.09, 0]
    _sim_eqa_c.forward_step_size = 0.1  # in metres
    _sim_eqa_c.turn_angle = 9  # in degrees
    _sim_eqa_c.sim = "Sim-v0"

    # Agent configuration
    agent_c = _sim_eqa_c.agents[0]
    agent_c.height = 1.5
    agent_c.radius = 0.1
    agent_c.mass = 32.0
    agent_c.linear_acceleration = 10.0
    agent_c.angular_acceleration = 5 * 3.14
    agent_c.linear_friction = 1.0
    agent_c.angular_friction = 1.0
    agent_c.coefficient_of_restitution = 0.15707963267

    return _sim_eqa_c


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = str(dataset.to_json())
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = dataset.__class__()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) > 0
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_mp3d_eqa_dataset():
    dataset_config = get_config(CFG_TEST).DATASET
    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        dataset_config
    ):
        pytest.skip(
            "Please download Matterport3D EQA dataset to " "data folder."
        )

    dataset = mp3d_dataset.Matterport3dDatasetV1(config=dataset_config)
    assert dataset
    assert (
        len(dataset.episodes) == mp3d_dataset.EQA_MP3D_V1_VAL_EPISODE_COUNT
    ), "Test split episode number mismatch"
    check_json_serializaiton(dataset)


def test_mp3d_eqa_sim():
    eqa_config = get_config(CFG_TEST)

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        eqa_config.DATASET
    ):
        pytest.skip(
            "Please download Matterport3D EQA dataset to " "data folder."
        )

    dataset = make_dataset(
        id_dataset=eqa_config.DATASET.TYPE, config=eqa_config.DATASET
    )
    env = habitat.Env(config=eqa_config, dataset=dataset)
    env.episodes = dataset.episodes[:EPISODES_LIMIT]

    assert env
    env.reset()
    while not env.episode_over:
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs = env.step(action)
        if not env.episode_over:
            assert "rgb" in obs, "RGB image is missing in observation."
            assert obs["rgb"].shape[:2] == (
                eqa_config.SIMULATOR.RGB_SENSOR.HEIGHT,
                eqa_config.SIMULATOR.RGB_SENSOR.WIDTH,
            ), (
                "Observation resolution {} doesn't correspond to config "
                "({}, {}).".format(
                    obs["rgb"].shape[:2],
                    eqa_config.SIMULATOR.RGB_SENSOR.HEIGHT,
                    eqa_config.SIMULATOR.RGB_SENSOR.WIDTH,
                )
            )

    env.close()


def test_mp3d_eqa_sim_correspondence():
    eqa_config = get_config(CFG_TEST)

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        eqa_config.DATASET
    ):
        pytest.skip(
            "Please download Matterport3D EQA dataset to " "data folder."
        )

    dataset = make_dataset(
        id_dataset=eqa_config.DATASET.TYPE, config=eqa_config.DATASET
    )
    env = habitat.Env(config=eqa_config, dataset=dataset)
    env.episodes = [
        episode
        for episode in dataset.episodes
        if int(episode.episode_id) in TEST_EPISODE_SET[:EPISODES_LIMIT]
    ]

    ep_i = 0
    cycles_n = 2
    while cycles_n > 0:
        env.reset()
        episode = env.current_episode
        assert (
            len(episode.goals) == 1
        ), "Episode has no goals or more than one."
        assert (
            len(episode.shortest_paths) == 1
        ), "Episode has no shortest paths or more than one."
        start_state = env.sim.get_agent_state()
        assert np.allclose(
            start_state.position, episode.start_position
        ), "Agent's start position diverges from the shortest path's one."

        rgb_mean = 0
        logger.info(
            "{id} {question}\n{answer}".format(
                id=episode.episode_id,
                question=episode.question.question_text,
                answer=episode.question.answer_text,
            )
        )

        for step_id, point in enumerate(episode.shortest_paths[0]):
            cur_state = env.sim.get_agent_state()

            logger.info(
                "diff position: {} diff rotation: {} "
                "cur_state.position: {} shortest_path.position: {} "
                "cur_state.rotation: {} shortest_path.rotation: {} action: {}"
                "".format(
                    cur_state.position - point.position,
                    cur_state.rotation
                    - habitat.utils.geometry_utils.quaternion_wxyz_to_xyzw(
                        point.rotation
                    ),
                    cur_state.position,
                    point.position,
                    cur_state.rotation,
                    point.rotation,
                    point.action,
                )
            )

            assert np.allclose(
                [cur_state.position[0], cur_state.position[2]],
                [point.position[0], point.position[2]],
                atol=CLOSE_STEP_THRESHOLD * (step_id + 1),
            ), "Agent's path diverges from the shortest path."

            obs = env.step(point.action)

            if not env.episode_over:
                rgb_mean += obs["rgb"][:, :, :3].mean()

        if ep_i < len(RGB_EPISODE_MEANS):
            rgb_mean = rgb_mean / len(episode.shortest_paths[0])
            assert np.isclose(
                RGB_EPISODE_MEANS[int(episode.episode_id)], rgb_mean
            ), "RGB output doesn't match the ground truth."

        ep_i = (ep_i + 1) % EPISODES_LIMIT
        if ep_i == 0:
            cycles_n -= 1

    env.close()
