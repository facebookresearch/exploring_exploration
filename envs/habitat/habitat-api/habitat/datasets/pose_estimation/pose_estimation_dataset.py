#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from itertools import cycle
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.pose_estimation.pose_estimation_task import (
    PoseEstimationEpisode,
    ShortestPathPoint,
)

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PoseEstimation-v1")
class PoseEstimationDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Pose Estimation dataset.
    """

    episodes: List[PoseEstimationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    shuffle_dataset: bool = True

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert PoseEstimationDatasetV1.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(config.DATA_PATH.format(split=config.SPLIT))

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = PoseEstimationDatasetV1(cfg)
        return PoseEstimationDatasetV1._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path, dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        self.shuffle_dataset = config.SHUFFLE_DATASET

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = PoseEstimationDatasetV1._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path, dataset_dir=dataset_dir,
            )

        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        self.sample_episodes(config.NUM_EPISODE_SAMPLE)

    def get_episode_iterator(self):
        r"""
        Creates and returns an iterator that iterates through self.episodes
        in the desirable way specified.
        Returns:
            iterator for episodes
        """
        # TODO: support shuffling between epoch and scene switching
        if self.shuffle_dataset:
            self.shuffle_episodes(50)
        return cycle(self.episodes)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = PoseEstimationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # Ignore edge case episodes where no references were sampled
            if len(episode.pose_ref_positions) == 0:
                continue

            self.episodes.append(episode)
