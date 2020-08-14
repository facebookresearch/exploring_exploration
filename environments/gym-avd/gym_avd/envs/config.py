#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

GYM_AVD_ROOT = "<PATH TO GYM-AVD>"
ROOT_DIR = "<PATH TO AVD DATASET ROOT>"
CLUSTER_ROOT_DIR = f"{GYM_AVD_ROOT}/gym_avd/data/avd_clusters"
AREAS_FILE = f"{GYM_AVD_ROOT}/gym_avd/data/environment_areas.json"
OBJ_COUNTS_FILE = f"{GYM_AVD_ROOT}/gym_avd/data/object_counts_per_env.json"
OBJ_PROPS_FILE = ""
VALID_INSTANCES_ROOT_DIR = f"{GYM_AVD_ROOT}/gym_avd/data/valid_instances_per_env"
SIZE_CLASSIFICATION_PATH = f"{GYM_AVD_ROOT}/gym_avd/data/size_classification.json.gz"
POINTNAV_TEST_EPISODES_PATH = f"{GYM_AVD_ROOT}/gym_avd/data/tdn_test_episodes.json"
MAX_STEPS = 200
