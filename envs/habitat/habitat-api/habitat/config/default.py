#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

from habitat.config import Config as CN  # type: ignore

DEFAULT_CONFIG_DIR = "configs"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SUCCESS_DISTANCE = 0.2
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
_C.TASK.GOAL_SENSOR_UUID = "pointgoal"
# -----------------------------------------------------------------------------
# # POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
# -----------------------------------------------------------------------------
# # STATIC POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.STATIC_POINTGOAL_SENSOR = CN()
_C.TASK.STATIC_POINTGOAL_SENSOR.TYPE = "StaticPointGoalSensor"
_C.TASK.STATIC_POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
# -----------------------------------------------------------------------------
# # HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# # PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# # ORACLE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_ACTION_SENSOR = CN()
_C.TASK.ORACLE_ACTION_SENSOR.TYPE = "OracleActionSensor"
_C.TASK.ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.25
_C.TASK.ORACLE_ACTION_SENSOR.ORACLE_TYPE = "random"
_C.TASK.ORACLE_ACTION_SENSOR.NUM_TARGETS = 500
# -----------------------------------------------------------------------------
# # SHORTEST PATH ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SP_ACTION_SENSOR = CN()
_C.TASK.SP_ACTION_SENSOR.TYPE = "SPActionSensor"
_C.TASK.SP_ACTION_SENSOR.GOAL_RADIUS = 0.25
# -----------------------------------------------------------------------------
# # SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
_C.TASK.SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# # TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
# -----------------------------------------------------------------------------
# # COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/" "van-gogh-room.glb"
)
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
_C.SIMULATOR.ENABLE_ODOMETRY_NOISE = False
_C.SIMULATOR.ODOMETER_NOISE_SCALING = 0.0
# -----------------------------------------------------------------------------
# # SENSORS
# -----------------------------------------------------------------------------
SENSOR = CN()
SENSOR.HEIGHT = 84
SENSOR.WIDTH = 84
SENSOR.HFOV = 90  # horizontal field of view in degrees
SENSOR.POSITION = [0, 1.25, 0]
# -----------------------------------------------------------------------------
# # RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
_C.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0
_C.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10
_C.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# # FINE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FINE_OCC_SENSOR = SENSOR.clone()
_C.SIMULATOR.FINE_OCC_SENSOR.TYPE = "HabitatSimFineOccSensor"
# -----------------------------------------------------------------------------
# # COARSE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.COARSE_OCC_SENSOR = SENSOR.clone()
_C.SIMULATOR.COARSE_OCC_SENSOR.TYPE = "HabitatSimCoarseOccSensor"
# -----------------------------------------------------------------------------
# # PROJ-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.PROJ_OCC_SENSOR = SENSOR.clone()
_C.SIMULATOR.PROJ_OCC_SENSOR.TYPE = "HabitatSimProjOccSensor"
# -----------------------------------------------------------------------------
# # HIGHRES-COARSE-OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HIGHRES_COARSE_OCC_SENSOR = SENSOR.clone()
_C.SIMULATOR.HIGHRES_COARSE_OCC_SENSOR.TYPE = \
        "HabitatSimHighResCoarseOccSensor"
# -----------------------------------------------------------------------------
# # OCCUPANCY MAPS
# -----------------------------------------------------------------------------
_C.SIMULATOR.OCCUPANCY_MAPS = CN()
_C.SIMULATOR.OCCUPANCY_MAPS.HEIGHT = SENSOR.HEIGHT
_C.SIMULATOR.OCCUPANCY_MAPS.WIDTH = SENSOR.WIDTH
_C.SIMULATOR.OCCUPANCY_MAPS.MAP_SCALE = 0.1
_C.SIMULATOR.OCCUPANCY_MAPS.MAP_SIZE = 800
_C.SIMULATOR.OCCUPANCY_MAPS.MAX_DEPTH = 3
_C.SIMULATOR.OCCUPANCY_MAPS.SMALL_MAP_RANGE = 20
_C.SIMULATOR.OCCUPANCY_MAPS.LARGE_MAP_RANGE = 100
_C.SIMULATOR.OCCUPANCY_MAPS.HEIGHT_LOWER = 0.5
_C.SIMULATOR.OCCUPANCY_MAPS.HEIGHT_UPPER = 2.0
_C.SIMULATOR.OCCUPANCY_MAPS.GET_PROJ_LOC_MAP = False
_C.SIMULATOR.OCCUPANCY_MAPS.GET_HIGHRES_LOC_MAP = False
_C.SIMULATOR.OCCUPANCY_MAPS.USE_GT_OCC_MAP = False
# -----------------------------------------------------------------------------
# # OBJECT ANNOTATIONS
# -----------------------------------------------------------------------------
_C.SIMULATOR.OBJECT_ANNOTATIONS = CN()
_C.SIMULATOR.OBJECT_ANNOTATIONS.IS_AVAILABLE = False
_C.SIMULATOR.OBJECT_ANNOTATIONS.PATH = './'
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.MASS = 32.0
_C.SIMULATOR.AGENT_0.LINEAR_ACCELERATION = 20.0
_C.SIMULATOR.AGENT_0.ANGULAR_ACCELERATION = 4 * 3.14
_C.SIMULATOR.AGENT_0.LINEAR_FRICTION = 0.5
_C.SIMULATOR.AGENT_0.ANGULAR_FRICTION = 1.0
_C.SIMULATOR.AGENT_0.COEFFICIENT_OF_RESTITUTION = 0.0
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.NUM_EPISODE_SAMPLE = -1
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)
_C.DATASET.SHUFFLE_DATASET = True


# -----------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
