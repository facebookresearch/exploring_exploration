# Active Vision Simulator
This directory contains the code to an [OpenAI gym](https://gym.openai.com/)-based environment for simulating discrete motion on the [Active Vision Dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/). 

## Installation instructions
1. Install dependencies.

  ```
  pip install -r requirements.txt
  export GYM_AVD_ROOT=<path to gym-avd directory>
  ```
2. Install `gym-avd`.

  ```
  cd $GYM_AVD_ROOT
  python setup.py install
  ```
3. Add the code root to `~/.bashrc`.

  ```
  export PYTHONPATH=$GYM_AVD_ROOT:$PYTHONPATH
  ```
4. Download data from the [AVD website](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html). The camera calibration information can be obtained from the author of AVD.

  ```
  tar -xvf ActiveVisionDataset_part1.tar
  tar -xvf ActiveVisionDataset_part2.tar
  tar -xvf ActiveVisionDataset_part3.tar
  tar -xvf ActiveVisionDataset_COLMAP_camera_params_part1-3.tar

  export AVD_DATASET_ROOT=<path to ActiveVisionDataset directory>
  ```

5. Download additional processed data for simulation.

 ```
 cd $GYM_AVD_ROOT/gym_avd
 mkdir data
 cd data
 wget https://dl.fbaipublicfiles.com/exploring-exploration/avd_extra_data.tar.gz -o data.tar.gz
 tar -xvf data.tar.gz
 rm data.tar.gz
 ```
6. Set configuration paths for the simulator in `$GYM_AVD_ROOT/gym_avd/envs/config.py`.

  ```
  GYM_AVD_ROOT=<path in GYM_AVD_ROOT>
  ROOT_DIR=<path in AVD_DATASET_ROOT>
  ```
7. Process dataset to extract images and connectivity:

  ```
  cd $GYM_AVD_ROOT
  python preprocess_raw_data.py --root_dir $AVD_DATASET_ROOT
  ```
  This will create the following files:

  ```
  $AVD_DATASET_ROOT/processed_images_84x84.h5
  $AVD_DATASET_ROOT/processed_scenes_84x84.npy
  ```

## Task demos
This repository supports four tasks:

- Exploration
- Pose estimation
- Reconstruction
- PointNav

Visual demos for each task are available.

```
cd $GYM_AVD_ROOT
python gym_avd/demos/exploration_demo.py
python gym_avd/demos/pose_estimation_demo.py
python gym_avd/demos/reconstruction_demo.py
python gym_avd/demos/pointnav_demo.py
```
