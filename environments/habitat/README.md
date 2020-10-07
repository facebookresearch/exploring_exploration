# Habitat
Our project uses a modified version of the original [habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) repositories for simulating 3D motion in Matterport3D dataset. 

## Installing habitat-sim
1. Create a root directory for Habitat.

  ```
  export HABITAT_ROOT=<path to exploring_exploration/envs/habitat/>
  mkdir $HABITAT_ROOT
  cd $HABITAT_ROOT
  ```
2. Clone `habitat-sim` and checkout a specific version used for this code-base.

  ```
  git clone git@github.com:facebookresearch/habitat-sim.git
  cd $HABITAT_ROOT/habitat-sim
  git checkout 15994e440560c1608b251a1c4059507d1cae801b
  ```

3. Follow installation instructions from `https://github.com/facebookresearch/habitat-sim` (at that specific commit).

4. Apply `habitat_sim.patch` to `habitat-sim` repository. This will incorporate some minor additions to the original simulator.

  ```
  cd $HABITAT_ROOT
  cp habitat_sim.patch habitat-sim
  cd habitat-sim
  git apply habitat_sim.patch
  ```

## Installing habitat-api
1. Clone `habitat-lab` and checkout a specific version used for this code-base.

  ```
  cd $HABITAT_ROOT
  git clone git@github.com:facebookresearch/habitat-lab.git habitat-api
  cd habitat-api
  git checkout 31318f81db05100099cfd308438d5930c3fb6cd2
  ```
2. Follow the [installation instructions](https://github.com/facebookresearch/habitat-api). Download the Matterport3D scene dataset as instructed.
3. Apply `habitat_api.patch` to `habitat-api` repository. This will incorporate the necessary additions to the original api.

  ```
  cd $HABITAT_ROOT
  cp habitat_sim.patch habitat-api
  cd habitat-api
  patch -p0 < habitat_api.patch
  ```

4. Download the task datasets.

  ```
  mkdir -p $HABITAT_ROOT/habitat-api/data
  cd $HABITAT_ROOT/habitat-api/data
  wget -O task_datasets.tar.gz https://dl.fbaipublicfiles.com/exploring-exploration/mp3d_task_datasets.tar.gz
  tar -xvf task_datasets.tar.gz
  rm task_datasets.tar.gz
  ```
5. Extract object annotations for MP3D:

  ```
  cd $HABITAT_ROOT/habitat-api
  python data_generation_scripts/extract_object_annotations_per_env.py
  ```

## Task demos
This repository supports four tasks:

- Exploration
- Pose estimation
- Reconstruction
- PointNav

Visual demos for each task are available.

```
python demos/exploration_demo.py
python demos/pose_estimation_demo.py
python demos/reconstruction_demo.py
python demos/pointnav_demo.py
```
