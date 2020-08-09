# An Exploration of Embodied Visual Exploration
This repository contains the code to run experiments from our work: [http://vision.cs.utexas.edu/projects/exploring-exploration/](http://vision.cs.utexas.edu/projects/exploring-exploration/) 

## Simulation environments
This codebase supports experiments on Active Vision dataset and Matterport3D. The corresponding simulators can be installed from the following directories:

- Active Vision simulator: `envs/gym-avd`
- Matterport3D simulator (via habitat-api): `envs/habitat`

## Installation instructions
1. Clone this github repository and add to path.

	```
	cd exploring_exploration
	export PYTHONPATH=$PWD:$PYTHONPATH
	export EXPLORING_EXPLORATION=$PWD
	```
1. Install dependencies.

	```
	cd $EXPLORING_EXPLORATION
	pip install requirements.txt
	```
1. Install OpenAI baselines

	```
	cd $EXPLORING_EXPLORATION/baselines
	python setup.py install
	```
1. Install `astar_pycpp` for fast planning.

	```
	cd $EXPLORING_EXPLORATION/exploring_exploration/models/
	git clone git@github.com:srama2512/astar_pycpp.git
	cd astar_pycpp
	git checkout exploration_study
	make
	```

## Downloading pre-trained models
We provide pre-trained models for different baselines and paradigms on both AVD and MP3D.

```
cd $EXPLORING_EXPLORATION
mkdir pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/exploring-exploration/avd_pretrained_models.tar.gz
wget https://dl.fbaipublicfiles.com/exploring-exploration/mp3d_pretrained_models.tar.gz
```

## Evaluating on visitation metrics

We evaluate exploration using three visitation metrics: the amount of area / landmarks / objects visited during exploration. The `evaluate_visitation.py` script evaluates performance on the visitation metrics.

### Evaluation on AVD
```
cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>

python -W ignore evaluate_visitation.py \
     --seed 123 \
     --num-steps 200 \
     --env-name avd-pose-v0 \
     --eval-split test \
     --num-processes 16 \
     --num-pose-refs 10 \
     --load-path $model_path \
     --eval-episodes 100 \
     --interval_steps 200 \
     --actor-type learned_policy \
     --visualize-policy False \
     --log-dir visitation_results
```

### Evaluation on MP3D

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>

python -W ignore evaluate_visitation.py \
     --seed 123 \
     --num-steps 1000 \
     --env-name habitat \
     --eval-split test \
     --num-processes 1 \
     --num-pose-refs 20 \
     --load-path $model_path \
     --eval-episodes 10 \
     --interval_steps 1000 \
     --actor-type learned_policy \
     --habitat-config-file configs/exploration/ppo_pose_test.yaml \
     --visualize-policy False \
     --log-dir visitation_results
```

Here, the metrics are computed after exploring for 1000 steps on each episode. The average results over all episodes are logged in `visitation_results/eval_log.txt`. Detailed statistics on a per-episode basis are stored in `visitation_results/statistics.json`.
### Note
- Performance can be measured at intermediate time-steps by setting the appropriate values for `--interval-steps`. For example, setting `--interval-steps 50 100 150 200` for AVD would additionally record the performance at 50, 100, 150 and 200 steps. 
- The number of evaluation episodes can be controlled by varying `--eval-episodes`. Evaluating on MP3D takes ~0.6 minutes per episode on a Quadro GP100. Evaluating on the full test set (1000 episodes) would take ~12 hours.

## Evaluating on view localization

The view localization task requires the agent to estimate the pose of a set of reference views in the environment using the information gathered during exploration. See Sec. 6 in the [supplementary](http://vision.cs.utexas.edu/projects/exploring-exploration/supp.pdf) for more details. We provide pre-trained pose estimation models for AVD and MP3D which can be used for evaluation in `$EXPLORING_EXPLORATION/pretrained_models/*/pose_estimation_nets`.

### Evaluation on AVD

```
cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>
export retrieval_net_path=<PATH TO RETRIEVAL NET>
export pairwise_pose_net_path=<PATH TO PAIRWISE POSE PREDICTOR>

python -W ignore evaluate_pose_estimation.py \
     --num-steps 200 \
     --map-size 31 \
     --env-name avd-pose-v0 \
     --map-scale 500.0 \
     --vote-kernel-size 3 \
     --num-processes 16 \
     --num-pose-refs 10 \
     --load-path $model_path \
     --eval-split test \
     --seed 123 \
     --pretrained-rnet $retrieval_net_path \
     --pretrained-posenet $pairwise_pose_net_path \
     --eval-episodes 100 \
     --interval_steps 200 \
     --actor-type learned_policy \
     --pose-predictor-type ransac \
     --ransac-niter 10 \
     --log-dir pose_estimation_results \
     --visualize-policy False
```

### Evaluation on MP3D

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>
export retrieval_net_path=<PATH TO RETRIEVAL NET>
export pairwise_pose_net_path=<PATH TO PAIRWISE POSE PREDICTOR>

python -W ignore evaluate_pose_estimation.py \
     --num-steps 1000 \
     --map-size 101 \
     --env-name habitat \
     --map-scale 0.5 \
     --vote-kernel-size 5 \
     --num-processes 1 \
     --num-pose-refs 20 \
     --load-path $model_path \
     --eval-split test \
     --seed 123 \
     --pretrained-rnet $retrieval_net_path \
     --pretrained-posenet $pairwise_pose_net_path \
     --eval-episodes 10 \
     --interval_steps 1000 \
     --actor-type learned_policy \
     --pose-predictor-type ransac \
     --ransac-niter 10 \
     --use-classification True \
     --num-classes 15 \
     --visualize-policy False \
     --use-multi-gpu True \
     --habitat-config-file configs/pose_estimation/ppo_pose_test.yaml \
     --log-dir pose_estimation_results
```

### Note
- Evaluating on MP3D takes ~1 minute per episode on a Quadro GP100. Evaluating on the full test set (1000 episodes) would take ~16 hours.


## Evaluating on reconstruction

The reconstruction task requires the agent to accurately reconstruct the concepts present at a set of reference poses in the environment. See Sec. 6 in the [supplementary](http://vision.cs.utexas.edu/projects/exploring-exploration/supp.pdf). We provide pre-trained reconstruction task-heads for AVD and MP3D which can be used for evaluation in `$EXPLORING_EXPLORATION/pretrained_models/*/pretrained_reconstruction/ckpt.pth`. We also provide the concept clusters extracted for both datasets [here](https://dl.fbaipublicfiles.com/exploring-exploration/reconstruction_data.tar.gz).

### Evaluation on AVD

```
cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>
export reconstruction_head_path=<PATH TO RECONSTRUCTION HEAD MODEL>
export clusters_path=reconstruction_data_generation/avd/imagenet_clusters/clusters_00030_data.h5

python -W ignore evaluate_reconstruction.py \
     --num-steps 200 \
     --env-name avd-recon-v0 \
     --load-path $model_path \
     --num-processes 16 \
     --seed 123 \
     --num-pose-refs 50 \
     --eval-split test \
     --clusters-path $clusters_path \
     --n-transformer-layers 2 \
     --load-path-rec $reconstruction_head_path \
     --eval-episodes 100 \
     --interval_steps 200 \
     --actor-type learned_policy \
     --visualize-policy False \
     --log-dir reconstruction_results
```

### Evaluation on MP3D

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
export model_path=<PATH TO MODEL>
export reconstruction_head_path=<PATH TO RECONSTRUCTION HEAD MODEL>
export clusters_path=reconstruction_data_generation/mp3d/imagenet_clusters/clusters_00050_data.h5

python -W ignore evaluate_reconstruction.py \
     --num-steps 1000 \
     --env-name habitat \
     --load-path $model_path \
     --num-processes 1 \
     --seed 123 \
     --num-pose-refs 100 \
     --eval-split test \
     --clusters-path $clusters_path \
     --n-transformer-layers 2 \
     --load-path-rec $reconstruction_head_path \
     --eval-episodes 10 \
     --interval_steps 1000 \
     --actor-type learned_policy \
     --visualize-policy False \
     --habitat-config-file configs/reconstruction_exploration/ppo_pose_test.yaml \
     --use-multi-gpu True \
     --log-dir reconstruction_results
```

### Note
- Evaluating on MP3D takes ~0.8 minute per episode on a Quadro GP100. Evaluating on the full test set (1000 episodes) would take ~13 hours.

## Training on Active Vision simulator

### Imitation learning pre-training
A simple exploration policy can be trained by imitating an oracle agent that sequentially visits a set of pre-defined locations in the environment using the shortest paths.

```
cd $EXPLORING_EXPLORATION
mkdir -p trained_models/imitation_learning/avd

python -W ignore -u pretrain_imitation.py  \
     --lr 1e-4 \
     --seed 123 \
     --num-processes 16 \
     --num-steps 200 \
     --num-rl-steps 50 \
     --env-name avd-pose-landmarks-oracle-v0 \
     --save-interval 100 \
     --eval-interval 100 \
     --num-episodes 16000 \
     --save-dir trained_models/imitation_learning/avd/ \
     --log-dir trained_models/imitation_learning/avd/ \
     --agent-start-action-prob 0.0 \
     --agent-end-action-prob 0.3 \
     --agent-action-prob-schedule 100 \
     --agent-action-prob-factor 0.1 \
     --use-inflection-weighting True
```
Different environments correspond to different oracles:
- `oracle-random`: set `env-name` to `avd-pose-random-oracle-v0`
- `oracle-landmarks`: set `env-name` to `avd-pose-landmarks-oracle-v0`
- `oracle-objects`: set `env-name` to `avd-pose-objects-oracle-v0`

In practice, we find that `oracle-landmarks` performs well across most metrics.

### Coverage + Novelty exploration training
For training the coverage and novelty agents, we use the `train_exploration.py` script. For example, the command to train an area-coverage agent:

```
cd $EXPLORING_EXPLORATION
mkdir -p trained_models/area_coverage/avd

python -W ignore -u train_exploration.py \
     --lr 3e-5 \
     --seed 123 \
     --num-processes 32 \
     --num-steps 200 \
     --num-rl-steps 50 \
     --env-name avd-pose-landmarks-oracle-v0 \
     --save-interval 100 \
     --eval-interval 100 \
     --num-episodes 64000 \
     --save-dir trained_models/area_coverage/avd \
     --log-dir trained_models/area_coverage/avd \
     --pretrained-il-model '' \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 4 \
     --area-reward-scale 0.3 \
     --smooth-coverage-reward-scale 0.0 \
     --novelty-reward-scale 0.0
```
Smooth coverage and novelty agents can be trained by setting the corresponding reward coefficients to a non-zero value and zeroing out the rest.

### Curiosity-based exploration training

For training the curiosity agent, we use the `train_curiosity_exploration.py` script:

```
cd $EXPLORING_EXPLORATION
mkdir -p trained_models/curiosity/avd

python -W ignore -u train_curiosity_exploration.py \
     --lr 1e-4 \
     --seed 123 \
     --num-processes 32 \
     --num-steps 200 \
     --num-rl-steps 50 \
     --save-interval 100 \
     --eval-interval 100 \
     --num-episodes 64000 \
     --env-name avd-pose-landmarks-oracle-v0 \
     --save-dir trained_models/curiosity/avd \
     --log-dir trained_models/curiosity/avd \
     --pretrained-il-model '' \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 4 \
     --reward-scale 1e-3 \
     --icm-embedding-type policy-lstm \
     --normalize-icm-rewards True
```

### Reconstruction-based exploration training

For training the reconstruction agent, there are two phases. The first phase is a pre-training of the reconstruction task-head.

```
cd $EXPLORING_EXPLORATION
mkdir -p trained_models/reconstruction/avd/pretraining

python -W ignore pretrain_reconstruction.py \
     --lr 1e-4 \
     --seed 123 \
     --num-processes 16 \
     --num-steps 200 \
     --num-rl-steps 200 \
     --env-name avd-recon-v0 \
     --save-interval 100 \
     --eval-interval 100 \
     --num-episodes 64000 \
     --save-dir trained_models/reconstruction/avd/pretraining \
     --log-dir trained_models/reconstruction/avd/pretraining \
     --num-pose-refs 50 \
     --clusters-path reconstruction_data_generation/avd/imagenet_clusters/clusters_00030_data.h5 \
     --rec-loss-fn-J 3 \
     --n-transformer-layers 2
```

For convenience, we provide a pre-trained reconstruction task-head in `$EXPLORING_EXPLORATION/pretrained_models/avd/pretrained_reconstruction/ckpt.pth`. The second phase is the training of the exploration policy. The exploration agent is rewarded for maximizing the reconstruction performance. Note, we keep the pre-trained reconstruction task-head frozen for this stage.

```
cd $EXPLORING_EXPLORATION
mkdir -p trained_models/reconstruction/avd/exploration_policy

python -W ignore train_reconstruction_exploration.py \
     --lr 3e-5 \
     --seed 123 \
     --num-processes 32 \
     --num-steps 200 \
     --num-rl-steps 50 \
     --env-name avd-recon-v0 \
     --save-interval 100 \
     --eval-interval 5 \
     --num-episodes 64000 \
     --save-dir trained_models/reconstruction/avd/exploration_policy \
     --log-dir trained_models/reconstruction/avd/exploration_policy \
     --load-path-rec pretrained_models/avd/pretrained_reconstruction/ckpt.pth \
     --pretrained-il-model '' \
     --num-pose-refs 50 \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 4 \
     --rec-reward-scale 1e-1 \
     --clusters-path reconstruction_data_generation/avd/imagenet_clusters/clusters_00030_data.h5 \
     --n-transformer-layers 2 \
     --rec-reward-interval 1
```

## Training on Habitat simulator

### Imitation learning pre-training
Command to train an exploration policy by imitating an oracle shortest-path follower:

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
mkdir -p trained_models/imitation_learning/mp3d

python -W ignore -u pretrain_imitation.py  \
     --lr 1e-4 \
     --seed 123 \
     --num-processes 8 \
     --num-steps 500 \
     --num-rl-steps 100 \
     --env-name habitat \
     --save-interval 200 \
     --eval-interval 200 \
     --save-unique True \
     --num-episodes 16000 \
     --save-dir trained_models/imitation_learning/mp3d/ \
     --log-dir trained_models/imitation_learning/mp3d/ \
     --habitat-config-file configs/pretrain_imitation/ppo_pose_train_random_oracle.yaml \
     --eval-habitat-config-file configs/exploration/ppo_pose_val.yaml \
     --agent-start-action-prob 0.0 \
     --agent-end-action-prob 0.5 \
     --agent-action-prob-schedule 1000 \
     --agent-action-prob-factor 0.1 \
     --agent-action-duration 1 \
     --use-inflection-weighting True
```

Different oracles can be selected by varying the `ORACLE_TYPE` variable in the configuration file. The following config files correspond to different oracles:  
- `oracle-random`: `configs/pretrain_imitation/ppo_pose_train_random_oracle.yaml`   
- `oracle-landmarks`: `configs/pretrain_imitation/ppo_pose_train_landmarks_oracle.yaml`   
- `oracle-objects`: `configs/pretrain_imitation/ppo_pose_train_objects_oracle.yaml`

### Coverage + Novelty exploration training
For training the coverage and novelty agents, we use the `train_exploration.py` script. For example, the command to train an area-coverage agent:

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
mkdir -p trained_models/area_coverage/mp3d

python -W ignore -u train_exploration.py \
     --lr 1e-1 \
     --seed 123 \
     --num-processes 8 \
     --num-steps 500 \
     --num-rl-steps 100 \
     --env-name habitat \
     --save-interval 200 \
     --eval-interval 200 \
     --num-episodes 16000 \
     --save-unique True \
     --save-dir trained_models/area_coverage/mp3d \
     --log-dir trained_models/area_coverage/mp3d \
     --habitat-config-file configs/exploration/ppo_pose_train.yaml \
     --eval-habitat-config-file configs/exploration/ppo_pose_val.yaml \
     --pretrained-il-model '' \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 2 \
     --area-reward-scale 1e-3 \
     --smooth-coverage-reward-scale 0.0 \
     --novelty-reward-scale 0.0
```
Smooth coverage and novelty agents can be trained by setting the corresponding reward coefficients to a non-zero value and zeroing out the rest. 

### Curiosity-based exploration training
For training the curiosity agent, we use the `train_curiosity_exploration.py` script:

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
mkdir -p trained_models/curiosity/mp3d

python -W ignore train_curiosity_exploration.py \
     --lr 1e-4 \
     --seed 123 \
     --num-processes 8 \
     --num-steps 500 \
     --num-rl-steps 100 \
     --env-name habitat \
     --save-interval 200 \
     --eval-interval 200 \
     --num-episodes 16000 \
     --save-unique True \
     --save-dir trained_models/curiosity/mp3d \
     --log-dir trained_models/curiosity/mp3d \
     --habitat-config-file configs/exploration/ppo_pose_train.yaml \
     --eval-habitat-config-file configs/exploration/ppo_pose_val.yaml \
     --pretrained-il-model '' \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 2 \
     --reward-scale 1e-3 \
     --icm-embedding-type policy-lstm \
     --normalize-icm-rewards True
```

### Reconstruction-based exploration training

For training the reconstruction agent, there are two phases. The first phase is a pre-training of the reconstruction task-head. 

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
mkdir -p trained_models/reconstruction/mp3d/pretraining

python pretrain_reconstruction.py \
     --lr 3e-5 \
     --seed 123 \
     --num-processes 8 \
     --num-steps 500 \
     --num-rl-steps 500 \
     --env-name habitat \
     --save-interval 200 \
     --eval-interval 200 \
     --save-unique True \
     --num-episodes 16000 \
     --save-dir trained_models/reconstruction/mp3d/pretraining \
     --log-dir trained_models/reconstruction/mp3d/pretraining \
     --habitat-config-file configs/pretrain_reconstruction/ppo_pose_train.yaml \
     --eval-habitat-config-file configs/pretrain_reconstruction/ppo_pose_val.yaml \
     --num-pose-refs 100 \
     --clusters-path reconstruction_data_generation/mp3d/imagenet_clusters/clusters_00050_data.h5 \
     --rec-loss-fn-K 3 \
     --n-transformer-layers 2 \
     --use-multi-gpu True
```

For convenience, we provide a pre-trained reconstruction task-head in `$EXPLORING_EXPLORATION/pretrained_models/mp3d/pretrained_reconstruction/ckpt.pth`. The second phase is the training of the exploration policy. The exploration agent is rewarded for maximizing the reconstruction performance given a frozen, pre-trained reconstruction task-head.

```
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

cd $EXPLORING_EXPLORATION
mkdir -p trained_models/reconstruction/mp3d/exploration_policy

python train_reconstruction_exploration.py \
     --lr 1e-5 \
     --seed 123 \
     --num-processes 8 \
     --num-steps 500 \
     --num-rl-steps 100 \
     --env-name habitat \
     --habitat-config-file configs/reconstruction_exploration/ppo_pose_train.yaml \
     --eval-habitat-config-file configs/reconstruction_exploration/ppo_pose_val.yaml \
     --save-interval 200 \
     --eval-interval 200 \
     --save-unique True \
     --num-episodes 16000 \
     --save-dir trained_models/reconstruction/mp3d/exploration_policy \
     --log-dir trained_models/reconstruction/mp3d/exploration_policy \
     --load-path-rec pretrained_models/mp3d/pretrained_reconsturction/ckpt.pth \
     --pretrained-il-model '' \
     --num-pose-refs 100 \
     --use-gae \
     --ppo-epoch 4 \
     --num-mini-batch 2 \
     --rec-reward-scale 1.0 \
     --clusters-path reconstruction_data_generation/mp3d/imagenet_clusters/clusters_00050_data.h5 \
     --n-transformer-layers 2 \
     --rec-reward-interval 5
```


## Data generation for reconstruction task
For training reconstruction-based exploration agents and evaluating on the reconstruction task, we need to first extract image clusters that represent concepts. 

### Concepts generation for AVD
1. Uniformly sample images from all environments.

	```
	cd $EXPLORING_EXPLORATION/reconstruction_data_generation/avd
	python -W ignore gather_uniform_points.py
	```
2. Extract imagenet features for the images, cluster them and save cluster statistics.

	```
	python generate_imagenet_clusters.py \
		--dataset-root avd/uniform_samples \
		--num-clusters 30 \
		--save-dir avd/imagenet_clusters
	```

### Concepts generation for MP3D
1. Uniformly sample images from all environments.
	
	```
	cd $EXPLORING_EXPLORATION/reconstruction_data_generation/mp3d
	chmod +x extract_data_script.sh && ./extract_data_script.sh
	```
2. Extract imagenet features for the images, cluster them and save cluster statistics.
	
	```
	python generate_imagenet_clusters.py \
		--image-size 84 \
		--dataset-root mp3d/uniform_samples \
		--num-clusters 50 \
		--save-dir mp3d/imagenet_clusters
	```


### Note
- The clusters can be visualized as follows:

	```
	# For AVD
	cd $EXPLORING_EXPLORATION/reconstruction_data_generation/avd/imagenet_clusters
	tensorboard --logdir=.
	
	# For MP3D
	cd $EXPLORING_EXPLORATION/reconstruction_data_generation/mp3d/imagenet_clusters
	tensorboard --logdir=.
	```

- To ensure reproducibility, we have provided the clusters we generated [here](https://dl.fbaipublicfiles.com/exploring-exploration/reconstruction_data.tar.gz). Copying them to `reconstruction_data_generation/avd/imagenet_clusters/clusters_00030_data.h5` and `reconstruction_data_generation/mp3d/imagenet_clusters/clusters_00030_data.h5` will ensure that the script re-uses the same clusters for generating the visualizations.
