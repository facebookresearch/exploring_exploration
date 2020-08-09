import os
import pdb
import json
import gzip
import habitat
import progressbar

IGNORE_CLASSES = [
    "floor",
    "wall",
    "door",
    "misc",
    "ceiling",
    "void",
    "stairs",
    "railing",
    "column",
    "beam",
    "",
    "board_panel",
]


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


safe_mkdir("data/object_annotations")
safe_mkdir("data/object_annotations/mp3d")
save_dir = "data/object_annotations/mp3d"

object_counts = {}
for split in ["val", "test", "train"]:
    config_path = (
        f"data_generation_scripts/configs/semantic_objects/mp3d_objects_{split}.yaml"
    )
    config = habitat.get_config_pose(config_path)

    env = habitat.Env(config=config)
    env.seed(1234)

    num_episodes = len(env._dataset.episodes)

    for epcount in progressbar.progressbar(range(num_episodes)):
        obs = env.reset()
        semantic_scene = env._sim._sim.semantic_scene
        semantic_objects = [
            {
                "center": obj.aabb.center.tolist(),
                "sizes": obj.aabb.sizes.tolist(),
                "id": obj.id,
                "category_name": obj.category.name(),
                "category_idx": obj.category.index(),
            }
            for obj in semantic_scene.objects
            if obj.category.name() not in IGNORE_CLASSES
        ]
        scene_id = env._current_episode.scene_id.split("/")[-1]
        save_path = os.path.join(save_dir, scene_id + ".json.gz")
        with gzip.open(save_path, "wt") as fp:
            json.dump(semantic_objects, fp)

    env.close()
