#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import h5py
import random
import argparse
import numpy as np
import subprocess as sp

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from exploring_exploration.models.reconstruction import FeatureNetwork


class RGBDataset(Dataset):
    def __init__(
        self, dataset_root, seed=123, transform=None, image_size=256, truncate_count=-1,
    ):
        random.seed(seed)
        self.dataset_root = dataset_root
        images = (
            sp.check_output(f"ls {dataset_root}", shell=True)
            .decode("utf-8")
            .split("\n")[:-1]
        )
        ndata = len(images)
        if truncate_count > 0:
            ndata = min(ndata, truncate_count)

        self.image_size = image_size

        self.dataset = [os.path.join(dataset_root, image) for image in images]

        random.shuffle(self.dataset)
        self.dataset = self.dataset[:ndata]

        # Data transform
        self.transform = transform if transform is not None else lambda x: x

        self.nimgs = ndata

    def __getitem__(self, index):
        path = self.dataset[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {"rgb": img}, {"rgb": path}

    def __len__(self):
        return self.nimgs


def main(args):
    # Enable cuda by default
    args.cuda = True

    # Define transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.Resize(args.image_size), transforms.ToTensor(), normalize]
    )

    # Create datasets
    datasets = {
        split: RGBDataset(
            os.path.join(args.dataset_root, split),
            seed=123,
            transform=transform,
            image_size=args.image_size,
            truncate_count=args.truncate_count,
        )
        for split in ["train", "val", "test"]
    }

    # Create data loaders
    data_loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create model
    net = FeatureNetwork()
    net.to(device)
    net.eval()

    # Generate image features for training images
    train_image_features = []
    train_image_paths = []

    for i, data in enumerate(data_loaders["train"], 0):

        # sample data
        inputs, input_paths = data
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features
        with torch.no_grad():
            feats = net(inputs["rgb"])  # (bs, 512)
        feats = feats.detach().cpu().numpy()
        train_image_features.append(feats)
        train_image_paths += input_paths["rgb"]

    train_image_features = np.concatenate(train_image_features, axis=0)

    # Generate image features for testing images
    test_image_features = []
    test_image_paths = []

    for i, data in enumerate(data_loaders["test"], 0):

        # sample data
        inputs, input_paths = data
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features
        with torch.no_grad():
            feats = net(inputs["rgb"])  # (bs, 512)
        feats = feats.detach().cpu().numpy()
        test_image_features.append(feats)
        test_image_paths += input_paths["rgb"]

    test_image_features = np.concatenate(test_image_features, axis=0)  # (N, 512)

    # ================= Perform clustering ==================
    kmeans = MiniBatchKMeans(
        init="k-means++",
        n_clusters=args.num_clusters,
        batch_size=args.batch_size,
        n_init=10,
        max_no_improvement=20,
        verbose=0,
    )
    save_h5_path = os.path.join(
        args.save_dir, f"clusters_{args.num_clusters:05d}_data.h5"
    )
    if os.path.isfile(save_h5_path):
        print("========> Loading existing clusters!")
        h5file = h5py.File(os.path.join(save_h5_path), "r")
        train_cluster_centroids = np.array(h5file["cluster_centroids"])
        kmeans.cluster_centers_ = train_cluster_centroids
        train_cluster_assignments = kmeans.predict(train_image_features)  # (N, )
        h5file.close()
    else:
        kmeans.fit(train_image_features)
        train_cluster_assignments = kmeans.predict(train_image_features)  # (N, )
        train_cluster_centroids = np.copy(
            kmeans.cluster_centers_
        )  # (num_clusters, 512)

    # Create a dictionary of cluster -> images for visualization
    cluster2image = {}
    if args.visualize_clusters:
        log_dir = os.path.join(
            args.save_dir, f"train_clusters_#clusters{args.num_clusters:05d}"
        )
        tbwriter = SummaryWriter(log_dir=log_dir)

    for i in range(args.num_clusters):
        valid_idxes = np.where(train_cluster_assignments == i)[0]
        valid_image_paths = [train_image_paths[j] for j in valid_idxes]
        # Shuffle and pick only upto 100 images per cluster
        random.shuffle(valid_image_paths)
        # Read the valid images
        valid_images = []
        for path in valid_image_paths[:100]:
            img = cv2.resize(
                np.flip(cv2.imread(path), axis=2), (args.image_size, args.image_size),
            )
            valid_images.append(img)
        valid_images = (
            np.stack(valid_images, axis=0).astype(np.float32) / 255.0
        )  # (K, H, W, C)
        valid_images = torch.Tensor(valid_images).permute(0, 3, 1, 2).contiguous()
        cluster2image[i] = valid_images
        if args.visualize_clusters:
            # Write the train image clusters to tensorboard
            tbwriter.add_images(f"Cluster #{i:05d}", valid_images, 0)

    h5file = h5py.File(
        os.path.join(args.save_dir, f"clusters_{args.num_clusters:05d}_data.h5"), "a"
    )

    if "cluster_centroids" not in h5file.keys():
        h5file.create_dataset("cluster_centroids", data=train_cluster_centroids)
    for i in range(args.num_clusters):
        if f"cluster_{i}/images" not in h5file.keys():
            h5file.create_dataset(f"cluster_{i}/images", data=cluster2image[i])

    h5file.close()

    if args.visualize_clusters:
        # Dot product of test_image_features with train_cluster_centroids
        test_dot_centroids = np.matmul(
            test_image_features, train_cluster_centroids.T
        )  # (N, num_clusters)
        if args.normalize_embedding:
            test_dot_centroids = (test_dot_centroids + 1.0) / 2.0
        else:
            test_dot_centroids = F.softmax(
                torch.Tensor(test_dot_centroids), dim=1
            ).numpy()

        # Find the top-K matching centroids
        topk_matches = np.argpartition(test_dot_centroids, -5, axis=1)[:, -5:]  # (N, 5)

        # Write the test nearest neighbors to tensorboard
        tbwriter = SummaryWriter(
            log_dir=os.path.join(
                args.save_dir, f"test_neighbors_#clusters{args.num_clusters:05d}"
            )
        )
        for i in range(100):
            test_image_path = test_image_paths[i]
            test_image = cv2.resize(
                cv2.imread(test_image_path), (args.image_size, args.image_size)
            )
            test_image = np.flip(test_image, axis=2).astype(np.float32) / 255.0
            test_image = torch.Tensor(test_image).permute(2, 0, 1).contiguous()
            topk_clusters = topk_matches[i]
            # Pick some 4 images representative of a cluster
            topk_cluster_images = []
            for k in topk_clusters:
                imgs = cluster2image[k][:4]  # (4, C, H, W)
                if imgs.shape[0] == 0:
                    continue
                elif imgs.shape[0] != 4:
                    imgs_pad = torch.zeros(4 - imgs.shape[0], *imgs.shape[1:])
                    imgs = torch.cat([imgs, imgs_pad], dim=0)
                # Downsample by a factor of 2
                imgs = F.interpolate(
                    imgs, scale_factor=0.5, mode="bilinear"
                )  # (4, C, H/2, W/2)
                # Reshape to form a grid
                imgs = imgs.permute(1, 0, 2, 3)  # (C, 4, H/2, W/2)
                C, _, Hby2, Wby2 = imgs.shape
                imgs = (
                    imgs.view(C, 2, 2, Hby2, Wby2)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(C, Hby2 * 2, Wby2 * 2)
                )
                # Draw a red border
                imgs[0, :4, :] = 1.0
                imgs[1, :4, :] = 0.0
                imgs[2, :4, :] = 0.0
                imgs[0, -4:, :] = 1.0
                imgs[1, -4:, :] = 0.0
                imgs[2, -4:, :] = 0.0
                imgs[0, :, :4] = 1.0
                imgs[1, :, :4] = 0.0
                imgs[2, :, :4] = 0.0
                imgs[0, :, -4:] = 1.0
                imgs[1, :, -4:] = 0.0
                imgs[2, :, -4:] = 0.0
                topk_cluster_images.append(imgs)

            vis_img = torch.cat([test_image, *topk_cluster_images], dim=2)
            image_name = f"Test image #{i:04d}"
            for k in topk_clusters:
                score = test_dot_centroids[i, k].item()
                image_name += f"_{score:.3f}"
            tbwriter.add_image(image_name, vis_img, 0)


def str2bool(v):
    return True if v.lower() in ["yes", "y", "true", "t"] else False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--truncate-count", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-clusters", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="visualization_dir")
    parser.add_argument("--visualize-clusters", type=str2bool, default=True)
    parser.add_argument("--normalize-embedding", type=str2bool, default=True)

    args = parser.parse_args()

    main(args)
