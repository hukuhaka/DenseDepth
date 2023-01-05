import os
import random

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch

import numpy as np
import pandas as pd
from PIL import Image
import random

from sklearn.utils import shuffle

from utils import *


class depthDatasetMemory(Dataset):
    def __init__(self, kitti_data, transform=None):
        self.kitti_dataset = kitti_data
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.kitti_dataset[idx]
        image = Image.open(sample[0]).resize((1280, 384))
        depth = Image.open(sample[1]).resize((1280//2, 384//2))
        
        # height = image.height
        # width = image.width
        
        # top_margin = int(height - 352)
        # left_margin = int((width - 1216) / 2)
        # depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352)).resize((1216//2, 352//2))
        # image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.kitti_dataset)


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]

        if random.random() < self.probability:
            image = F.hflip(image)
            depth = F.hflip(depth)

        return {"image": image, "depth": depth}


class RandomChannelSwap(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]

        if random.random() < self.probability:
            indices = torch.randperm(3)
            image = np.asarray(image)
            image = Image.fromarray(image[..., indices])

        return {"image": image, "depth": depth}


class ToTensor(object):
    def __init__(self, is_test=False, minDepth=1.0, maxDepth=80.0):
        self.is_test = is_test
        self.minDepth = minDepth
        self.maxDepth = maxDepth

    def __call__(self, sample):
        tf_toTensor = transforms.ToTensor()

        image, depth = sample["image"], sample["depth"]
        # num = (np.random.randint(0, image.width-704) // 2) * 2

        image = np.clip(np.array(image) / 255, 0, 1)
        image = tf_toTensor(image)
        image = image[:, :, : 640]

        depth = np.clip(np.array(depth) / 256 * self.maxDepth,
                        self.minDepth, self.maxDepth)

        depth = tf_toTensor(depth)
        depth = depth[:, :, : 640//2]

        return {"image": image, "depth": depth}


def getNoTransform(is_test=False, minDepth=1.0, maxDepth=80.0):
    return transforms.Compose([
        ToTensor(is_test=is_test, minDepth=minDepth, maxDepth=maxDepth)])


def getDefaultTrainTransform(minDepth=1.0, maxDepth=80.0):
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(),
        ToTensor(minDepth=minDepth, maxDepth=maxDepth)])


def kitti_DataLoader(csvpath, datapath, batch_size, test=False, minDepth=1.0, maxDepth=8000.0):
    csv = pd.read_csv(csvpath, names=["iter", "raw", "depth"])

    csv["raw"] = csv["raw"].apply(lambda x: os.path.join(datapath, x))
    csv["depth"] = csv["depth"].apply(lambda x: os.path.join(datapath, x))

    csv_train = shuffle(csv, random_state=42)
    csv_test = csv_train[-500:]
    csv_train = list(csv_train[["raw", "depth"]].values.tolist())
    csv_test = list(csv_test[["raw", "depth"]].values.tolist())

    kitti_train = list(csv_train)
    kitti_val = list(csv_train[-4:])
    kitti_test = list(csv_test)

    if test == True:
        kitti_train = kitti_train[:batch_size*50]
        kitti_test = kitti_test[:batch_size*10]

    print(
        f"Train: {len(kitti_train)}, Validation: {len(kitti_val)}, Test: {len(kitti_test)}")

    ### Making DataLoader
    TrainLoader = depthDatasetMemory(
        kitti_data=kitti_train,
        transform=getDefaultTrainTransform(minDepth=minDepth, maxDepth=maxDepth))
    TrainLoader = DataLoader(
        dataset=TrainLoader, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=8, drop_last=True)

    ValidationLoader = depthDatasetMemory(
        kitti_data=kitti_val, transform=getNoTransform(minDepth=minDepth, maxDepth=maxDepth))
    ValidationLoader = DataLoader(
        dataset=ValidationLoader, batch_size=4, shuffle=False,
        pin_memory=True, num_workers=8, drop_last=True)

    TestLoader = depthDatasetMemory(
        kitti_data=kitti_test, transform=getNoTransform(minDepth=minDepth, maxDepth=maxDepth))
    TestLoader = DataLoader(
        dataset=TestLoader, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=8, drop_last=True)

    return TrainLoader, ValidationLoader, TestLoader
