from zipfile import ZipFile

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch

import numpy as np
from PIL import Image
from io import BytesIO
import random

from sklearn.utils import shuffle

from utils import *


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_data, transform=None):
        self.data, self.nyu_dataset = data, nyu2_data
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]])).resize((640, 480))
        depth = Image.open(BytesIO(self.data[sample[1]])).resize((320, 240))
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


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
    def __init__(self, is_test=False, minDepth=0.1, maxDepth=10.0):
        self.is_test = is_test
        self.minDepth = minDepth
        self.maxDepth = maxDepth

    def __call__(self, sample):
        tf_toTensor = transforms.ToTensor()
        image, depth = sample["image"], sample["depth"]

        image = np.clip(np.array(image) / 255, 0, 1)
        image = tf_toTensor(image)

        depth = np.clip(np.array(depth) / 256 * self.maxDepth,
                        self.minDepth, self.maxDepth)

        depth = tf_toTensor(depth)

        return {"image": image, "depth": depth}


def getNoTransform(is_test=False, minDepth=0.1, maxDepth=10.0):
    return transforms.Compose([
        ToTensor(is_test=is_test, minDepth=minDepth, maxDepth=maxDepth)])


def getDefaultTrainTransform(minDepth=0.1, maxDepth=10.0):
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(),
        ToTensor(minDepth=minDepth, maxDepth=maxDepth)])


def nyu_DataLoader(path, batch_size, test=False, minDepth=0.1, maxDepth=10.0):
    ### Load Data from ZipFile
    input_zip = ZipFile(path)
    data = {path: input_zip.read(path) for path in input_zip.namelist()}

    nyu2_train = list((row.split(",") for row in (
        data["data/nyu2_train.csv"]).decode("utf-8").split("\n") if len(row) > 0))
    nyu2_test = list((row.split(",") for row in (
        data["data/nyu2_test.csv"]).decode("utf-8").split("\n") if len(row) > 0))

    nyu2_train = shuffle(nyu2_train, random_state=42)
    nyu2_val = nyu2_test[-4:]
    nyu2_test = nyu2_test

    ### Test Dataset
    if test == True:
        nyu2_train = nyu2_train[:batch_size*50]

    print(f"Total Train Data: {len(nyu2_train)}")
    print(f"Total Validation Data: {len(nyu2_val)}")

    ### Making DataLoader
    TrainLoader = depthDatasetMemory(
        data=data, nyu2_data=nyu2_train,
        transform=getDefaultTrainTransform(minDepth=minDepth, maxDepth=maxDepth))
    TrainLoader = DataLoader(
        dataset=TrainLoader, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    ValidationLoader = depthDatasetMemory(
        data=data, nyu2_data=nyu2_val,
        transform=getNoTransform(minDepth=minDepth, maxDepth=maxDepth))
    ValidationLoader = DataLoader(
        dataset=ValidationLoader, batch_size=4, shuffle=False, pin_memory=True, num_workers=8, drop_last=True)

    TestLoader = depthDatasetMemory(
        data=data, nyu2_data=nyu2_test,
        transform=getNoTransform(minDepth=minDepth, maxDepth=maxDepth))
    TestLoader = DataLoader(
        dataset=TestLoader, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=True)

    return TrainLoader, ValidationLoader, TestLoader
