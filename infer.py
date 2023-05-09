import numpy as np
import cv2

import torch
from torch import nn
from torchvision import transforms

from .networks.model import DenseDepth

class InferDenseDepth:
    def __init__(self, path="/home/model_disk/DenseDepth/0506-1944/043-mobilevitv2-2.1103084578107505.pt")