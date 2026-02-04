import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from random import randint
import random


def euclidean_metric(x, y):
    n = x.size(0)
    m = y.size(0)
    x = x.unsqueeze(1).expand(n, m, -1)
    y = y.unsqueeze(0).expand(n, m, -1)
    distance = -((x-y)**2).sum(dim=2)
    return distance/16



