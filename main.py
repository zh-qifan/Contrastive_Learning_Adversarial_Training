# import numpy as np
# import pandas as pd
# import argparse

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms.v2 as transforms
import torch
import lightning as L
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchmetrics import Accuracy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI
from resnet import LResnet
from attack_method import *
from data_modules import *
from lightning.pytorch import seed_everything

# Experiment Seed
# 42
# 123
seed_everything(123, workers=True)

if __name__ == "__main__":
    cli = LightningCLI(seed_everything_default=123)