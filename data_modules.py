import torch
import numpy as np
import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms.v2 as transforms
# Pretrained normalization based on https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
means, stds = [0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]
means, stds = np.array(means), np.array(stds)

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToImage(), # Converts to tensor
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=means, std=stds)
            ]
            )

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar_full = CIFAR10(root=self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [40000, 10000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar_test = CIFAR10(root=self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.cifar_predict = CIFAR10(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar_predict, batch_size=self.batch_size)

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToImage(), # Converts to tensor
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=means, std=stds)
            ]
            )

    def prepare_data(self):
        # download
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)