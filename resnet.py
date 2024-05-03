import os
import torch
import torch.nn as nn
import pandas as pd
import torchvision
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from collections import defaultdict
from attack_method import *
from torchmetrics import Accuracy
from trades import trades_loss
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything

seed_everything(123, workers=True)

class LResnet(L.LightningModule):
    def __init__(self, name, loss_module,
                    num_target_classes: int = 10, 
                    adv_train_method: Attack = None, 
                    eps: float = 8/255,
                    iters: int = 20,
                    alpha: float = 2/255,
                    embedding: int = 5,
                    pair_training: bool = False,
                    enable_adv_pair_training: bool = True,
                    gamma: float = 0.01,
                    use_trades_loss: bool = False,
                    beta: float = 1.0,
                    adv_test_method: Attack = None,
                    results_file: str = "./results.csv"
                    ):
        super().__init__()
        assert not pair_training or not use_trades_loss, "Can not set pair training and use TRADES at the same time!"
        # Name
        self.name = name
        # Set loss module
        self.loss_module = loss_module
        
        self.num_target_classes = num_target_classes
        # Accuracy metric for training logs and testing evaluation
        self.accuracy1 = Accuracy(task="multiclass", num_classes=self.num_target_classes, top_k=1)
        self.accuracy3 = Accuracy(task="multiclass", num_classes=self.num_target_classes, top_k=3)
        self.accuracy5 = Accuracy(task="multiclass", num_classes=self.num_target_classes, top_k=5)
        # Adversarial generation method for training
        self.adv_train_method = adv_train_method
        # Whether use pair training method
        self.pair_training = pair_training
        self.enable_adv_pair_training = enable_adv_pair_training
        self.gamma = gamma
        # Whether use TRADES loss 
        self.use_trades_loss = use_trades_loss
        self.beta = beta
        if self.use_trades_loss:
            self.criterion_kl = nn.KLDivLoss(size_average=False)
        # Attack method used in testing
        self.adv_test_method = adv_test_method
        # Filename to save the results in testing
        self.results_file = results_file
        # Load pretrained model weights
        self.model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        # Change final layer from 1000 (ImageNet) classes to 10 (CIFAR-10) classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_target_classes)
        # Save embed to a dictionary
        self.embedding = embedding
        self.embed = {
            1: self.embed1, # 0.05
            2: self.embed2, # 0.05
            3: self.embed3, # 0.05
            4: self.embed4, # 0.05
            5: self.embed5, 
            6: self.forward
        }
        self.save_hyperparameters()

    def forward(self, imgs):
        return self.model(imgs)
    
    def embed5(self, imgs):
        out = self.model.conv1(imgs)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = self.model.avgpool(out)

        return out

    def embed1(self, imgs):
        out = self.model.conv1(imgs)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)

        return out

    def embed2(self, imgs):
        out = self.model.conv1(imgs)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)

        return out

    def embed3(self, imgs):
        out = self.model.conv1(imgs)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)

        return out
    
    def embed4(self, imgs):
        out = self.model.conv1(imgs)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)

        return out

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.use_trades_loss:
            imgs_natural = imgs.clone()
            imgs = imgs.detach() + 0.001 * torch.randn(imgs.shape).cuda().detach()
        if self.adv_train_method is not None:
            opt = self.optimizers()
            opt.zero_grad()
            # Change the images to adversarial examples
            att_imgs = self.adv_train_method(self.model, imgs, labels)
            # adv_train_method sets the model to eval
            self.model.train()
            # Reset accumulated gradients from adversarial generation
            opt.zero_grad()
            outputs = self.model(att_imgs)
        # Once we have the correct training images,
        # we can use the usual Lightning forward pass
        else:
            outputs = self.model(imgs)
        
        if not self.enable_adv_pair_training:
            outputs = self.model(imgs)

        if self.use_trades_loss:
            
            outputs_natural = self.model(imgs_natural)
            loss_natural = self.loss_module(outputs_natural, labels)
            loss_robust =  (1.0 / imgs.shape[0]) * self.criterion_kl(F.log_softmax(outputs, dim=1),
                                                    F.softmax(outputs_natural, dim=1))
            loss = loss_natural + self.beta * loss_robust
        else:
            loss = self.loss_module(outputs, labels)
        if self.adv_train_method is not None and self.pair_training:
            imgs_emb, att_imgs_emb = self.embed[self.embedding](imgs), self.embed[self.embedding](att_imgs)
            # ct_loss = 1 / imgs.shape[0] * torch.sum((att_imgs_emb - imgs_emb) ** 2)
            ct_loss = torch.mean((att_imgs_emb - imgs_emb) ** 2)
            loss = loss + ct_loss * self.gamma
        
        acc = self.accuracy1(outputs, labels)
        # Log accuracy and loss per-batch for Tensorboard
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = self.loss_module(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        # No need to return to call backward() on the loss
    
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_output_list = defaultdict(list)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.adv_test_method is not None:
            att_imgs = self.adv_test_method(self.model, imgs, labels)
        else:
            att_imgs = imgs
        outputs = self.model(att_imgs)
        acc1 = self.accuracy1(outputs, labels)
        acc3 = self.accuracy3(outputs, labels)
        acc5 = self.accuracy5(outputs, labels)
        self.log("test_acc_1", acc1, prog_bar=True)
        self.log("test_acc_3", acc3, prog_bar=True)
        self.log("test_acc_5", acc5, prog_bar=True)
        self.test_output_list["test_acc_1"].append(acc1)
        self.test_output_list["test_acc_3"].append(acc3)
        self.test_output_list["test_acc_5"].append(acc5)

    def on_test_epoch_end(self):
        outputs = self.test_output_list
        test_acc_1 = torch.stack(self.test_output_list["test_acc_1"]).mean().cpu().numpy()
        test_acc_3 = torch.stack(self.test_output_list["test_acc_3"]).mean().cpu().numpy()
        test_acc_5 = torch.stack(self.test_output_list["test_acc_5"]).mean().cpu().numpy()
        df = pd.DataFrame({
            "model": [self.name],
            "attack": [self.adv_test_method.name],
            "top_1_accuracy": [test_acc_1],
            "top_3_accuracy": [test_acc_3],
            "top_5_accuracy": [test_acc_5]
        })
        if os.path.exists(self.results_file):
            df_prev = pd.read_csv(self.results_file)
            df = pd.concat([df_prev, df], axis=0).reset_index(drop=True)
        df.to_csv(self.results_file, index=False)
        
if __name__ == "__main__":
    cli = LightningCLI(LResnet)
