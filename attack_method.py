"""
This scripts include all attack method used in this project. 
In this project, we only consider l_\infty attack for simplicity.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from abc import ABC, abstractmethod


class Attack(ABC):
    
    def __init__(self):
        self.name = ""
        pass
    
    @abstractmethod
    def __call__(self, model, imgs, labels):
        raise NotImplementedError

class idAttack(Attack):

    def __init__(self):
        self.name = "id"

    def __call__(self, model, imgs, labels):
        return imgs.detach()
        
class FGSMAttack(Attack):
    #NOTE: Specifying the class type is necessary since we use jsonparse
    def __init__(self, loss_module: _Loss, eps=8/255, attack_type="untargeted"):
        self.name = "fgsm"
        self.loss_module = loss_module
        self.eps = eps
        self.attack_type = attack_type
    
    @torch.inference_mode(False)
    def __call__(self, model, imgs, labels, clone=True):
        model.eval()
        #NOTE: In inference phase, the imgs and labels are inference tensor and can only be set required_grad = True
        #when inference_mode = True. But setting inference_mode = True will disable torch to calculate gradients.
        #The workaround is to clone them.
        if clone: 
            imgs = imgs.clone().detach()
            labels = labels.clone().detach()
        imgs.requires_grad = True
        out = model(imgs)
        
        if self.attack_type == "untargeted":
            loss = self.loss_module(out, labels)
            loss.backward()
            att_imgs = imgs + self.eps * torch.sign(imgs.grad)
        elif self.attack_type == "targeted":
            loss = self.loss_module(out, labels)
            loss.backward()
            att_imgs = imgs - self.eps * torch.sign(imgs.grad)

        return att_imgs.detach()

class PGDAttack(FGSMAttack):

    def __init__(self, loss_module: _Loss, alpha=8/255, iters=20, eps=2/255, attack_type="untargeted"):
        super().__init__(loss_module=loss_module, eps=eps, attack_type=attack_type)
        self.name = "pgd"
        self.iters = iters
        self.alpha = alpha

    @torch.inference_mode(False)
    def __call__(self, model, imgs, labels):
        model.eval()
        imgs = imgs.clone().detach()
        labels = labels.clone().detach()
        att_imgs = imgs
        min_bound, max_bound = imgs - self.alpha, imgs + self.alpha
        min_bound, max_bound = min_bound.detach(), max_bound.detach()
        for _ in range(self.iters):
            att_imgs = super().__call__(model=model, imgs=att_imgs, labels=labels, clone=False)
            att_imgs = torch.clamp(att_imgs, min=min_bound, max=max_bound)
        
        return att_imgs

class DeepfoolAttack(Attack):

    def __init__(self, max_iters=50, overshoot=0.02):
        self.name = "deepfool"
        self.max_iters = max_iters
        self.overshoot = overshoot

    @torch.inference_mode(False)
    def __call__(self, model, imgs, labels):
        model.eval()
        imgs = imgs.clone().detach()
        labels = labels.clone().detach()

        batch_size = len(imgs)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach()
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = imgs[idx : idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.max_iters):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                early_stop, pre, adv_image = self._forward_indiv(
                    model, adv_images[idx], labels[idx]
                )
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        att_imgs = torch.cat(adv_images).detach()

        return att_imgs

    @torch.inference_mode(False)
    def _forward_indiv(self, model, image, label):
        image.requires_grad = True
        fs = model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + self.overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    @torch.inference_mode(False)
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)

if __name__ == "__main__":

    model = torchvision.models.resnet18()

    test_imgs = torch.randn(1, 3, 32, 32)

    att = FGSMAttack(torch.nn.CrossEntropyLoss())

    att(model, test_imgs, torch.zeros((1,), dtype=torch.long))