# Contrastive Learning in Embedding Space Improves Adversarial Robustness

## About The Project

This is a coursework project for CPSC 471/571 Trustworthy Deep Learning at Yale University, taught by Professor [Rex Ying](https://www.cs.yale.edu/homes/ying-rex/). The problem that this project investigates is whether a contrastive loss component in embedding space can improve adversarial robustness of deep learning models. Given the time constraint on this project, Weonly evaluate the loss component on image classification tasks. Robustness against
evasion attack usually indicates a smoother neural network which is insensitive to imperceptible
changes on inputs. One approach to achieve this goal, named TRADES [Zhang et al., 2019], imposes
a contrastive loss component in output space i.e.

$$f \in \min_f \mathbb{E}_{(\mathbf{x}_0, y_0) \sim \mathcal{D}} \left\{\mathcal{L}(f(\mathbf{x}_0),y)+\beta\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}\mathcal{L}(f(\mathbf{x}),f(\mathbf{x}_0))\right\}.$$

Nevertheless, neural networks such as ResNet and Transformer tends to have diminishing rank on
image datasets [Feng et al., 2022], which indicates that the deeper linear mappings are degenerated.
Thus to achieve the robustness goal, clean inputs and its perturbed inputs are only constrained to be
close in quotient space instead of in embedding space. Thus it is natural to ask whether imposing
contrastive loss can further improve adversarial robustness. The goal of this project is to test whether
the following new loss function can improve adversarial robustness

$$    f \in \min_f \mathbb{E}_{(\mathbf{x}_0, y_0) \sim \mathcal{D}} \left\{\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}\mathcal{L}(f(\mathbf{x}),y)+\beta\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}d_{\text{emb}}(g(\mathbf{x}),g(\mathbf{x}_0))\right\},$$

where $g$ is the mapping from inputs to its embedding. Further discussion on the proposed method
can be found in Section 4. The base model in this project is the ResNet-18 [He et al., 2016]. The
attack methods are white-box untargeted attack such as FGSM [Goodfellow et al., 2014], PGD
[Madry et al., 2018] and Deepfool [Moosavi-Dezfooli et al., 2016]. The baseline methods are regular
training, adversarial training [Madry et al., 2018] and TRADES [Zhang et al., 2019].

## Getting Started

### Prerequisites

### Datasets

## Usage

## Roadmap

* [X] Set up training and testing pipeline using LightningCLI. 
* [X] Add FGSM, PGD and Deepfool attack methods.
* [X] Add regular training, adversarial traing, [TRADES](https://github.com/yaodongyu/TRADES) and contrastive learning training methods.
* [ ] Run and test baseline training.
* [ ] Run and test adversarial training.
* [ ] Run and test [TRADES](https://github.com/yaodongyu/TRADES).
* [ ] Run and test contrastive learning.

## Contact

Qifan Zhang - qifansz1008@gmail.com

If you have any questions and suggestions for this project, please post an issue or open a pull request. Thank you!

## Acknowledgement

CPSC 471/571 is wonderful introduction to the Trustworthy AI 