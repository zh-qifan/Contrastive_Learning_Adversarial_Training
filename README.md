# Can Contrastive Learning in Embedding Space Improve Adversarial Robustness?

## About The Project

This is a coursework project for CPSC 471/571 Trustworthy Deep Learning at Yale University, taught by Professor [Rex Ying](https://www.cs.yale.edu/homes/ying-rex/). The problem that this project investigates is whether a contrastive loss component in embedding space can improve adversarial robustness of deep learning models. Given the time constraint on this project, Weonly evaluate the loss component on image classification tasks. Robustness against
evasion attack usually indicates a smoother neural network which is insensitive to imperceptible
changes on inputs. One approach to achieve this goal, named TRADES (Zhang et al., 2019), imposes
a contrastive loss component in output space i.e.

```math
f \in \min_f \mathbb{E}_{(\mathbf{x}_0, y_0) \sim \mathcal{D}} \left\{\mathcal{L}(f(\mathbf{x}_0),y)+\beta\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}\mathcal{L}(f(\mathbf{x}),f(\mathbf{x}_0))\right\}.
```

Nevertheless, neural networks such as ResNet and Transformer tends to have diminishing rank on
image datasets (Feng et al., 2022), which indicates that the deeper linear mappings are degenerated.
Thus to achieve the robustness goal, clean inputs and its perturbed inputs are only constrained to be
close in quotient space instead of in embedding space. Thus it is natural to ask whether imposing
contrastive loss can further improve adversarial robustness. The goal of this project is to test whether
the following new loss function can improve adversarial robustness

```math
f \in \min_f \mathbb{E}_{(\mathbf{x}_0, y_0) \sim \mathcal{D}} \left\{\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}\mathcal{L}(f(\mathbf{x}),y)+\beta\max_{d(\mathbf{x},\mathbf{x}_0)\leq \epsilon}d_{\text{emb}}(g(\mathbf{x}),g(\mathbf{x}_0))\right\}
```

where $g$ is the mapping from inputs to its embedding. Further discussion on the proposed method
can be found in Section 4. The base model in this project is the ResNet-18 (He et al., 2016). The
attack methods are white-box untargeted attack such as FGSM (Goodfellow et al., 2014), PGD
(Madry et al., 2018) and Deepfool (Moosavi-Dezfooli et al., 2016). The baseline methods are regular
training, adversarial training (Madry et al., 2018) and TRADES (Zhang et al., 2019).

## Getting Started

### Prerequisites

- Python (3.8.17)
- torch (2.2.2)
- torchvision (0.17.2)
- lightning (2.2.2)
- CUDA

The `environment.yml` is also provided. One can run 

```sh
conda env create -f ./environment.yml
```

to load the virtual environment used in this project.

### Datasets

The datasets used in this project are [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009) and [MNIST](http://yann.lecun.com/exdb/mnist/) (Lecun et al., 1998). These are all standard image classification datasets. 

## Experiments

One can directly run the following scripts to reproduce the experiment results. **After each experiment, please save the generated `model/` and `lightning_logs/` folder as well as the `results.csv` file manually, since at the beginning of each experiments, these files will be removed.** To run the experiment multiple times, you need to manually add argument `--seed_everything_default=<random seed>` at the end of each `python` command or change it in `main.py`.

### Results on CIFAR-10

```sh
sh scripts/experiments_cifar_10.sh
```

### Results on MNIST

```sh
sh scripts/experiments_mnist.sh
```

### Ablation study on embedding space

```sh
sh scripts/ablation_embedding_cifar_10.sh
```

### Ablation study on tradeoff parameter

```sh
sh scripts/ablation_gamma_cifar_10.sh
```

## Results

TBA

## Roadmap

* [X] Set up training and testing pipeline using LightningCLI. 
* [X] Add FGSM, PGD and Deepfool attack methods.
* [X] Add regular training, adversarial traing, [TRADES](https://github.com/yaodongyu/TRADES) and contrastive learning training methods.
* [X] Run and test baseline training on CIFAR-10.
* [X] Run and test adversarial training on CIFAR-10.
* [X] Run and test [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10.
* [X] Run and test contrastive learning on CIFAR-10.
* [X] Ablation study on embedding space on CIFAR-10 (6 layers in total). 
* [X] Ablation study on tradeoff parameter on CIFAR-10 (0.1, 0.5, 1, 3, 6, 7). 
* [ ] Run and test on MNIST.

## Contact

Qifan Zhang - qifansz1008@gmail.com

If you have any questions and suggestions for this project, please post an issue or open a pull request. Thank you!

## Acknowledgement

- CPSC 471/571 at Yale University is a wonderful introduction to the Trustworthy AI  and also offers me a totally different perspective to the AI research compared to my previous study. A big thanks to the Professor [Rex Ying](https://www.cs.yale.edu/homes/ying-rex/) and all the TAs. AI is not perfect in its explainability, robustness, fairness and efficiency and still needs time and researchers' efforts to help it improve and evolve to the AGI stage.
- [Lightning](https://lightning.ai/docs/pytorch/stable/) is a beautiful tool in training deep learning models and it greatly saves my time in engineering. [LightningCLI](https://lightning.ai/docs/pytorch/stable/api_references.html#cli) provides a perfect CLI to execute my experiments from a shell terminal. This makes the whole project cleaner and more concise. Thank you, Lightning Team!

## Reference

- Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning, 2019.
- Ruili Feng, Kecheng Zheng, Yukun Huang, Deli Zhao, Michael Jordan, and Zheng-Jun Zha. Rank diminishing in deep neural networks, 2022.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.
- Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.
- Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2574–2582, 2016.