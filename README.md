# V-MCR2
An unofficial implementation of V-MCR${}^2$ in the paper "[Efficient Maximal Coding Rate Reduction by Variational Forms](https://arxiv.org/abs/2204.00077v1)".

## Introduction
MCR${}^2$, short for "Maximal Coding Rate Reduction", was first proposed by Yu et al in the paper "[Learning diverse and discriminative representations via the principle of maximal coding rate reduction](https://arxiv.org/abs/2006.08558v1)", seeking the linear representations of features of high-dimensional data. V-MCR${}^2$ is its variational form, which reduces the computational complexity, and from what the authors say,  V-MCR${}^2$ improves the accuracy achieved by MCR${}^2$.

## Technical Details
Though the algorithm of  V-MCR${}^2$ was proposed in paper, there was no official implementation. What's more important is that, although the paper says that the training is proceeded iteratively by batch (*batch_size*=1000 for mnist and cifar-10, and 2000 for cifar-100 and tiny-imagenet), the algorithm seems to use the whole training data at once in one epoch, to generate its features, compute and update its dictionary, and update the gradient of the model being trained. And, the dictionary is initialized (which is also called 'latching') at the beginning and at every 50 epochs, from the whole training data and labels. So where is the '*batch_size*'? Based on this point, I cooperate the algorithm and my own understanding to write down the code of my-versioned V-MCR${}^2$.

In latching, the dictionary $\Gamma$ and its sparse coding $A$ are initialized from the whole training data and labels. However, when computing the features, the matrix $Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$ is divided by the number of samples in class $j$. Therefore the nuclear norm is forced to be 1, namely $\|Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}\|_{*}=1$.

Then at each epoch in the training stage, I compute $Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$ in each batch and force its nuclear norm to be 1, then update $\Gamma$, $A$, and $\theta$ (where $\theta$ denotes the trainable weights of model), just as what the V-MCR${}^2$ algorithm tells in paper. 

This operation is important, because we wish $\Gamma Diag(A_j)\Gamma^T = Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$. However these two parts are computed at different stage and different scale (computed using the whole training data or computed using a batch of the training data). If the scale is not regularized, empirically the training would fail.

However the update of $A$ might break the constraint, because the original algorithm doesn't regularize its scale, unlike what $\Gamma$ is treated. The ideal way to regularize the scale of $A$ is to set $\|A_j\|_1=1$, but in practice it results in slow convergence. Using $\|A_j\|_2=1$ or even $\|A_j\|_3=1$ instead would unexpectedly have a faster convergence, and finally get a quite good accuracy.

Besides, I change the value of $\epsilon^2$ dynamically during the training stage to speed up the convergence.

## How to Start Training
Here are some examples.
```bash
python train.py --batch_size 1000 --epoch 2000 --latch_epoch 50 --dataset 'mnist' --out_dim 128 --num_comps 20 --learning_rate 1e-3 --metric_mcr2

python train.py --batch_size 1000 --epoch 2000 --latch_epoch 50 --dataset 'cifar10' --out_dim 128 --num_comps 20 --learning_rate 1e-3 --metric_mcr2

python train.py --batch_size 2000 --epoch 2000 --latch_epoch 50 --dataset 'cifar100' --out_dim 500 --num_comps 10 --learning_rate 1e-3 --metric_mcr2
```

## How to Start Evaluation
Here is an example.
```bash
python evaluate_forward.py --path "./log/data[mnist]_fdim[128]_ncomps[20]_epoch[2000]_batch[1000]_lr[0.001]_seed[77]/checkpoints/model_epoch1999.pth" --batch_size 1000
```
