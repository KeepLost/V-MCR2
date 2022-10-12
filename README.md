# V-MCR2
An unofficial implementation of V-MCR${}^2$ in the paper "[Efficient Maximal Coding Rate Reduction by Variational Forms](https://arxiv.org/abs/2204.00077v1)".

## Introduction
MCR${}^2$, short for "Maximal Coding Rate Reduction", was first proposed by Yu et al in the paper "[Learning diverse and discriminative representations via the principle of maximal coding rate reduction](https://arxiv.org/abs/2006.08558v1)", seeking the linear representations of features of high-dimensional data. V-MCR${}^2$ is its variational form, which reduces the computational complexity, and from what the authors say,  V-MCR${}^2$ slightly improves the accuracy achieved by MCR${}^2$.

## Some Technical Details
Though the algorithm of  V-MCR${}^2$ was proposed in the paper, there was no official implementation at present. What's more important is that, although the paper says that the training is proceeded iteratively by batch (*batch_size*=1000 for mnist and cifar10, and 2000 for cifar100 and tiny-imagenet), the algorithm seems to use the whole training data at once in one epoch, to generate its features, compute and update its dictionary, and update the gradient of the model being trained. And, the dictionary is initialized (which is also called 'latching') at the beginning and at every 50 epochs, using the whole training data and labels. So where is the usage of '*batch_size*'? Based on this point, I cooperate the algorithm and my own understanding to write down the code of my-versioned V-MCR${}^2$.

In latching, the dictionary $\Gamma$ and its sparse coding $A$ are initialized from the whole training data and labels. However, when computing the features, the matrix $Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$ is divided by the number of samples in class $j$. Therefore the nuclear norm is forced to be 1, namely $\|Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}\|_{*}=1$.

Then at each epoch in the training stage, I compute $Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$ in each batch and force its nuclear norm to be 1, then update $\Gamma$, $A$, and $\theta$ (where $\theta$ denotes the trainable weights of model), just as what the V-MCR${}^2$ algorithm is like in the paper. 

The operation of regularizing the nuclear norm is important, because we wish $\forall j, \Gamma Diag(A_j)\Gamma^T = Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}$. However these two parts are computed at different stages and different scales (computed using the whole training data or computed using a batch of the training data). If the scale is not regularized, empirically the training would fail.

However the update of the sparse coding $A$ might break the constraint, because the original algorithm doesn't regularize its scale, unlike what $\Gamma$ is treated. The ideal way to regularize $A$ is to set $\|A_j\|_1=1$, but in practice it results in slow convergence. Using $\|A_j\|_2=1$ or even $\|A_j\|_3=1$ instead would unexpectedly have a faster convergence, and finally get a quite good accuracy. I suspect that  $\|A_j\|_2=1$ or $\|A_j\|_3=1$ could still regularize $\|A_j\|_1$ but its value is not $1$. For example, given an arbitrary vector $v\in R^2$, if we force $\| v\|_2 =1$, then we definitely have $1\leq \| v\|_1\leq \sqrt{2}$. If we force $\| v\|_3=1$, then $1\leq \| v\|_1\leq 2^{\frac{2}{3}}$. So, it's resonable to guess that $\|A_j\|_1$ would be bounded by forcing $\|A_j\|_2=1$ or $\|A_j\|_3=1$. And by minimizing $\| \Gamma Diag(A_j)\Gamma^T - Z_{\theta}Diag(\Pi_j)Z_{\theta}^{T}\|_F^2$, if $\Gamma$ is fixed, then elements in $A_j$ that are large might become larger, forcing $\Gamma Diag(A_j)\Gamma^T$ to discover a structure with lower dimensions.

Surely there is another way: just change $\nu_A$ (and possibly $\nu_{\Gamma}$). I tried this way but found that the training is unstable and the best result is not as good as those from regularizing $p$-norm of $A$.

Besides, I change the value of $\epsilon^2$ dynamically during the training stage to speed up the convergence at the beginning of the training.

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

## Compared with the Original Paper
I ran my codes on mnist and cifar10 datasets, and found that the accuracies are a bit lower than those in *Table 2* of the original paper (mnist: 0.937, cifar10: 0.832). If the evaluation method is changed to SVM, then these two numbers could be 0.01~0.03 higher. I tried many ways but results are almost the same, sometimes even worse.
