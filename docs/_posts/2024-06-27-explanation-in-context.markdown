---
layout: post
title:  "An Explanation of In-Context Learning"
date:   2024-06-27 10:25:14 +0200
categories: jekyll update
mathjax: true
# usemathjax: true
---
{% include mathjax.html %}

In recent years, large language models (LLMs) and in particular transformers
[[1]](#1) have demonstrated astounding abilities in various fields of modern
machine learning. It has been hypothesized that their success is due to a
capability called *in-context* learning [[2]](#2): Without changing its
parameters, the LLM produces the correct output for a certain task solely based
on instructions or demonstrations contained in a prompt.

The mechanisms enabling in-context learning remain poorly understood to this
day. A possible explanation, which is actively being researched, is the ability
of transformers to implement algorithms that extract specific information from
the input. In particular, recent work [[3,4,5]](#3) has shown that certain
in-context learners can develop a context-dependent model within their hidden
activations that is trained on the examples provided in the prompt.

In this blog post, we revisit the work in [[3,4,5]](#3) and explain the
algorithms that can be learned by transformers to perform in-context learning.
While Oswald et al. and Ahn et al. [[3,4]](#3) focus on linear transformers
(i.e. the attention module contains no nonlinear activations) learning linear
functions, Cheng et al. [[5]](#3) consider the more general case of nonlinear
transformers learning certain nonlinear functions.

## Input Data for In-Context Learning

As a data model that makes in-context learning tractable, Oswald et al., Ahn et
al. and Cheng et al. [[3,4,5]](#3) consider the following random instances: The
prompt contains $n$ context tokens $e^{(i)} = (x^{(i)}, y^{(i)}) \in
\mathbb{R}^{d+1}$ and a query token $e^{(n+1)} = (x^{(n+1)}, 0) \in
\mathbb{R}^{d+1}$, where $[x^{(1)},\ldots, x^{(n+1)}]$ are convariates drawn
from a joint distribution $\mathcal{P}\_X$ and $[y^{(1)}, \ldots, y^{(n+1)}]$
are corresponding labels drawn from joint $\mathcal{P}\_{Y|X}$. Note,
$y^{(n+1)}$ is unobserved. In matrix notation, the problem is given by

$$
Z_0 =
\begin{bmatrix} z^{(1)} & z^{(2)} & \dots & z^{(n)} & z^{(n+1)} \\
\end{bmatrix} =
\begin{bmatrix} x^{(1)} & x^{(2)} & \dots & x^{(n)} & x^{(n+1)} \\
y^{(1)} & y^{(2)} & \dots & y^{(n)} & 0
\end{bmatrix} \in \mathbb{R}^{(d+1) \times (n+1)}.
$$

The zero is a placeholder for the unknown value $y^{(n+1)}$ and our goal is to
predict this label with a transformer model that is trained on this data
distribution. Therefore, we need to define a loss function that measures the
difference of the predicted label $y_{\text{pred}}^{(n+1)}$ and the true label
$y^{(n+1)}$.  The so-called *in-context loss* is the expected squared error:

$$
\begin{align}
\mathbb{E}_{Z_0, y^{(n+1)}}
\left[ \left( y_{\text{pred}}^{(n+1)} - y^{(n+1)} \right)^2 \right].
\end{align}
$$

Last, its worth mentioning that an important case of $\mathcal{P}\_{Y|X}$ is
when $y^{(i)} = \phi(x^{(i)})$ for some unknown function $\phi$. In particular,
Ahn et al. [[3]](#3) consider $\phi(x) = \langle x, \theta \rangle$ for some
$\theta \in \mathbb{R}^{d}$ which corresponds to a linear regression model.

## Simple Transformer Model

Having seen the input data setup, we now introduce the transformer architecture
that is used to theoretically analyze in-context learning. The central component
in a transformer model is the attention module [[1]](#1). Originally, an
attention layer was defined as a function that maps values $V \in
\mathbb{R}^{(d+1) \times (d+1)}$, keys $K \in \mathbb{R}^{(d+1) \times (d+1)}$
and querys $Q \in \mathbb{R}^{(d+1) \times (d+1)}$ to an output:

$$
\operatorname{Attn}_{Q,K,V}^{smax}(Z)
=VZ \cdot \operatorname{smax}\left(Z^\top K^\top Q Z\right).
$$


Cheng et al. [[5]](#5) generalize this definition to incorporate any matrix
valued function $\tilde{h} : \mathbb{R}^{d \times (n+1)} \times \mathbb{R}^{d
\times (n+1)} \to \mathbb{R}^{(n+1) \times (n+1)}$ instead of the softmax
funtion. Accordingly, the *generalized* attention module is defined as

$$
\operatorname{Attn}^{\tilde{h}}_{V, B, C}(Z) := V Z M \tilde{h}(BX, CX),
\qquad
M := \begin{bmatrix}
I_n & 0 \\
0 & 0
\end{bmatrix}
$$

where $V \in \mathbb{R}^{(d+1) \times (d+1)}$, $B \in \mathbb{R}^{d \times d}$,
$C \in \mathbb{R}^{d \times d}$ are the value, key, and query matrices. $M$ is a
mask that reflects the asymmetric structure of the input data $Z$ that results
from the missing label $y^{(n+1)}$ and $X = [x^{(1)}, \ldots, x^{(n+1)}] \in
\mathbb{R}^{d \times (n+1)}$ consists of the firs $d$ rows of $Z$.

Usually, a transformers module consists of multiple multi-head attention
modules, normalization and feed-forward layers [[1]](#1). However, for the
purpose of analyzing in-context learning, we consider a single-head
attention only $L$-layer transformer that is constructed by stacking $L$
attention modules. $Z_l$ denotes the output of the $(l-1)$-th layer and is
defined as

$$
Z_l
:= Z_{l-1} + \operatorname{Attn}^{\hat{h}_l}_{V_l, B_l, C_l}(Z_{l-1})
= Z_{l-1} + V_l Z_{l-1} M_l \tilde{h}_l(B_l X_{l-1}, C_l X_{l-1}).
$$

This architecture is in the following referred to as *simple* transformer. The
predicted label $y_{\text{pred}}^{(n+1)}$ is obtained by the output of the last
layer of the transformer and reads $y_{\text{pred}}^{(n+1)} := -[Z_L]_{(d+1),
(n+1)}$. Since the goal is to train this simple transformer on the data
distribution of $Z_0$ and w.r.t. the in-context loss (1), we define:

$$
\begin{align*}
f(\{V\}_{l=1}^L, \{B\}_{l=1}^L, \{C\}_{l=1}^L)
&= \mathbb{E}_{Z_0, y^{(n+1)}}
\left[\left([Z_L]_{(d+1),(n+1)} + y^{(n+1)}\right)^2\right].
\end{align*}
$$

## Linear Transformers implement Gradient Descent

In this section, we consider the problem of learning linear functions, i.e.
$\phi(x) = \langle x, \theta \rangle$ with $\theta \sim \mathcal{P}\_\theta$ and
a simple $L$-layer transformer with linear attention layers which are defined as
follows: For

<!-- In the setting of learning linear functions, i.e. $\phi(x) = \langle x, \theta
\rangle$ with $\theta \sim \mathcal{P}\_\theta$, Oswald et al. and Ahn et al.
[[3,4]](#3) have shown that the output generated by a forward pass of a simple
$L$-layer transformer with linear attention modules is equivalent to $L$ steps
of gradient descent. Linear attention layers are a special case of the
generalized attention module: For -->

$$
Q := \begin{pmatrix}
B & 0\\
0 & 0
\end{pmatrix}, \qquad
K := \begin{pmatrix}
C & 0\\
0 & 0
\end{pmatrix}
$$

we have

$$
\operatorname{Attn}^{linear}_{V_l, K_l, Q_l}(Z_{l-1})
:= V_l Z M Z^\top K_l^\top Q_l Z.
$$

Therefore, a linear attention layer is a special case of the generalized
attention layer and corresponds to setting the function $\tilde{h}(U,W) = U^\top
W$. Note that an $L$-layer simple transformer with linear attention layers
(short: linear Transformer) is not a linear model because each attention module
is still cubic in the input. For simplicity, we reparametrize the weights and
write only $Q_l$ for the product $Q_l^\top K_l$ in the rest of this section. The
key result of the work by Oswald et al. and Ahn et al. [[3,4]](#3) is that the
output generated by a linear transformer can equivalently be obtained by
executing gradient descent on the input $Z_0$.

To start, we consider the case of a single attention module, i.e. $L=1$. Ahn
et al. [[3]](#3) have proven that for $x^{(i)} \sim \mathcal{N}(0,\Sigma)$ with
$\Sigma = U \operatorname{diag}(\lambda_1, \ldots, \lambda_d) U^\top$ and
$\theta \sim \mathcal{N}(0, I_n)$ the following parameters

$$
\begin{align}
V_0 = \begin{pmatrix}
0_{d \times d} & 0\\
0 & 1
\end{pmatrix}, \qquad
Q_0 = - \begin{pmatrix}
U \operatorname{diag}\left(\left\{
\frac{1}{\frac{n+1}{n} \lambda_i + \frac{1}{n} \sum_k \lambda_k}
\right\}_{i=1,\ldots,d} \right)U^\top & 0\\
0 & 0
\end{pmatrix}
\end{align}
$$

are global minimizers of the in-context loss $f$. Note that this configuration
is not unique because one can simply rescale by setting $V_ 0 \leftarrow s V_0$
and $Q_0 \leftarrow Q_0/s$ with $s\in \mathbb{R}$. The purpose of this result is
to demonstrate that this parameter configuration could indeed be obtained by
training the linear transformer on the data $Z_0$ w.r.t. the in-context loss.
Furthermore, for $\Sigma = I_d$, Oswald et al.  [[4]](#4) have shown that a
linear transformer with the same parameters as in (2) implements one step of
gradient descent.

As the setting of a single layer and isotropic Gaussians for the input is quite
restrictive, Ahn et al. [[3]](#3) have partially extended the results to
multiple layers and general covariance matrices.

## References

<a id="1">[1]</a>
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances
in neural information processing systems, 2017.

<a id="2">[2]</a>
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A.
Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot
learners. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

<a id="3">[3]</a>
Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn
to implement preconditioned gradient descent for in-context learning. arXiv
preprint arXiv:2306.00297, 2023.

<a id="4">[4]</a>
Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento,
Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn
in-context by gradient descent. In International Conference on Machine Learning,
pages 35151–35174. PMLR, 2023.

<a id="5">[5]</a>
Xiang Cheng, Yuxin Chen, & Suvrit Sra. (2024). Transformers Implement Functional
Gradient Descent to Learn Non-Linear Functions In Context.
