---
layout: post
title:  "An Explanation of In-Context Learning"
date:   2024-06-27 10:25:14 +0200
categories: jekyll update
mathjax: true
# usemathjax: true
---
{% include mathjax.html %}

In recent years, large language models (LLMs) and in particular Transformers
[[1]](#1) have demonstrated astounding abilities in various fields of modern
machine learning. It has been hypothesized that their success is due to a
capability called *in-context* learning [[2]](#2): Without changing its
parameters, the LLM produces the correct output for a certain task solely based
on instructions or demonstrations contained in a prompt.

The mechanisms enabling in-context learning remain poorly understood to this
day. A possible explanation, which is actively being researched, is the ability
of Transformers to implement algorithms that extract specific information from
the input. In particular, recent work [[3,4,5]](#3) has shown that certain
in-context learners can develop a context-dependent model within their hidden
activations that is trained on the examples provided in the prompt.

In this blog post, we revisit the work in [[3,4,5]](#3) and explain the
algorithms that can be learned by Transformers to perform in-context learning.
While Oswald et al. and Ahn et al. [[3,4]](#3) focus on linear Transformers
(i.e. the attention module contains no nonlinear activations) learning linear
functions, Cheng et al. [[5]](#3) consider the more general case of nonlinear
Transformers learning certain nonlinear functions.

## Input Data for In-Context Learning

As a data model that makes in-context learning tractable, Oswald et al., Ahn et
al. and Cheng et al. [[3,4,5]](#3) consider the following random instances: The
prompt contains $n$ context tokens $z^{(i)} = (x^{(i)}, y^{(i)})^\top \in
\mathbb{R}^{d+1}$ and a query token $z^{(n+1)} = (x^{(n+1)}, 0)^\top \in
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

The zero is a placeholder for the unknown value $y^{(n+1)}$. Since our goal is
to predict this label with a Transformer model that is trained on this data
distribution, we need to define a loss function that measures the difference of
the predicted label $y_{\text{pred}}^{(n+1)}$ and the true label $y^{(n+1)}$.
The so-called *in-context loss* is the expected squared error:

$$
\begin{align}
\mathbb{E}_{Z_0, y^{(n+1)}}
\left[ \left( y_{\text{pred}}^{(n+1)} - y^{(n+1)} \right)^2 \right].
\end{align}
$$

Last, we remark that an important case of $\mathcal{P}\_{Y|X}$ is when $y^{(i)}
= \phi(x^{(i)})$ for some unknown function $\phi$. In particular, Ahn et al.
[[3]](#3) consider $\phi(x) = \langle x, \theta \rangle$ for some $\theta \in
\mathbb{R}^{d}$ which corresponds to a linear regression model.

## Simple Transformer Model

Having seen the input data setup, we now introduce the Transformer architecture
that is used to theoretically analyze in-context learning. The central component
in a Transformer model is the attention module [[1]](#1). Originally, Vaswani et
al. [[1]](#1) defined an attention layer as a function that maps values $V \in
\mathbb{R}^{(d+1) \times (d+1)}$, keys $K \in \mathbb{R}^{(d+1) \times (d+1)}$
and querys $Q \in \mathbb{R}^{(d+1) \times (d+1)}$ to an output:

$$
\operatorname{Attn}_{Q,K,V}^{smax}(Z)
=VZ \cdot \operatorname{smax}\left(Z^\top K^\top Q Z\right).
$$


The softmax function serves as a nonlinear activation function. Cheng et al.
[[5]](#5) introduce the *generalized* attention module to incorporate any matrix
valued function $\tilde{h} : \mathbb{R}^{d \times (n+1)} \times \mathbb{R}^{d
\times (n+1)} \to \mathbb{R}^{(n+1) \times (n+1)}$:

$$
\operatorname{Attn}^{\tilde{h}}_{V, B, C}(Z) := V Z M \tilde{h}(BX, CX),
\qquad
M := \begin{bmatrix}
I_n & 0 \\
0 & 0
\end{bmatrix}
$$

where $X = [x^{(1)}, \ldots, x^{(n+1)}] \in \mathbb{R}^{d \times (n+1)}$
consists of the firs $d$ rows of $Z$ and $V \in \mathbb{R}^{(d+1) \times
(d+1)}$, $B \in \mathbb{R}^{d \times d}$, $C \in \mathbb{R}^{d \times d}$ are
the value, key, and query matrices. $M$ acts simply as a mask that reflects the
asymmetric structure of the input data $Z$ that results from the missing label
$y^{(n+1)}$.

Usually, a Transformer module consists of multiple multi-head attention
modules, normalization and feed-forward layers [[1]](#1). However, for the
purpose of analyzing in-context learning, we consider a single-head
attention only $L$-layer Transformer that is constructed by stacking $L$
attention modules. $Z_l$ denotes the output of the $(l-1)$-th layer and is
defined as

$$
Z_l
:= Z_{l-1} + \operatorname{Attn}^{\tilde{h}_l}_{V_l, B_l, C_l}(Z_{l-1})
= Z_{l-1} + V_l Z_{l-1} M_l \tilde{h}_l(B_l X_{l-1}, C_l X_{l-1}).
$$

This architecture is in the following referred to as *simple* Transformer. The
predicted label $y_{\text{pred}}^{(n+1)}$ is obtained by the output of the last
layer of the Transformer and reads $y_{\text{pred}}^{(n+1)} := -[Z_L]_{(d+1),
(n+1)}$. Since the goal is to train this simple Transformer on the data
distribution of $Z_0$ and w.r.t. the in-context loss (1), we define:

$$
\begin{align*}
f(\{V_l\}_{l=1}^L, \{B_l\}_{l=1}^L, \{C_l\}_{l=1}^L)
&= \mathbb{E}_{Z_0, y^{(n+1)}}
\left[\left([Z_L]_{(d+1),(n+1)} + y^{(n+1)}\right)^2\right]
\tag{1}
\end{align*}
$$

where $\\{V\_l\\}\_{l=1}^L, \\{B\_l\\}\_{l=1}^L, \\{C\_l\\}\_{l=1}^L$ denote
collections of the attention parameters across layers.

## Linear Transformers implement Gradient Descent

In this section, we consider the problem of learning linear functions, i.e.
$\phi(x) = \langle x, \theta_* \rangle$ with $\theta_* \sim \mathcal{P}\_\theta$
and a simple $L$-layer Transformer with linear attention layers which are
defined as follows: For

$$
Q_l := \begin{pmatrix}
B_l & 0\\
0 & 0
\end{pmatrix}, \qquad
K_l := \begin{pmatrix}
C_l & 0\\
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
W$. Note that such an $L$-layer *linear* Transformer is not a linear model
because each attention module is still cubic in the input. Furthermore,
experiments have shown that linear attention outperforms the usual softmax
attention for solving the linear regression tasks we consider here [[4]](#4).
The choice of the sparse weight matrices as above is a major restriction
compared to the standard attention but will be justified later.  For simplicity,
we reparametrize the weights and write only $Q_l$ for the product $Q_l^\top K_l$
in the rest of this section. The key result of the work by Oswald et al. and Ahn
et al. [[3,4]](#3) is that linear Transformers can learn to implement
(preconditioned) gradient descent. In the following, we will explain this result
and how it can be derived.

To start, consider the case of a single attention module, i.e. $L=1$. Ahn et al.
[[3]](#3) have proven that for $x^{(i)} \sim \mathcal{N}(0,\Sigma)$ with $\Sigma
= U \operatorname{diag}(\lambda_1, \ldots, \lambda_d) U^\top \in \mathbb{R}^{d
\times d}$ and $\theta \sim \mathcal{N}(0, I_n)$ the following parameters

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
\tag{2}
\end{align}
$$

are global minimizers w.r.t. the in-context loss $f$ (1). In orther words, these
parameters produce optimal predictions $y_{\text{pred}}^{(n+1)}$ within the
limitations of the architecture. Note that configuration (2) is not unique
because one can simply rescale by setting $V_ 0 \leftarrow s V_0$ and $Q_0
\leftarrow Q_0/s$ with $s\in \mathbb{R}$.  Furthermore, for $\Sigma = I_d$,
Oswald et al. [[4]](#4) have shown that a linear Transformer with the same
parameters as in (2) implements one step of gradient descent. Taking these
results together, this means that the forward pass of a single-layer linear
Transformer that is trained on the in-context loss (1) can be described by a
step of gradient descent.

Next, we clarify what it means to execute gradient descent on the input $Z_0$
and generalize the setting to multiple layers, i.e. $L>1$. Inspired by the
parameters in (2), we restrict the parameter space to the following set of
matrices:

$$
\begin{align}
V_l = \begin{pmatrix}
0_{d \times d} & 0\\
0 & 1
\end{pmatrix},
\quad
Q_l = - \begin{pmatrix}
A_l & 0\\
0 & 0
\end{pmatrix}
\quad
\text{where}
\quad
A_l \in \mathbb{R}^{d \times d}
.
\tag{3}
\end{align}
$$

Although this set is quite limited, it turns out that $L$-layer linear
Transformers with the parameters (3) can implement various standard optimization
methods. Specifically, Ahn et al. [[3]](#3) have shown that for the prediction
$y_l^{(n+1)}$ of the $l$-th layer it holds $y_l^{(n+1)} = - \langle x^{(n+1)},
\theta_l^{\text{gd}} \rangle$ where the sequence ${\theta_l^{\text{gd}}}$ is
defined as

$$
\begin{align}
\theta_0^{\text{gd}} = 0, \qquad
\theta_l^{\text{gd}} = \theta_{l-1}^{\text{gd}} -
A_l \nabla R_{\theta_*}(\theta_{l-1}^{\text{gd}}).
\tag{4}
\end{align}
$$

$R_{\theta_*}(\theta)$ denotes the empirical in-context loss

$$
\begin{align}
R_{\theta_*}(\theta) := \frac{1}{2n} \sum_{i=1}^n (\theta^\top x_i -
\theta_*^\top x_i)^2.
\end{align}
$$

Interestingly, the iterative scheme in (4) covers multiple standard optimization
methods:

* gradient descent (GD) for $A_l = \eta_l I_d$ where $\eta_l$ is the step size,
* preconditioned gradient descent (PGD) for $A_l = P_l$ where $P_l$ is a time
dependent preconditioner,
* Newton's method for $A_l = \left(\nabla^2
R_{\theta_*}(\theta_{l-1}^{\text{gd}}) \right)^{-1}$.

Therefore, the forward pass of an $L$-layer linear Transformer can be described
by one of these optimization methods, as long as its parameters are chosen
accordingly. At this point, the remaining question is: What parameters can be
learned by a linear Transformer when being trained on the in-context loss (1)?

Ahn et al. [[3]](#3) have found a parameter configuration that is part of the
space (3) and contains a stationary point of the in-context loss. Assuming
normal distributed data $x^{(i)} \sim \mathcal{N}(0, \Sigma)$ and $\theta_* \sim
\mathcal{N}(0, \Sigma^{-1})$, we define an even more restricted parameter space
$\mathcal{S} := \left\\{ \\{A_l\\}_{l=1}^L \ | \ A_l = a_l \Sigma^{-1}\ \forall
l \right\\}$. Then, it holds

$$
\inf_{A \in \mathcal{S}} \sum_{i=0}^{L-1} \|\nabla_{A_i} f(A)\|_F^2 = 0,
$$

where $f(A)$ is the in-context loss (1) and $\nabla_{A_i} f$ is the derivative
w.r.t. the Frobenius norm [[3]](#3). Plugging the parameters $A_l = a_l
\Sigma^{-1}$ into the iterative scheme (4) reveals that a trained linear
Transformer could implement gradient descent with the data-dependent
preconditioner $\Sigma^{-1}$. Furthermore, experiments have confirmed this
theory and shown that the suggested parameters are indeed recovered during
training [[3]](#3). We remark that this result and in particular the
corresponding proof heavily relies on the normality assumption of the data and
it is questionable to what extent this result can be generalized to other
distributions. One possible direction for future research could be to exploit
the approximation of more general distributions by normal mixtures.

Having seen that linear Transformers can learn to implement gradient descent,
one might wonder if they can also implement other potentially more sophisticated
algorithms. To answer this question, Ahn et al. [[3]](#3) have extended the
parameter space (3) to

$$
\begin{align}
V_l = \begin{pmatrix}
B_l & 0\\
0 & 1
\end{pmatrix},
\quad
Q_l = - \begin{pmatrix}
A_l & 0\\
0 & 0
\end{pmatrix}
\quad
\text{where}
\quad
A_l, B_l \in \mathbb{R}^{d \times d}
.
\tag{5}
\end{align}
$$

By doing so, the Transformer gains representational power and it turns out that
the matrices in (5) also contain a stationary point of the in-context loss (1).
Assuming once again normal distributed data $x^{(i)} \sim \mathcal{N}(0,
\Sigma)$, $\theta_* \sim \mathcal{N}(0, \Sigma^{-1})$ and defining the parameter
space $\mathcal{S} := \left\\{ \\left(\\{A_l\\}\_{l=1}^L, \\{B\\}\_{l=1}^L
\\right) \ | \ A_l = a_l \Sigma^{-1}, \ B_l = b_l I_d \ \ \forall l \right\\}$,
Ahn et al. [[3]](#3) have shown that it holds:

$$
\inf_{(A,B) \in \mathcal{S}} \sum_{i=0}^{L-1}
\left\| \nabla_{A_i} f(A, B)\right\|_F^2 +
\left\| \nabla_{B_i} f(A, B) \right\|_F^2 = 0.
$$

Setting the suggested parameters in the linear attention layer
$\operatorname{Attn}^{linear}_{V_l, Q_l}(Z)$ reveals that the matrix $A_l$ still
acts as a preconditioner, while the matrix $B_l$ is responsible for transforming
the gram matrix $XX^\top$ to improve its conditioning. In particular, the
covariance $\Sigma = I_d$ corresponds to the GD++ algorithm that was introduced
by Oswald et al. [[4]](#4). This algorithm is based on gradient descent but uses
an interative curvature correction $x_i \leftarrow (b_l I- XX^\top)x_i$ for all
inputs $x_i$. Experiments have shown that GD++ outperforms plain gradient
descent on the linear regression setup and describes the behavior of trained
Transformers very well [[4]](#4).

So far, we have seen that linear Transformers can learn to implement
preconditioned gradient descent but also more involved algorithms. Note that
there might be further unexplored algorithms that can be learned by linear
Transformers. However, since the linear attention layer is only a special case
of the generalized attention layer, we will now turn to more general cases and
try to reintroduce non-linearities into the attention module.

## Non-Linear Transformers implement Functional GD

In this section, we extend the problem of linear regression to kernel regression
for a kernel $\mathcal{K}$. Cheng et al.  [[5]](#5) have shown that for a
certain choice of matrices $\\{V_l\\}\_{l=1}^L, \\{B_l\\}\_{l=1}^L,
\\{C_l\\}\_{l=1}^L$ and function $\tilde{h}$, Transformers can learn to
implement functional gradient descent.

First, we clarify the question: What is *functional gradient descent* or
*gradient descent in function space*? Let $\mathbb{H}$ be a Hilbert space of
functions $g: \mathbb{R}^d \rightarrow \mathbb{R}$ and $L(g): \mathbb{H}
\rightarrow \mathbb{R}$ be a loss. Then, the functional gradient descent on
$L(g)$ is defined as the sequence

$$
g_{l+1} = g_l - \eta_l \nabla L(g_l)
$$

where $\nabla L(g) := \arg\min_{\\|e\\|\_{\mathcal{H}}=1} \left\. \frac{d}{dt}
L(g + te) \right\|_{t=0}$ and $\eta_l$ is the step size [[5]](#5).

In the context of kernel regression, we consider the following setup: The
Hilbert space $\mathbb{H}$ is the reproducing kernel Hilbert space (RKHS) of the
kernel $\mathcal{K}$ and $L(g) := \sum_{i=1}^n \left( g(x^{(i)}) - y^{(i)}
\right)^2$ is the empirical squared error of the in-context samples. Now, the
key result of Cheng et al. [[5]](#5) is the following: Let $V_l =
\begin{pmatrix} 0\_{d \times d} & 0 \\\ 0 & -\eta_{\ell} \end{pmatrix}$,
$B_l=I_d$, $C_l = I_d$ and $\left[\tilde{h}(U,W)\right]_{ij} =
\mathcal{K}(U^{(i)},W^{(j)})$ where $U^{(i)}, W^{(j)}$ are the $i$-th and $j$-th
columns of $U$ and $W$. Then, there exist stepsizes $\eta_0, \ldots, \eta_l \in
\mathbb{R}$ such that the prediction $y_l^{(n+1)}$ of the $l$-th layer of the
Transformer matches the prediction of the functional gradient descent after $l$
steps:

$$
\begin{align}
y_l^{(n+1)} = - g_l(x^{(n+1)}).
\tag{6}
\end{align}
$$

In other words, Transformers with a kernel as the activation function
$\tilde{h}$, can implement functional gradient descent if its weights are chosen
accordingly. Note that this result covers a wide range of activation functions.
For linear attention modules, i.e. the euclidean inner product kernel
$\mathcal{K}^{\text{linear}}(u,w) = \langle u, w\rangle$, this result has
already been discovered by Ahn et al. [[3]](#3) and corresponds to setting $A_l
= \eta_l I_d$ in (4). Nevertheless, there is problem with result (6): It does
not hold for ReLU and softmax activations because $\tilde{h}^{\text{ReLU}},
\tilde{h}^{\text{smax}}$ are not kernels. Since softmax is the most common
activation function in practice, we will now explain how to work with it anyway.

It turns out that Transformers with softmax activation can implement a modified
variant of functional gradient descent in which the stepsizes $\eta_l$ are
multiplied by a normalizing factor $\tau(\cdot)$ [[5]](#5). This can be derived
by comparing the behavior of the softmax attention with attention based on
exponential kernel $\mathcal{K}^{\text{smax}}(u,w) = \exp\left(\frac{\langle u,
w\rangle}{\sigma}\right)$ where $\sigma$ is a scaling factor. The latter is
obtained by using $\tilde{h}^{\text{exp}}(\cdot) = \text{exp}(\cdot)$ and
setting

$$
\begin{align*}
V_l = \begin{pmatrix}
0_{d \times d} & 0\\
0 & -\eta_l
\end{pmatrix}, \quad
B_l = \frac{1}{\sigma} I_d, \quad
C_l = \frac{1}{\sigma} I_d.
\end{align*}
$$

The usual softmax attention module can be written as

$$\operatorname{Attn}^{\text{smax}}_{V_l, B_l, C_l}(Z) = V_l Z \cdot
\text{smax}\left(
Z^\top
\begin{pmatrix}
B_l & 0\\
0 & 0
\end{pmatrix}^\top
\begin{pmatrix}
C_l & 0\\
0 & 0
\end{pmatrix}
Z
\right)
$$

where $\text{smax}(\cdot)$ denotes the masked softmax function

$$
\left[ \text{softmax}(W) \right]_{ij} =
\begin{cases}
\frac{\exp(W_{ij})}{\sum_{k=1}^{n} \exp(W_{kj})} & \text{for } i \ne n + 1 \\
0 & \text{for } i = n + 1
\end{cases}.
$$

Setting $V_l, B_l, C_l$ as above, we obtain the algorithm

$$
g_{l+1}(\cdot) = g_l(\cdot) + \eta_l \tau(\cdot) \sum_{i=1}^{n}
\left(
y^{(i)} - g_l(x^{(i)})
\right)
\mathcal{K}(\cdot, x^{(i)}),
$$

where $\tau(\cdot) := 1 / \sum_{i=1}^{n} \mathcal{K}(\cdot, x^{(j)})$ is the
softmax normalization.

Now, we turn back to result (6). So far, we have not seen a justification for
the choice of parameters $\\{V_l\\}\_{l=1}^L, \\{B_l\\}\_{l=1}^L,
\\{C_l\\}\_{l=1}^L$. In particular, the question is: Can these parameters be
learned by Transformers? Similar to the result of Ahn et al. [[3]](#3) for the
linear case, Cheng et al. [[5]](#5) have proven that these parameters form a
stationary point w.r.t. the in-conext loss (1). However, we need addtional
assumptions for this result: First, assume that for a matrix $S \in
\mathbb{R}^{d \times d}$, it holds $\tilde{h}(W, V) = \tilde{h}(S^\top W,
S^{-1}V)$. Second, let there be a symmetric invertible matrix $\Sigma \in
\mathbb{R}^{d \times d}$ such that for any orthogonal matrix $U$ and $X \sim
\mathcal{P}\_X$ it holds $\Sigma^{-\frac{1}{2}} U \Sigma^{\frac{1}{2}} X
\overset{d}{=} X$. Third, for the covariance $\mathbb{K}(X) =\mathbb{E}_{Y|X}[Y
Y^\top]$ of $Y := [y^{(1)}, \ldots, y^{(n+1)}]$ let it hold
$\mathbb{K}(\Sigma^{\frac{1}{2}} U \Sigma^{- \frac{1}{2}}X) = \mathbb{K}(X)$.
Then, there exist stationary points where $B_l = b_l \Sigma^{-\frac{1}{2}}$ and
$C_l = c_l \Sigma^{-\frac{1}{2}}$. Setting $\Sigma = I_d$, we obtain the same
construction as in (6) and therefore a Transformer with kernel activations can
learn to implement functional gradient descent.

The assumptions for the result above are quite strong but can be justified:
Linear, ReLU and softmax activation all fulfill the required matrix invariance.
Moreover, the distribution assumption on $X$ holds when choosing a normal
distribution or a normal mixture for $\mathcal{P}\_X$. The covariance assumption
on $Y$ is for example satisfied in the linear regression setting, i.e. $y^{(i)}
= \langle \xi^{(i)}, \theta \rangle$ where $x^{(i)} = \Sigma^{1/2} \xi^{(i)}$
and $\theta \sim \mathcal{N}(0, I_d)$.

We close this section by demonstrating the power of the functional gradient
descent learned by Transformers: Cheng et al. [[5]](#5) have shown that this
algorithm produces nearly statistical optimal predictions if the activation
$\tilde{h}$ matches the data distribtion. To obtain a data distribution that is
based on our kernel $\mathcal{K}$, we define a $\mathcal{K}$ *Gaussian process*
as the conditional distribution

$$
Y | X \sim \mathcal{N}\left(0, \mathcal{K}(X)\right),
$$

where $\mathcal{K}(X)\_{ij} = \mathcal{K}(x^{(i)}, x^{(j)})$. Now, let
$\mathcal{P}\_{Y|X}$ be a $\mathcal{K}$-Gaussian process,
$\left[\tilde{h}(U, W)\right]_{ij} := \mathcal{K}\left(U_i, W_j \right) $ and
consider the functional gradient descent as in (6). Then, as the number of
layers in the Transformer $l \rightarrow \infty$, the prediction $y_l^{(n+1)}$
of each layer approaches the Bayes optimal prediction in terms of the in-context
loss [[5]](#5). In other words, the functional gradient descent learned by
Transformers generates optimal outputs after infinite steps. Note that this
result is limited to the parameter space of matrices $\\{V_l\\}\_{l=1}^L,
\\{B_l\\}\_{l=1}^L, \\{C_l\\}\_{l=1}^L$ used to implement the functional
gradient descent in (6). For different parameters or a finite amout of layers,
there might be a different choice of the activation $\tilde{h}$ that recovers
the Bayes estimator, too.

## Limitations and Outlook

The results we have presented in this blog post are a huge step forward towards
understanding in-context learning of modern Transformers architectures. However,
there are still some limitations and open questions. The major restriction in
this line of research is the data generation or task the Transformer is given.
All of the results are tailored to the specific setup of random instances of
regression and it is unclear if the results can be generalized to other tasks or
input data. In contrast, other work focussing on the *induction head* mechanism
[[7]](#7) instead of algorithms hidden in the Transformers parameters, use far
more sophisticated data setups. For instance, Bietti et al. [[6]](#6) work with
a bigram language model, i.e. Markov chain, that combines a global and an
in-context distribution. On the other hand, their Transformer model is
restricted to $2$-layers and can therefore not compete with the architectures we have seen in [[3,4,5]](#3).

Another limitation are the restrictions of the parameter space. With one
exception, all shown results are based on the sparsity constraints:

$$
\begin{align*}
V_l = \begin{pmatrix}
V_l & 0\\
0 & v_l
\end{pmatrix}, \quad
Q_l := \begin{pmatrix}
B_l & 0\\
0 & 0
\end{pmatrix}, \quad
K_l := \begin{pmatrix}
C_l & 0\\
0 & 0
\end{pmatrix}
\end{align*}
\quad
\text{where}
\quad
V_l =0_{d \times d}.
$$

Intuitively, this ensures that the key and query matrix do not put any attention
on the unseen label $y^{(n+1)}$ and the value matrix maps only the prediction
$y_{\text{pred}}^{(n+1)}$. However, in practice, $V_l$ is often a multiple of
$I_d$ [[5]](5) and for linear activations correponds to the setup as in (5).

To conclude, Transformers can learn to implement preconditioned and functional
gradient descent in its forward pass, enabling them to learn (non-)linear
functions in-context. Apart from giving useful insights into the internal
mechanisms of Transformers, this work could be used in the future to improve
training processes and obtaining prediction error bounds. All of the results
have been verified empirically and we encourage the reader to check out the
original papers [[3,4,5]](#3) for more details. Future research could
investigate the found set of stationary points of the in-context loss (1),
quantify their optimality and try to find a global minimum for multi-layer
Transformers. Moreover, understanding more sophisticated algorithms implemented
by Transformers beyond standard optimization methods would be interesting.

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
Xiang Cheng, Yuxin Chen, and Suvrit Sra. Transformers Implement Functional
Gradient Descent to Learn Non-Linear Functions In Context. arXiv preprint
arXiv:2312.06528, 2023.

<a id="6">[6]</a>
Bietti, A., Cabannes, V., Bouchacourt, D., Jegou, H., & Bottou, L. (2023). Birth
of a Transformer: A Memory Viewpoint. In Advances in Neural Information
Processing Systems (pp. 1560–1588). Curran Associates, Inc..

<a id="7">[7]</a>
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep
Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt,
Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and
Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread,2021. https://transformer-circuits.pub/2021/framework/index.html.
