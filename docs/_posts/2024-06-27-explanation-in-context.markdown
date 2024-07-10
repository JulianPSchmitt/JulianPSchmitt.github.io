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
\tag{1}
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

Usually, a transformer module consists of multiple multi-head attention
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
f(\{V_l\}_{l=1}^L, \{B_l\}_{l=1}^L, \{C_l\}_{l=1}^L)
&= \mathbb{E}_{Z_0, y^{(n+1)}}
\left[\left([Z_L]_{(d+1),(n+1)} + y^{(n+1)}\right)^2\right]
\end{align*}
$$

where $\\{V\_l\\}\_{l=1}^L, \\{B\_l\\}\_{l=1}^L, \\{C\_l\\}\_{l=1}^L$ denote
collections of the attention parameters across layers.

## Linear Transformers implement Gradient Descent

In this section, we consider the problem of learning linear functions, i.e.
$\phi(x) = \langle x, \theta_* \rangle$ with $\theta_* \sim \mathcal{P}\_\theta$
and a simple $L$-layer transformer with linear attention layers which are
defined as follows: For

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
W$. Note that such an $L$-layer *linear* Transformer is not a linear model
because each attention module is still cubic in the input. Furthermore,
experiments have shown that linear attention outperforms the usual softmax
attention [[4]](#4). For simplicity, we reparametrize the weights and write only
$Q_l$ for the product $Q_l^\top K_l$ in the rest of this section. The key result
of the work by Oswald et al. and Ahn et al. [[3,4]](#3) is that the output
generated by a linear transformer can equivalently be obtained by executing
gradient descent on the input $Z_0$.

To start, consider the case of a single attention module, i.e. $L=1$. Ahn et al.
[[3]](#3) have proven that for $x^{(i)} \sim \mathcal{N}(0,\Sigma)$ with $\Sigma
= U \operatorname{diag}(\lambda_1, \ldots, \lambda_d) U^\top$ and $\theta \sim
\mathcal{N}(0, I_n)$ the following parameters

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

are global minimizers w.r.t. the in-context loss $f$. In orther words, these
parameters produce optimal predictions $y_{\text{pred}}^{(n+1)}$ within the
limitations of the architecture. Note that configuration (2) is not unique
because one can simply rescale by setting $V_ 0 \leftarrow s V_0$ and $Q_0
\leftarrow Q_0/s$ with $s\in \mathbb{R}$.  Furthermore, for $\Sigma = I_d$,
Oswald et al. [[4]](#4) have shown that a linear transformer with the same
parameters as in (2) implements one step of gradient descent. Taking these
results together, this means that the forward pass of a linear transformer that
is trained on the in-context loss (1) can be described by a step of gradient
descent.

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
transformers with the parameters (3) can implement various standard optimization
methods. Specifically, Ahn et al. [[3]](#3) have shown that for the prediction
$y_l^{(n+1)}$ of the $l$-th layer it holds $y_l^{(n+1)} = - \langle x^{(n+1)},
\theta_l^{\text{gd}} \rangle$. The sequence ${\theta_l^{\text{gd}}}$ is defined
as

$$
\begin{align}
\theta_0^{\text{gd}} = 0, \qquad
\theta_l^{\text{gd}} = \theta_{l-1}^{\text{gd}} -
A_l \nabla R_{\theta_*}(\theta_{l-1}^{\text{gd}})
\tag{4}
\end{align}
$$

where we used the empirical in-context loss

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

In particular, the forward pass of a linear transformer can be described by one
of these optimization methods, as long as its parameters are chosen accordingly.
Therefore, the next step is to investigate the parameters that are learned by a linear transformer when being trained on the in-context loss (1).

Ahn et al. [[3]](#3) have found a parameter configuration that contains a
stationary point of the in-context loss and as a result could be obtained by
training. Assuming normal distributed data $x^{(i)} \sim \mathcal{N}(0, \Sigma)$
and $\theta_* \sim \mathcal{N}(0, \Sigma^{-1})$, we define an even more
restricted parameter space $\mathcal{S} := \left\\{ \\{A_l\\}_{l=1}^L \ | \ A_l
= a_l \Sigma^{-1}\ \forall l \right\\}$. Then, it holds

$$
\inf_{A \in \mathcal{S}} \sum_{i=0}^{L-1} \|\nabla_{A_i} f(A)\|_F^2 = 0,
$$

where $f(A)$ is the in-context loss (1) and $\nabla_{A_i} f$ is the derivative
w.r.t. the Frobenius norm [[3]](#3). Plugging the parameters $A_l = a_l
\Sigma^{-1}$ into the iterative scheme (4) reveals that a trained linear
transformer could implement gradient descent with the data-dependent
preconditioner $\Sigma^{-1}$. Furthermore, experiments have confirmed this
theory and shown that the suggested parameters are indeed recovered during
training [[3]](#3). We remark that this result and in particular the
corresponding proof heavily relies on the normality assumption of the data and
it is questionable to what extent this result can be generalized to other
distributions. One possible direction for future research could be to exploit
the approximation of more general distributions by normal mixtures.

Having seen the key result of paper [[3]](#3), one might wonder if trained
linear transformers can also implement other algorithms than preconditioned
gradient descent. To answer this question, Ahn et al. [[3]](#3) have extended
the parameter space (3) to

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

By doing so, the transformer gains representational power and it turns out that
it can learn more sophisticated algorithms beyond preconditioned gradient
descent. Assuming once again normal distributed data $x^{(i)} \sim
\mathcal{N}(0, \Sigma)$ and $\theta_* \sim \mathcal{N}(0, \Sigma^{-1})$ and
defining the parameter space $\mathcal{S} := \left\\{ \\left(\\{A_l\\}\_{l=1}^L,
\\{B\\}\_{l=1}^L \\right) \ | \ A_l = a_l \Sigma^{-1}, \ B_l = b_l I_d \ \
\forall l \right\\}$, Ahn et al. [[3]](#3) have shown that it holds:

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
transformers very well [[4]](#4).

So far, we have seen that linear transformers can learn to implement gradient
but also far more involved algorithms. Note that there might be further
unexplored algorithms that can be learned by linear transformers. However, since
the linear attention layer is only a special case of the generalized attention
layer, we will now turn our attention to more general cases and try to
reintroduce non-linearities into the attention module.

## Non-Linear Transformers implement Functional GD

In this section, we extend the problem of solving random linear regression
instances to kernel regression for a kernel $\mathcal{K}$. Cheng et al.
[[5]](#5) have shown that for a certain choice of matrices $V_l, B_l, C_l$ and
function $\tilde{h}$, a non-linear transformer can learn to implement functional
gradient descent.

First, let's clarify the question: What is *functional gradient descent* or
*gradient descent in function space*? Let $\mathbb{H}$ be a Hilbert space of
functions $g: \mathbb{R}^d \rightarrow \mathbb{R}$ and $L(g): \mathbb{H}
\rightarrow \mathbb{R}$ be a loss. Then, the functional gradient descent on
$L(f)$ is defined as the sequence

$$
g_{l+1} = g_l - \eta_l \nabla L(g_l)
$$

where $\nabla L(f) := \arg\min_{\\|g\\|\_{\mathcal{H}}=1} \left\. \frac{d}{dt}
L(f + tg) \right\|_{t=0}$ and $\eta_l$ is the step size [[5]](#5).

In the context of kernel regression, we consider the following setup: The
Hilbert space $\mathbb{H}$ is the reproducing kernel Hilbert space (RKHS) of the
kernel $\mathcal{K}$ and $L(f) := \sum_{i=1}^n \left( g(x^{(i)}) - y^{(i)}
\right)^2$ is the empirical squared error of the in-context samples. Now, the
key result of Cheng et al. [[5]](#5) is the following: Let $V_l =
\begin{bmatrix} 0 & 0 \\\ 0 & -r_{\ell} \end{bmatrix}$, $B_l=I_d$, $C_l = I_d$
and $\left[\tilde{h}(U,W)\right]_{ij} = \mathcal{K}(U^{(i)},W^{(j)})$ where
$U^{(i)}, W^{(j)}$ are the $i$-th and $j$-th columns of $U$ and $W$. Then, the
prediction $y_l^{(n+1)}$ of the $l$-th layer of the transformer matches the
prediction of the functional gradient descent after $l$ steps:

$$
y_l^{(n+1)} = - g_l(x^{(n+1)}).
$$

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
Xiang Cheng, Yuxin Chen, & Suvrit Sra. Transformers Implement Functional
Gradient Descent to Learn Non-Linear Functions In Context. arXiv preprint
arXiv:2312.06528, 2023.
