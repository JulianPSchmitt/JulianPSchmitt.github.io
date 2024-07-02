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
of transformers to implement algorithms that extract information from the input
prompt. In particular, recent work [[3,4,5]](#3) has shown that certain
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
distribution.

An important case of $\mathcal{P}\_{Y|X}$ is when $y^{(i)} = \phi(x^{(i)})$ for
some unknown function $\phi$. In particular, Ahn et al. [[3,4]](#3) consider
$\phi(x) = \langle x, \theta \rangle$ for some $\theta \in \mathbb{R}^{d}$ which
corresponds to a linear regression model.

## Simple Transformer Model

The central component in a transformer model is the attention mechanism
[[1]](#1). Cheng et al. [[5]](#5) define the *generalized* attention module as

$$
\operatorname{Attn}^{\hat{h}}_{V, B, C}(Z) := V Z M \tilde{h}(BX, CX),
\qquad
M := \begin{bmatrix}
I_n & 0 \\
0 & 0
\end{bmatrix}
$$

where $V \in \mathbb{R}^{(d+1) \times (d+1)}$, $B \in \mathbb{R}^{d \times d}$,
$C \in \mathbb{R}^{d \times d}$ are the value, key, and query matrices. $M$ is a
mask that reflects the asymmetric structure of the input data $Z$ that results
from the missing label $y^{(n+1)}$. $\tilde{h} : \mathbb{R}^{d \times (n+1)}
\times \mathbb{R}^{d \times (n+1)} \to \mathbb{R}^{(n+1) \times (n+1)}$ is a
matrix valued function and $X = [x^{(1)}, \ldots, x^{(n+1)}] \in \mathbb{R}^{d
\times (n+1)}$ consists of the firs $d$ rows of $Z$.

Usually, a transformers module consists of multiple attention modules and
feed-forward layers [[1]](#1). However, for the purpose of analyzing in-context
learning, a *simple* $L$-layer transformer is constructed by stacking $k$
attention modules. $Z_l$ denotes the output of the $(l-1)$-th layer and defined
as

$$
Z_l
:= Z_{l-1} + \operatorname{Attn}^{\hat{h}_l}_{V_l, B_l, C_l}(Z_{l-1})
= Z_{l-1} + V_l Z_{l-1} M_l \tilde{h}_l(B_l X_{l-1}, C_l X_{l-1}).
$$

Since the goal is to train this simple transformer on the data distribution of
the input $Z_0$, we need to define a loss function that measures the difference
of the predicted label $y_{\text{pred}}^{(n+1)}$ and the true label $y^{(n+1)}$.
The prediction is obtained by the output of the last layer of the transformer
and reads $y_{\text{pred}}^{(n+1)} := -[Z_L]_{(d+1), (n+1)}$. The in-context
loss is then defined as

$$
f(\{V\}_{l=1}^L, \{B\}_{l=1}^L, \{C\}_{l=1}^L)
= \mathbb{E}_{Z_0, y^{(n+1)}}
\left[ \left( y_{\text{pred}}^{(n+1)} - y^{(n+1)} \right)^2 \right].
$$

## Gradient Descent for Linear Transformers

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
