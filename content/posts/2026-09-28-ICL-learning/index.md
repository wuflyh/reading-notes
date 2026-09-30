---
title: What learning algorithm is in-context learning? Investigations with linear models
date: 2026-09-28
tags:
  - ICL
  - meta-learning
math: true
draft: false
meta:
  authors:
    - Ekin Akyurek et al.
  venue: "ICLR 2023"
  url: https://arxiv.org/abs/2211.15661
  year: 2023
  month: 5
---
{{< paper >}}

## 1. Summary

This paper studies the **ICL** mechanism of transformers on linear regression tasks. The forward pass of a trained transformer closely replicates the **learning trajectory of the underlying linear regressor**. Either gradient descent iteration, or ridge regression close-form update can be constructed through transformers with fixed hidden width or layer depth. The algorithmic alignment measure on **SPD**(Squared Prediction Difference) and **ILWD**(Implicit Linear Weight Difference) confirms that transformer ICL produces the same numerical outputs and converged to the exact same linear parameter solution.

## 2. ICL as learning

- **Step1**: Encode a linear regression dataset and final query into a sequence of tokens.

$$
\mathcal{D}=\left\{\left(\mathbf{x}_1, y_1\right),\left(\mathbf{x}_2, y_2\right), \ldots,\left(\mathbf{x}_n, y_n\right)\right\}
$$

$$
\tilde{\mathbf{x}}_i=\left[\begin{array}{c}
0 \\
\mathbf{x}_i
\end{array}\right] \in \mathbb{R}^{d+1}, \quad \tilde{\mathbf{y}}_i=\left[\begin{array}{c}
y_i \\
\mathbf{0}_d
\end{array}\right] \in \mathbb{R}^{d+1}
$$

$$
\mathbf{S}=\left[\tilde{\mathbf{x}}_1, \tilde{\mathbf{y}}_1, \tilde{\mathbf{x}}_2, \tilde{\mathbf{y}}_2, \ldots, \tilde{\mathbf{x}}_n, \tilde{\mathbf{y}}_n, \tilde{\mathbf{x}}_{\text{query}}\right]
$$

Where $\mathcal{D}$ is the dataset with feature vectors $\mathbf{x}_i \in \mathbb{R}^d$ and labels $y_i \in \mathbb{R}$; $\mathbf{S}$ is the sequence of $(2n+1)$ tokens, whose last token $\mathbf{x}_{\text{query}}$ is the encoded query vector. The first coordinate acts as a scalar/label channel (holding ${y}_i$ or $\hat{y_i}$). The remaining $d$ coordinates hold the feature vectors $\mathbf{x}_i$.

- **Step2**: Forward inference from left to right equals one step of gradient descent — **(1)** the current prediction is performed (MLP/Attention), **(2)** residual error computed (Causal Attention/Residual), **(3)** gradient aggregated (Self-Attention Pooling), and **(4)** weight vector updated (Residual Stream Addition)
$$
\hat{y}_i = \mathbf{w}^{(l)\top} \mathbf{x}_i \tag{1}
$$
$$
e_i = \hat{y}_i - y_i \tag{2}
$$

$$
\Delta \mathbf{w} = -\frac{\eta}{n} \sum_{i=1}^n e_i \mathbf{x}_i \tag{3}
$$
$$
\mathbf{w}^{(l+1)} = \mathbf{w}^{(l)} + \Delta \mathbf{w} \tag{4}
$$

- **Step3**: Loop Step2 $L$ times, where $L$ is the number of transformer layers.
- **Step4**: Make the final prediction with the query vector and the weight vector, writing the scalar into the label coordinate of the output token
$$\hat{y}_{\text{query}} = \mathbf{w}^{*\top} \mathbf{x}_{\text{query}}$$

## 3. How is the transformer trained?

The transformer is trained across a massive distribution of different synthetic regression tasks  using mean squared error (MSE) loss

$$\min_{\theta} \mathbb{E}_{f \sim p(f), \; \mathbf{x}_1, \dots, \mathbf{x}_n \sim p(\mathbf{x})} \left[ \sum_{i=1}^n \mathcal{L}\Big(f(\mathbf{x}_i), \; T_\theta([\tilde{\mathbf{x}}_1, \tilde{\mathbf{y}}_1, \dots, \tilde{\mathbf{x}}_i])\Big) \right]$$

Each regression dataset is sampled on the fly:

- Task / Weight Vector Sampling ($p(f)$)
$$
$\mathbf{w}^* \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)
$$
- Feature Input Sampling ($p(\mathbf{x})$)
$$
\mathbf{x}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d) \quad \text{or} \quad \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$
- Label Generation (with optional noise)
$$
y_i = \mathbf{w}^{*\top} \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$
- Sequence Construction
$$[\tilde{\mathbf{x}}_1, \tilde{\mathbf{y}}_1, \tilde{\mathbf{x}}_2, \tilde{\mathbf{y}}_2, \dots, \tilde{\mathbf{x}}_n, \tilde{\mathbf{y}}_n]$$
Note: the error loss is back propagated on the $\tilde{\mathbf{x}}_i$ positions, but not the $\tilde{\mathbf{y}}_i$ token positions
## 4. Takeaways

- The transformer is a meta-learner, instead of being a standard supervised learning model that learns a function $f: \mathcal{X} \to \mathcal{Y}$ a.k.a 
$$\mathbf{x} \mapsto \hat{y} \quad (\text{Parameters encode } \mathbf{w})$$, it learns a mapping from an entire training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ and an arbitrary query point $\mathbf{x}$ to a prediction $$\mathcal{A}: (\mathcal{D}, \mathbf{x}) \mapsto \hat{y} \quad (\text{Parameters encode an algorithm } \mathcal{A})$$
- Large language models (and transformers trained on sequence prediction) are the modern expression of **model-based meta-learners**
- Meta learning usually have two loops, the task adaptation **inner loop** takes one single standard gradient descent step
  $$\theta'_1 = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{D}_{\text{supp}}}(\theta)$$
the meta-optimization **outer loop** updates $\theta$ using the test loss (computed using $\theta'_1$)
$$\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\mathcal{D}_{\text{query}}}(\theta'_1)$$
At inference time, the sampled input only changes the adapted parameters ($\theta'$) via the inner loop gradient step. 