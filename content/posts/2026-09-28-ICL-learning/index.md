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

- **Step2**: Forward inference from left to right equals one step of gradient descent — the current prediction is performed (MLP/Attention), residual error computed (Causal Attention/Residual), gradient aggregated (Self-Attention Pooling), and weight vector updated (Residual Stream Addition).
- **Step3**: Loop Step 2 $L$ times, where $L$ is the number of transformer layers.
- **Step4**: Make the final prediction with the query vector and the weight vector.

## 3. Key Mathematical Formulations


## 4. Personal Insights

