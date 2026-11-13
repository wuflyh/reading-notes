---
title: Learning to Rank in Generative Retrieval
date: 2026-11-11T06:27:18-08:00
tags:
  - LTR
  - GR
  - DPR
  - Beam-Search
  - FM-index
categories:
  - Retrieval & Ranking
math: true
draft: false
meta:
  authors:
    - Yongqi Li et al.
  venue: AAAI
  url: https://arxiv.org/abs/2306.15222
  year: 2023
  month: 12
  rating: 3
---
{{< paper >}}

## 1. Summary

**Generative retrieval** trains a sequence-to-sequence model (e.g., BART or T5) to generate document identifiers (DocIDs)—such as titles, numeric IDs, or multiview substrings—directly from a query. However, the generated list of DocIDs does not inherently optimize the top-$k$ passage rank order. **LTRGR** (a learning-to-rank framework for generative retrieval) decouples the training process into two distinct stages: learning to generate and learning to rank to address the issue. Accordingly, BART-Large LTRGR reached **40.2 Recall@5** compared to **29.5** for BART-Large MINDER (+10.7 points) he retrieval process and inference latency remain identical to standard beam-search generative retrieval., showing solid gains across **Recall@5/20/10**0 and **MRR@10**. and, the inference latency remains the same as the standard beam-search generative retrieval. 

While LTRGR set a new state of the art among generative retrieval methods, it still trailed dense retrieval baselines (like **DPR**) at tight cutoffs (e.g., Hits@5/20 on TriviaQA) due to compounding token errors in beam search and lack of direct passage content visibility.

## 2. Method
#####  Stage1: Learning to Generate

**LTRGR** model is pre-trained with standard autoregressive generation loss over target **identifiers** (building upon the **MINDER** framework using multiview identifiers like **titles, substrings, and pseudo-queries**)

The loss function is

$$
\mathcal{L}_{\text{gen}} = -\sum_{t=1}^{\vert{}y^+\vert{}} \log P_\theta(y_t^+ \mid y_{\lt t}^+, q)
$$

where $y^+ = \text{DocID}(d^+)$ is the sequence of tokens representing the relevant document **identifier**.
##### Stage2: Learning to Rank

Next the pretrained **LTRGR** model takes a query, generates a list of identifiers $I_1, I_2, ..., I_b$ using beam-search. For any candidate DocID $I = (y_1, \dots, y_L)$ is extracted directly from the sequence generation probability:

$$
s(q, I) = \log P_\theta(I \mid q) = \sum_{t=1}^L \log P_\theta(y_t \mid y_{\lt t}, q)
$$

As identifiers $Y$ and passages $D$ are many-to-many mapping, **LTRGR** model maps the list of identifiers to a list of distinctive passages $P_1, P_2, ..., P_n$. For each passage $P$ that aggregates a list of identifiers:

$$
s(q, P) = \sum_{I \in I_P} s(q, I)
$$

Where $I_P$ is the set of selected identifiers that appear in the passage $P$.

Now with the list $q, P_1, P_2, ..., P_n$ including a positive passage $P_+$ and other negative passages $P_-$ , the **LTRGR** model is trained to minimize the two pairwise ranking losses:

- $\mathcal{L}_{\text{rank1}}$ (**Hard Negative Mining**): Selects the **top-2 scoring passages** from the model's beam search or candidate list
- $\mathcal{L}_{\text{rank2}}$ (**Random / Soft Negative Exploration**): Selects **2 passages uniformly at random** from the broader candidate pool *(random selection for negative only)*

The loss function is

$$
\mathcal{L}_{\text{rank}} = \max\bigl(0,\; s(q, P_-) - s(q, P_+) + m\bigr)
$$

so the overall loss function is:

$$
\mathcal{L} = \mathcal{L}_{\text{rank}_1} + \mathcal{L}_{\text{rank}_2} + \lambda\, \mathcal{L}_{\text{gen}}
$$

but training is split into two stages.
![](learning-to-generate-and-learning-to-rank.png)
## 3. Implementation Details

##### Multiview Identifiers Represent Each Passage

Each passage $P$ in the corpus is represented from **three distinct semantic views/identifiers**:

- **Title:** The document or article title. e.g. `Apollo 11`

- **Substring:** A salient n-gram or substring extracted directly from the passage text. e.g. `Neil Armstrong was an American astronaut`

- **Pseudo-query:** A synthetic query generated from the passage. e.g. - `who was the first moon landing commander`
##### The Training Dataset in LTRGR
Because there are multiple views $I_+$ for each passage, the training set expands each positive query-passage pair $(q, P_+)$ into multiple training instances:
$$\text{Input: } [\text{View Prefix}] \circ q \quad \longrightarrow \quad \text{Target: } \text{Identifier String } I_+$$
-  Example 1: `Title: who was the first person to walk on the moon?` $\longrightarrow$ `Apollo 11`
    
- Example 2: `Substring: who was the first person to walk on the moon?` $\longrightarrow$ `Neil Armstrong was an American astronaut`
    
- Example 3: `Query: who was the first person to walk on the moon?` $\longrightarrow$ `who was the first moon landing commander`
During training, these $(q, I_+)$ pairs are shuffled and fed to the **seq2seq** model (e.g., BART-Large) using standard cross-entropy loss.
#####  Ferragina-Manzini index

The **LTRGR** model performs beam search to generate predicted identifiers. The FM-index data structure is used to support generating valid identifiers. All identifiers of passages are stored in FM-index. Given a start token or a string, FM-index could provide the list of possible token successors, which force the model to generate valid identifiers via constrained generation.

Once beam search produces the top-$b$ valid identifier strings  $I_1, I_2, ..., I_b$ using beam-search,  the FM-index is queried again to locate every passage $P$ in the corpus that contains at least one of the generated identifiers $I_i$. (Because a substring or pseudo-query might appear in multiple passages, a single generated identifier can map to several candidate passages).
## 4. Personal Insights

#### How to Implement LambdaRank that bypasses an explicit scalar loss function and defines the $\lambda$-gradients directly using pytorch framework?

##### Method 1: The Vectorized Pairwise Surrogate Loss (Idiomatic PyTorch)

Recall the LambdaRank gradient for document $i$:

$$\lambda_i = \sum_{j : y_i > y_j} \lambda_{ij} - \sum_{k : y_k > y_i} \lambda_{ki}$$

where:

$$\lambda_{ij} = -\frac{\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot \vert{}\Delta\text{NDCG}_{ij}\vert{}$$

Notice that $\frac{\sigma}{1 + e^{\sigma(s_i - s_j)}}$ is the exact derivative of the pairwise logistic loss:

$$C_{ij} = \log\left(1 + e^{-\sigma(s_i - s_j)}\right)$$

If we treat $\vert{}\Delta\text{NDCG}_{ij}\vert{}$ as a **detached constant weight** computed from the current sorted ranks, the surrogate loss:

$$\mathcal{L}_{\text{surrogate}} = \sum_{i, j : y_i > y_j} \vert{}\Delta\text{NDCG}_{ij}\vert{} \cdot \log\left(1 + e^{-\sigma(s_i - s_j)}\right)$$

differentiates with respect to $s_i$ as:

$$\frac{\partial \mathcal{L}_{\text{surrogate}}}{\partial s_i} = \sum_{j : y_i > y_j} -\sigma(1 - P_{ij})\vert{}\Delta\text{NDCG}_{ij}\vert{} - \sum_{k : y_k > y_i} -\sigma(1 - P_{ki})\vert{}\Delta\text{NDCG}_{ki}\vert{} = -\lambda_i$$

Because standard gradient descent steps in the direction of $-\nabla \mathcal{L}$, running `loss.backward()` applies the exact LambdaRank update forces.

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaRankLoss(nn.Module):

    def __init__(self, sigma=1.0, k=None):
        super().__init__()
        self.sigma = sigma
        self.k = k  # Evaluation cutoff for NDCG@k (None means full list)

    def compute_idcg(self, y_true):
        """Computes Ideal DCG for a query."""
        sorted_labels, _ = torch.sort(y_true, descending=True)
        gains = torch.pow(2.0, sorted_labels) - 1.0
        ranks = torch.arange(
            1, len(sorted_labels) + 1, device=y_true.device, dtype=torch.float32
        )
        discounts = 1.0 / torch.log2(ranks + 1.0)

        if self.k is not None:
            gains = gains[: self.k]
            discounts = discounts[: self.k]

        return torch.sum(gains * discounts)

    def forward(self, scores, y_true):
        """Args:

        scores: Tensor of shape (N,) containing predicted real-valued margins
        y_true: Tensor of shape (N,) containing relevance labels (discrete or
        continuous)
        """
        N = scores.size(0)
        if N <= 1:
            return scores.sum() * 0.0

        idcg = self.compute_idcg(y_true)
        if idcg == 0.0:
            return scores.sum() * 0.0

        # 1. Determine current ordinal ranks based on predicted scores
        sorted_indices = torch.argsort(scores.detach(), descending=True)
        ranks = torch.zeros(N, device=scores.device, dtype=torch.float32)
        # ranks are 1-based: rank 1, 2, ..., N
        ranks[sorted_indices] = torch.arange(
            1, N + 1, device=scores.device, dtype=torch.float32
        )

        # 2. Vectorized pairwise index matrices: (i, j)
        # y_diff[i, j] > 0 means item i is strictly more relevant than item j
        y_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
        valid_pairs = y_diff > 0

        if not valid_pairs.any():
            return scores.sum() * 0.0

        # 3. Compute |ΔNDCG_ij| as a detached scaling constant
        gain_diff = torch.abs(
            (torch.pow(2.0, y_true.unsqueeze(1)) - 1.0)
            - (torch.pow(2.0, y_true.unsqueeze(0)) - 1.0)
        )

        rank_i = ranks.unsqueeze(1)
        rank_j = ranks.unsqueeze(0)
        discount_diff = torch.abs(
            (1.0 / torch.log2(rank_i + 1.0)) - (1.0 / torch.log2(rank_j + 1.0))
        )

        delta_ndcg = (gain_diff * discount_diff) / idcg
        delta_ndcg = delta_ndcg.detach()  # Disconnect from computation graph

        # 4. Pairwise logistic loss: log(1 + exp(-sigma * (s_i - s_j)))
        # PyTorch softplus(x) computes log(1 + exp(x)), so softplus(-sigma * (s_i - s_j))
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        pairwise_loss = F.softplus(-self.sigma * score_diff)

        # 5. Weighted listwise sum over valid preference pairs
        loss = torch.sum(pairwise_loss[valid_pairs] * delta_ndcg[valid_pairs])
        return loss
```
##### Method 2: Direct Lambda Gradient Injection (`scores.backward(gradient=lambdas)`)

If you want to construct the exact physical $\lambda_i$ vector without defining an artificial loss function, compute the net push/pull forces directly in Python and feed them into `backward()`

```python
def compute_lambdas(scores, y_true, sigma=1.0):
    """Computes the explicit net lambda vector for all documents in a query."""
    with torch.no_grad():
        N = scores.size(0)
        sorted_indices = torch.argsort(scores, descending=True)
        ranks = torch.zeros(N, device=scores.device, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(1, N + 1, device=scores.device)

        # Compute IDCG
        sorted_labels, _ = torch.sort(y_true, descending=True)
        idcg = torch.sum(
            (torch.pow(2.0, sorted_labels) - 1.0)
            / torch.log2(
                torch.arange(1, N + 1, device=scores.device, dtype=torch.float32)
                + 1.0
            )
        )
        if idcg == 0.0:
            return torch.zeros_like(scores)

        # Pairwise differences
        s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        p_ij = torch.sigmoid(sigma * s_diff)

        gain_diff = torch.abs(
            (torch.pow(2.0, y_true.unsqueeze(1)) - 1.0)
            - (torch.pow(2.0, y_true.unsqueeze(0)) - 1.0)
        )
        discount_diff = torch.abs(
            (1.0 / torch.log2(ranks.unsqueeze(1) + 1.0))
            - (1.0 / torch.log2(ranks.unsqueeze(0) + 1.0))
        )
        delta_ndcg = (gain_diff * discount_diff) / idcg

        # λ_ij = -sigma * (1 - P_ij) * |ΔNDCG_ij|
        lambda_matrix = -sigma * (1.0 - p_ij) * delta_ndcg

        # Filter strictly where y_i > y_j
        valid_pairs = (y_true.unsqueeze(1) - y_true.unsqueeze(0)) > 0
        lambda_matrix = torch.where(
            valid_pairs, lambda_matrix, torch.zeros_like(lambda_matrix)
        )

        # Net push on doc i: sum of forces where i beats j minus forces where k beats i
        # Since lambda_matrix has negative values representing gradient directions:
        lambdas = torch.sum(lambda_matrix, dim=1) - torch.sum(
            lambda_matrix, dim=0
        )

    # Return gradients pointing in the direction of minimization
    return -lambdas
```
##### Complete PyTorch Training Loop
Here is how you train a neural ranking model (MLP / Transformer scoring head) per query using the surrogate loss:
```python
# 1. Define a standard PyTorch scoring network
class NeuralRanker(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Scalar ranking score
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# 2. Training loop over grouped queries
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralRanker(input_dim=16).to(device)
criterion = LambdaRankLoss(sigma=1.0, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy dataset: list of queries, each query having (features, labels)
# X_q shape: (n_docs_in_query, feature_dim), y_q shape: (n_docs_in_query,)
query_batches = [
    (
        torch.randn(12, 16, device=device),
        torch.randint(0, 5, (12,), device=device).float(),
    )
    for _ in range(50)
]

model.train()
for epoch in range(5):
    total_loss = 0.0
    optimizer.zero_grad()

    for X_q, y_q in query_batches:
        # Forward pass: compute scalar scores for all candidates in the query
        scores = model(X_q)

        # Method 1 (Surrogate loss):
        loss = criterion(scores, y_q)
        loss.backward()

        # --- OR Method 2 (Direct gradient injection) ---
        # lambdas = compute_lambdas(scores, y_q)
        # scores.backward(gradient=lambdas)

        total_loss += loss.item()

    optimizer.step()
    print(f"Epoch {epoch+1} Complete - Query Loss: {total_loss:.4f}")
```
##### Architectural Notes

- **Grouping by Query is Mandatory:** Unlike pointwise classification where samples shuffle across a batch, Lambda gradients depend strictly on rank swaps within the **same query group**. Each forward/backward step must process query groups individually or use padded 3D tensors `(batch_size, num_candidates, feature_dim)`.
    
- **Detaching the Permutation:** In Method 1, computing `ranks` from `torch.argsort(scores.detach())` is crucial. The permutation is a non-differentiable discrete step; detaching ensures autograd only differentiates through the continuous `score_diff` terms.