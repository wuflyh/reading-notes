---
title: "LTRR: Learning To Rank Retrievers for LLMs"
date: 2026-11-08T11:53:46-08:00
tags:
  - LTR
  - RAG
categories:
  - Retrieval & Ranking
math: true
draft: false
meta:
  authors: [To Eun Kim et al.]
  venue: "SIGIR"
  url: "https://arxiv.org/abs/2506.13743"
  year: 2026
  month: 04
  rating: 3
---
{{< paper >}}

## 1. Summary

The paper introduced LTRR which is trained using learning-to-rank algorithms to routes queries to the top retrievers in the RAG systems.  Their empirical results show that RAG systems equipped with trained LTRR (**using the AC utility metric and pairwise learning-to-rank algorithms is the best**) can outperform standard single-retriever RAG systems and generalize to unseen query types. 

## 2. Method

LTRR establishes a router function $\mathcal{F}$ that scores and ranks available candidate retrievers $\mathcal{R} = \{\mathcal{R}_0, \mathcal{R}_1, \dots, \mathcal{R}_M\}$ for a given query $q$, in which $\mathcal{R}_0$ is "No-Retrieval".

Given query $q$ and ground truth generation $y$, LTRR measures the empirical utility gain $\delta_i$ achieved by retriever $\mathcal{R}_i$ relative to the "No-Retrieval" baseline:

$$\delta_i = \mu_u\big(G(x_i), y\big) - \mu_u\big(G(x), y\big)$$
where:

- $x_i$ represents the retrieval results by retriever $\mathcal{R}_i$ with query $q$. Specifically, $x$ corresponds to 
  "No-Retrieval".
  
- $G(x)$ is the response generated using only parametric memory (no retrieval).
    
- $G(x_i)$ is the response generated using passages retrieved by retriever $\mathcal{R}_i$.
    
- $y$ is the ground-truth target answer.
    
- $\mu_u$ is a downstream utility metric (e.g., **BERTScore/BEM or Answer Correctness / AC**).

Then, The gains $\delta_i$ are **min-max normalized** to ranking score $$y_i = \frac{\delta_i - \min_j(\delta_j)}{\max_j(\delta_j) - \min_j(\delta_j)}$$per query into $[0, 1]$ to serve as the **ground-truth ranking targets**.

 Next,  router feature vector $\Phi(q, \mathcal{R}_i)$ is devised with Pre-Retrieval Features(Query length, lexical ambiguity, semantic query embeddings, and question category/type) and Post-Retrieval Features (Aggregated retrieval scores, score variance, intra-list diversity, and Cross-Retriever Similarity)

Lastly, learn the scoring function $f(\Phi(q, \mathcal{R}_i))$ with standard LTR paradigms

On inference, the system takes a query $q$ invokes multiple retrievers at once, compiles the feature $\Phi(q, \mathcal{R}_i)$, generates scores $s_i = f(\Phi(q, \mathcal{R}_i))$ to rank and select the retrievers, sends the retrievals to the generator to get the ultimate answer.  
## 3. The Three LTR Approaches

#### Pointwise LTR
It fits MSE loss directly against $y_i$ (e.g., via XGBoost / FFN) with loss function $$\mathcal{L} = \frac{1}{2}\sum_{i=1}^N (s_i - y_i)^2$$ with the gradients
- **First derivative ($g_i$):** $g_i = s_i - y_i$ 
- **Second derivative ($h_i$):** $h_i = 1$ 
with code snippet
```python
import xgboost as xgb

# X: feature matrix (N, D), y: relevance scores (N,)
model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=100, learning_rate=0.1
)
model.fit(X, y)

# Predict continuous scores and sort per query at inference
scores = model.predict(X_test)
```
#### Pairwise LTR
It minimizes ranking inversions between retrievers (e.g., Pairwise XGBoost / SVMRank).
For every pair $(i, j)$ in query $q$ where item $i$ is more relevant than item $j$ ($y_i > y_j$):

$$\mathcal{L}_q = \sum_{i, j : y_i > y_j} \log\left(1 + e^{-\sigma(s_i - s_j)}\right)$$
with the gradients
- **Pairwise Gradient ($\lambda_{ij}$):**
    
    $$\frac{\partial \mathcal{L}_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} = -\sigma (1 - P_{ij})$$
    
    where $P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$.
- **Accumulating Pointwise Residuals:** Each row $i$ aggregates its pushes against all other candidates in its query:

$$g_i = \sum_{j: y_i > y_j} -\sigma (1 - P_{ij}) - \sum_{k: y_k > y_i} -\sigma (1 - P_{ki})$$
$$h_i = \sum_{j: y_i \neq y_j} \sigma^2 P_{ij}(1 - P_{ij})$$
with code snippet
```python
model = xgb.XGBRanker(
    objective="rank:pairwise",
    n_estimators=100,
    learning_rate=0.1,
    lambdarank_pair_method="mean",
)
# qid indicates query boundaries (must be sorted by qid)
model.fit(X, y, qid=qid)
```
#### Listwise LTR
It optimizes listwise ranking surrogate objectives over retriever orderings
There is no a closed-form loss $\mathcal{L}$. It scales the pairwise RankNet gradients by the absolute change in the listwise metric resulting from swapping items $i$ and $j$:
$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot \vert{}\Delta\text{NDCG}_{ij}\vert{}$$where the swap penalty depends dynamically on the current predicted ranks $r_i$ and $r_j$:

$$\vert{}\Delta\text{NDCG}_{ij}\vert{} = \frac{1}{\text{IDCG}} \cdot \left\vert{} 2^{y_i} - 2^{y_j} \right\vert{} \cdot \left\vert{} \frac{1}{\log_2(r_i + 1)} - \frac{1}{\log_2(r_j + 1)} \right\vert{}$$ and here is the tree construction
- Documents in query $q$ are sorted by current predicted score $s^{(t-1)}$ to find current ranks $r_i$.
    
- Pairwise forces $\lambda_{ij}$ and curvatures $h_{ij}$ are evaluated.
    
- Row gradients are accumulated:
    
    $$g_i = -\sum_{j: y_i > y_j} \lambda_{ij} + \sum_{k: y_k > y_i} \lambda_{ki}$$
    
    $$h_i = \sum_{j: y_i \neq y_j} \sigma^2 P_{ij}(1 - P_{ij}) \cdot \vert{}\Delta\text{NDCG}_{ij}\vert{}$$
    
- A standard regression tree is built to fit $g_i$ and $h_i$, with optimal leaf values:
    
    $$w^* = -\frac{\sum_{i \in \text{leaf}} g_i}{\sum_{i \in \text{leaf}} h_i + \lambda_{\text{reg}}}$$
with code snippet
```python
model = xgb.XGBRanker(
    objective="rank:ndcg",
    eval_metric="ndcg@5",
    n_estimators=100,
    learning_rate=0.1,
    lambdarank_pair_method="topk",  # Focuses pair sampling on top ranks
    lambdarank_num_pair_per_sample=8,
)
model.fit(X, y, qid=qid, eval_set=[(X_val, y_val)], eval_qid=[qid_val])
```


