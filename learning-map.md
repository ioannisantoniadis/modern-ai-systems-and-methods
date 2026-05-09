# Learning Map

This map organizes modern AI by problem type, not by hype cycle.

## 1. Prediction From Labeled Data

Core methods:

- Linear regression, logistic regression, GLMs.
- Trees, random forests, gradient boosting.
- Neural networks and deep supervised learning.
- Calibration, uncertainty, class imbalance, feature engineering.

Use when: labels are available and the task is prediction, classification, ranking, or regression.

## 2. Modeling Uncertainty And Hidden Structure

Core methods:

- Bayesian inference.
- Probabilistic graphical models.
- Mixture models.
- Latent variable models.
- EM, variational inference, MCMC.

Use when: uncertainty, missing data, hidden states, noisy labels, or generative assumptions matter.

## 3. Learning Representations

Core methods:

- PCA and matrix factorization.
- Embeddings.
- Autoencoders and VAEs.
- Contrastive learning.
- Self-supervised learning.

Use when: raw data is high-dimensional, labels are sparse, or reusable features are needed.

## 4. Modeling Sequences And State

Core methods:

- Markov chains.
- HMMs.
- State-space models.
- RNNs, LSTMs, transformers.
- Time-series forecasting and anomaly detection.

Use when: order, dynamics, temporal dependence, or hidden state drives behavior.

## 5. Making Decisions

Core methods:

- Bandits.
- Reinforcement learning.
- MDPs.
- Q-learning and policy gradients.
- Offline RL and RLHF.

Use when: actions influence future states or when exploration and exploitation trade off.

## 6. Retrieval, Ranking, And Recommendation

Core methods:

- Lexical retrieval.
- Dense retrieval.
- Learning to rank.
- Collaborative filtering.
- Sequential and generative recommendation.

Use when: the system selects or orders candidates from a large set.

## 7. Generating Data Or Content

Core methods:

- Autoregressive models.
- Diffusion models.
- VAEs and GANs.
- LLMs and multimodal foundation models.

Use when: the task is to synthesize text, images, audio, code, actions, or structured outputs.

## 8. Estimating Causes

Core methods:

- Randomized experiments.
- Causal graphs.
- Potential outcomes.
- Propensity scores.
- Instrumental variables.
- Difference-in-differences.

Use when: "what caused what?" matters more than prediction.

## 9. Operating ML Systems

Core methods:

- Data pipelines.
- Feature stores.
- Training infrastructure.
- Serving and inference optimization.
- Monitoring, drift, evaluation, rollback.

Use when: models must be reliable, observable, cost-effective, and safe in production.

## 10. Building Responsible AI

Core methods:

- Privacy-preserving ML.
- Fairness and bias analysis.
- Interpretability.
- Robustness.
- Safety and security.

Use when: model behavior affects users, institutions, or high-stakes decisions.
