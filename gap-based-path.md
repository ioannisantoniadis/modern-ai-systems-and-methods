# Gap-Based Path

This path assumes you are already comfortable with logistic regression, tree ensembles, ranking, recommender systems, deep learning basics, and GenAI concepts.

## Phase 1: Probabilistic Thinking

Read:

1. `topics/04-probabilistic-modeling.md`
2. `topics/05-graphical-latent-variable-models.md`
3. `topics/06-sampling-approximate-inference.md`
4. `topics/16-conformal-prediction.md`

Be able to explain:

- Likelihood versus posterior.
- Prior versus regularization.
- Generative versus discriminative modeling.
- Why inference becomes hard.
- How EM, MCMC, and variational inference differ.
- Why conformal prediction's coverage guarantee doesn't require a probabilistic model at all.

Practice prompts:

- "How would you model missing labels probabilistically?"
- "What is a latent variable model?"
- "Why do we need sampling?"
- "How would you add a reliability guarantee to a model you can't retrain as Bayesian?"

## Phase 2: State, Sequences, And Dynamics

Read:

1. `topics/09-sequence-time-series-state-models.md`
2. `topics/10-reinforcement-learning-bandits.md`

Be able to explain:

- Markov assumption.
- HMMs and hidden state.
- State-space models.
- MDPs, value functions, policies, rewards.
- Bandits versus full RL.

Practice prompts:

- "How is a Markov chain different from an HMM?"
- "When would you use a bandit instead of supervised learning?"
- "What makes offline RL difficult?"

## Phase 3: Representation Learning Beyond Standard Embeddings

Read:

1. `topics/07-representation-learning-autoencoders.md`
2. `topics/12-generative-ai-foundation-models.md`

Be able to explain:

- PCA versus autoencoders.
- Reconstruction objectives.
- VAE latent spaces.
- Contrastive learning.
- Diffusion versus autoregressive generation.

Practice prompts:

- "When are autoencoders useful?"
- "How does a VAE differ from a standard autoencoder?"
- "What is self-supervised learning?"

## Phase 4: Causal And Systems Thinking

Read:

1. `topics/13-causal-inference-experimentation.md`
2. `topics/14-ml-systems-mlops.md`
3. `topics/15-responsible-private-robust-ai.md`

Be able to explain:

- Prediction versus causation.
- Confounding.
- Treatment effects.
- Drift and monitoring.
- Privacy, fairness, robustness, and interpretability.

Practice prompts:

- "Why can a predictive churn model fail as an intervention policy?"
- "How do you know a model caused a metric improvement?"
- "What does production readiness mean for ML?"
