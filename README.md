# Modern AI Systems and Methods

A map-style learning repository for the main directions, domains, and techniques in modern machine learning and AI.

The goal is not to replace textbooks. The goal is to build a practical mental map: what each method is, when to use it, how it connects to other methods, what assumptions it makes, and what interview or research questions might expose gaps.

## How To Use This Repo

Use the topic spine first, then deepen selectively:

1. Read `learning-map.md` to understand the full landscape.
2. Use `gap-based-path.md` if you already know supervised learning, ranking, deep learning, recommender systems, and GenAI but want to fill probabilistic modeling, sampling, RL, and state-model gaps.
3. Read topic pages in order when you want broad coverage.
4. Use `interview-checkpoints.md` to test whether you can explain each area under pressure.
5. Add deeper notes with `templates/topic-note.md` and `templates/method-card.md`.

## Topic Spine

| Area | File |
| --- | --- |
| AI and ML landscape | `topics/01-ai-ml-landscape.md` |
| Supervised learning foundations | `topics/02-supervised-learning-foundations.md` |
| Trees, ensembles, and tabular ML | `topics/03-trees-ensembles-tabular.md` |
| Probabilistic modeling | `topics/04-probabilistic-modeling.md` |
| Graphical models and latent variables | `topics/05-graphical-latent-variable-models.md` |
| Sampling and approximate inference | `topics/06-sampling-approximate-inference.md` |
| Representation learning and autoencoders | `topics/07-representation-learning-autoencoders.md` |
| Deep learning foundations | `topics/08-deep-learning-foundations.md` |
| Sequence, time-series, and state models | `topics/09-sequence-time-series-state-models.md` |
| Reinforcement learning and bandits | `topics/10-reinforcement-learning-bandits.md` |
| Information retrieval, ranking, and recommendations | `topics/11-ir-ranking-recommenders.md` |
| Generative AI and foundation models | `topics/12-generative-ai-foundation-models.md` |
| Causal inference and experimentation | `topics/13-causal-inference-experimentation.md` |
| ML systems and MLOps | `topics/14-ml-systems-mlops.md` |
| Responsible, private, and robust AI | `topics/15-responsible-private-robust-ai.md` |

## Mental Model

Most AI methods can be located by five questions:

- What is observed, and what is hidden?
- Is the goal prediction, generation, decision-making, discovery, or causal estimation?
- Are labels available, weak, delayed, biased, or missing?
- Is uncertainty central or incidental?
- Does the method need to run as a product system with latency, privacy, monitoring, and feedback loops?

## Gap-Filling Priorities

Given a background in ranking, deep learning, recommender systems, GenAI, logistic regression, and tree ensembles, the highest-leverage gaps are:

1. Probabilistic modeling: Bayes rule, likelihoods, priors, posteriors, uncertainty, latent variables.
2. Sampling and approximate inference: Monte Carlo, MCMC, Gibbs, Metropolis-Hastings, variational inference.
3. State and sequence models: Markov chains, HMMs, state-space models, time-series forecasting.
4. Reinforcement learning: MDPs, value functions, Q-learning, policy gradients, bandits, offline RL.
5. Representation learning beyond standard embeddings: PCA, matrix factorization, autoencoders, VAEs, contrastive learning.
6. Causal inference: treatment effects, confounding, causal graphs, counterfactuals.

## Repository Philosophy

Each topic should answer:

- What is the idea?
- What problem does it solve?
- What assumptions does it make?
- How does it relate to other methods?
- What should I be able to derive or explain?
- Where does it show up in modern AI systems?

## License

MIT License. See `LICENSE`.
