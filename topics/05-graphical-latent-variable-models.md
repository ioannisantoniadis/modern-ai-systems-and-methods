# Graphical Models and Latent Variables

## Core Idea

Graphical models use graphs to represent conditional dependencies. Latent variable models explain observed data through hidden factors.

## Why They Matter

They provide a language for:

- Hidden user state.
- Missing data.
- Clustering.
- Topic discovery.
- Temporal dynamics.
- Structured uncertainty.

## Graphical Model Types

- Bayesian networks: directed graphs representing causal or generative dependencies.
- Markov random fields: undirected graphs representing compatibility between variables.
- Factor graphs: bipartite graphs connecting variables and factors.

## Latent Variable Examples

- Mixture model: hidden cluster generates each observation.
- HMM: hidden state evolves over time and emits observations.
- Topic model: hidden topic mixture generates words.
- Matrix factorization: hidden user and item factors explain ratings or interactions.
- VAE: neural latent variable model with amortized inference.

## EM Algorithm

Expectation-Maximization alternates:

1. E-step: infer expected hidden variables given current parameters.
2. M-step: update parameters given expected hidden assignments.

Use it when hidden variables make direct maximum likelihood difficult but conditional updates are tractable.

## Conditional Independence

Conditional independence assumptions make complex distributions manageable. They can also be wrong, so they should be treated as modeling assumptions, not facts.

## Interview Check

Be able to explain a Gaussian mixture model: each data point is assumed to come from one hidden cluster, each cluster has a Gaussian distribution, and EM alternates between soft cluster assignments and parameter updates.
