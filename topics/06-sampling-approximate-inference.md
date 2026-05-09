# Sampling and Approximate Inference

## Core Idea

Inference asks for probabilities or expectations under a model. Exact inference is often impossible because the space of hidden variables or parameters is too large, so we approximate.

## Monte Carlo

Monte Carlo estimates quantities by sampling:

```text
expected value approximately equals average over samples
```

The core idea is simple: if samples come from the right distribution, averages over samples estimate expectations under that distribution.

## MCMC

Markov Chain Monte Carlo builds a Markov chain whose stationary distribution is the target distribution.

Methods to recognize:

- Metropolis-Hastings.
- Gibbs sampling.
- Hamiltonian Monte Carlo.

## Metropolis-Hastings

High-level flow:

1. Start at a current state.
2. Propose a new state.
3. Accept it with a probability based on how plausible it is relative to the current state.
4. Repeat until samples approximate the target distribution.

## Gibbs Sampling

Gibbs sampling repeatedly samples one variable at a time from its conditional distribution given all other variables. It is useful when conditionals are easier than the joint distribution.

## Variational Inference

Variational inference turns inference into optimization. It chooses a simpler distribution family and finds the member closest to the true posterior.

Tradeoff:

- MCMC can be asymptotically accurate but slow.
- Variational inference is often faster but biased by the approximation family.

## Importance Sampling

Importance sampling estimates expectations under one distribution using samples from another distribution, weighted by probability ratios. It is powerful but can have high variance.

## Interview Check

Be able to say why sampling exists: many useful probabilistic models define distributions that are easy to write down but hard to sum, integrate, or normalize exactly.
