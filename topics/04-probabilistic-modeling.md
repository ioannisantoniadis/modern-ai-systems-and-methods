# Probabilistic Modeling

## Core Idea

Probabilistic modeling represents uncertainty explicitly. Instead of only producing a prediction, it defines a probability model for data, hidden variables, parameters, or outcomes.

## Essential Vocabulary

- Random variable: quantity with uncertain value.
- Distribution: probabilities over possible values.
- Likelihood: probability of observed data under parameters.
- Prior: belief about parameters before seeing data.
- Posterior: updated belief after seeing data.
- Evidence: normalizing probability of the observed data.
- Marginalization: summing or integrating out unknown variables.

## Bayes Rule

Bayesian inference updates beliefs:

```text
posterior proportional to likelihood times prior
```

In symbols:

```text
p(theta | data) = p(data | theta) p(theta) / p(data)
```

## Generative Versus Discriminative

- Generative model: models how data and labels are generated, such as `p(x, y)` or `p(x | y)`.
- Discriminative model: models the decision boundary or label probability directly, such as `p(y | x)`.

Naive Bayes is generative. Logistic regression is discriminative.

## Why It Matters

Probabilistic modeling helps with:

- Noisy labels.
- Missing data.
- Latent structure.
- Small data with prior knowledge.
- Uncertainty-aware decisions.
- Calibration and risk.

## Common Models

- Bernoulli, categorical, Gaussian, Poisson.
- Naive Bayes.
- Gaussian mixture models.
- Bayesian linear regression.
- Probabilistic matrix factorization.
- Topic models.

## Interview Check

Be able to explain the difference between likelihood and posterior. Likelihood treats parameters as inputs and asks how probable the data is. Posterior treats data as observed and asks how plausible the parameters are after seeing it.
