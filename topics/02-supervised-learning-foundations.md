# Supervised Learning Foundations

## Core Idea

Supervised learning estimates a mapping from inputs to labels using examples. It is the backbone of classification, regression, ranking features, risk scoring, forecasting baselines, and many production ML systems.

## Key Concepts

- Empirical risk minimization.
- Bias-variance tradeoff.
- Loss functions.
- Regularization.
- Feature engineering.
- Generalization.
- Calibration.
- Class imbalance.

## Main Methods

- Linear regression and ridge/lasso.
- Logistic regression.
- Generalized linear models.
- SVMs.
- k-nearest neighbors.
- Neural networks.

## When To Use

Use supervised learning when:

- Labels are available and meaningful.
- The decision surface is well-defined.
- Historical examples are representative enough.
- You can evaluate with held-out data and product metrics.

## What Can Go Wrong

- Labels are biased proxies.
- Train/test split leaks future information.
- Class imbalance hides failures.
- Model is accurate but poorly calibrated.
- Offline objective does not match deployment action.

## Connections

- Logistic regression connects to probabilistic modeling through Bernoulli likelihood.
- Regularization connects to Bayesian priors.
- Neural networks extend supervised learning with learned representations.
- Ranking can be built from pointwise supervised scoring or pairwise/listwise objectives.

## Interview Check

Be able to explain why logistic regression is still useful: it is fast, interpretable, easy to calibrate, strong on sparse features, and a good baseline for production systems.
