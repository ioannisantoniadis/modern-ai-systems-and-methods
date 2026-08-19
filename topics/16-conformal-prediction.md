# Conformal Prediction

## Core Idea

Conformal prediction wraps around any predictive model to produce prediction sets or intervals with a finite-sample, distribution-free coverage guarantee. The guarantee holds even if the underlying model is wrong or poorly calibrated, as long as the data is exchangeable.

## How It Differs From Bayesian Uncertainty

- Bayesian inference requires a full probabilistic model of the data, and its uncertainty is only as good as that model's assumptions.
- Conformal prediction is a model-agnostic wrapper around any point predictor, including a black-box neural network, and its guarantee holds in finite samples under exchangeability, not correctness of the underlying model.

These are complementary, not competing: a Bayesian posterior can itself be conformalized to fix miscalibration.

## Mechanism

- Nonconformity score: a measure of how unusual a candidate prediction is, for example `|y - y_hat|` for regression or `1 - softmax probability of the true class` for classification.
- Calibration set: held-out labeled data used only to compute the distribution of nonconformity scores.
- Prediction set at level `1 - alpha`: every candidate label or value whose nonconformity score falls below the `(1 - alpha)` quantile of the calibration scores.

```text
new input -> candidate label -> nonconformity score -> compare to calibration quantile -> in or out of the prediction set
```

## Variants

- Split (inductive) conformal: one calibration set, cheap, the version used in almost all practical systems.
- Full conformal: refit the model once per candidate label, exact but computationally impractical at scale.
- Mondrian or class-conditional conformal: separate calibration per group, used when miscoverage needs to be balanced across classes or subgroups rather than only on average.
- Conformalized quantile regression: wraps quantile regression to produce tighter intervals than symmetric split conformal.
- Weighted or adaptive conformal: reweights calibration scores to handle covariate shift or time-series settings where exchangeability doesn't strictly hold.

## Assumptions And Failure Modes

- Relies on exchangeability between the calibration set and future data. Breaks under distribution shift, non-stationary time series, or a calibration set that's too small to estimate the quantile reliably.
- The guarantee is marginal, averaged over repeated draws, not conditional on any single input. An individual interval isn't guaranteed correct, only the long-run coverage rate.
- Conformal prediction guarantees validity, not informativeness. A wide, uselessly conservative interval still satisfies the coverage guarantee, so interval width is a separate quality axis from coverage.

## Production Relevance

Wrapping an existing production model conformally is far cheaper than a full Bayesian rewrite, since no retraining or generative reformulation is required. It's the usual choice when a system needs a hard, auditable reliability guarantee, for example flagging low-confidence predictions for human review, without access to a trusted posterior.

## Interview Check

Be able to explain why the coverage guarantee is distribution-free and finite-sample rather than asymptotic, what exchangeability actually buys you and a concrete setting where it breaks, and how split conformal trades exactness for cost against full conformal.
