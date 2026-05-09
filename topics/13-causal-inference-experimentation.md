# Causal Inference and Experimentation

## Core Idea

Causal inference asks what would happen under an intervention. Prediction asks what is likely given observed patterns. These are related but not the same.

## Essential Concepts

- Treatment.
- Outcome.
- Confounder.
- Counterfactual.
- Average treatment effect.
- Selection bias.
- Propensity score.
- Instrumental variable.

## Randomized Experiments

A/B tests are the cleanest product causal design when randomization is feasible.

Key issues:

- Unit of randomization.
- Sample ratio mismatch.
- Interference.
- Novelty effects.
- Guardrail metrics.
- Multiple testing.

## Observational Methods

- Matching.
- Propensity weighting.
- Difference-in-differences.
- Regression discontinuity.
- Instrumental variables.
- Causal graphs and adjustment sets.

## Uplift And Policy Learning

Sometimes the goal is not predicting who will churn, but predicting who will respond to an intervention. This is treatment effect modeling.

## Connections

- Counterfactual evaluation appears in recommender systems and ads.
- Experimentation is production causal inference.
- RL and bandits also reason about interventions and policies.

## Interview Check

Be able to explain why a churn model can fail as an intervention tool: it predicts risk, but the best target is users whose outcome changes because of the intervention.
