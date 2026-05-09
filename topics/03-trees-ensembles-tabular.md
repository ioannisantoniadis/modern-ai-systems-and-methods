# Trees, Ensembles, and Tabular ML

## Core Idea

Tree-based methods split feature space into regions. Ensembles combine many weak trees to reduce variance, bias, or both. They remain extremely strong for structured/tabular data.

## Main Methods

- Decision trees.
- Random forests.
- Gradient boosted trees.
- XGBoost, LightGBM, CatBoost.
- ExtraTrees and bagging variants.

## Why They Work Well

- Capture nonlinear feature interactions.
- Handle mixed feature types.
- Require less feature scaling.
- Work well with medium-sized tabular datasets.
- Provide feature importance and debugging hooks.

## Tradeoffs

- Less natural for raw text, images, audio, or long sequences.
- Can be poorly calibrated without post-processing.
- Large ensembles can be harder to serve on device.
- Extrapolate poorly outside the training feature range.

## Connections

- GBDTs often serve as strong baselines before deep models.
- Tree leaf indices can become embeddings or features for downstream models.
- LambdaMART and related boosted methods are central to learning-to-rank.

## Interview Check

Be able to compare logistic regression, GBDT, and deep neural networks for tabular product data. A good answer should mention data size, interactions, latency, interpretability, calibration, and feature availability.
