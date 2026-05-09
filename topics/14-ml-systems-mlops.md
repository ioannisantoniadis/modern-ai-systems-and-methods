# ML Systems and MLOps

## Core Idea

ML systems turn models into reliable products. The hard parts are often data, evaluation, serving, monitoring, ownership, and feedback loops.

## System Components

- Data ingestion.
- Label generation.
- Feature pipelines.
- Training.
- Evaluation.
- Model registry.
- Serving.
- Logging.
- Monitoring.
- Retraining.
- Rollback.

## Production Failure Modes

- Data drift.
- Label drift.
- Train/serve skew.
- Feature freshness issues.
- Silent dependency failures.
- Latency regressions.
- Miscalibration.
- Logging bugs.
- Feedback loops.

## Evaluation Layers

- Unit tests for feature and metric code.
- Offline validation.
- Slice analysis.
- Shadow tests.
- A/B tests.
- Guardrail monitoring.
- Post-launch audits.

## Inference Optimization

- Quantization.
- Distillation.
- Pruning.
- Batching.
- Caching.
- Approximate nearest neighbor search.
- Hardware acceleration.

## Interview Check

Be able to walk through a production model regression systematically: metric, data, labels, features, model, serving, evaluation, and product context.
