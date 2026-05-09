# Deep Learning Foundations

## Core Idea

Deep learning uses layered differentiable function approximators trained with gradient-based optimization. Its power comes from learned representations, scale, and composability.

## Essentials

- Backpropagation.
- Gradient descent variants.
- Activation functions.
- Initialization.
- Normalization.
- Regularization.
- Embeddings.
- Attention.

## Major Architectures

- MLPs for generic function approximation.
- CNNs for spatial structure.
- RNNs and LSTMs for sequences.
- Transformers for attention-based sequence modeling.
- Graph neural networks for relational data.

## Training Concepts

- Loss landscapes.
- Overfitting and regularization.
- Dropout, weight decay, early stopping.
- Batch normalization and layer normalization.
- Transfer learning and fine-tuning.
- Scaling laws and data quality.

## What Can Go Wrong

- Data leakage.
- Shortcut learning.
- Poor calibration.
- Distribution shift.
- Spurious correlations.
- Expensive inference.
- Hard-to-debug failures.

## Connections

- Deep learning powers modern representation learning, GenAI, ranking, vision, speech, and multimodal systems.
- Probabilistic thinking still matters for losses, uncertainty, generative models, and calibration.
- Systems thinking matters because model size, latency, and serving cost shape what can ship.

## Interview Check

Be able to explain attention: each token builds a weighted combination of other token representations, where weights come from query-key similarity and values provide the information being mixed.
