# Representation Learning and Autoencoders

## Core Idea

Representation learning discovers useful features from data. Instead of hand-designing every feature, the model learns embeddings or latent variables that preserve task-relevant structure.

## Classical Foundations

- PCA: linear dimensionality reduction maximizing variance.
- Matrix factorization: decomposes interactions into latent factors.
- Word2Vec-style embeddings: learn vectors from co-occurrence or prediction tasks.

## Autoencoders

An autoencoder learns:

```text
input -> encoder -> latent code -> decoder -> reconstruction
```

The training objective is usually reconstruction quality. The latent code can be used as a compact representation.

## Variants

- Denoising autoencoder: reconstructs clean input from corrupted input.
- Sparse autoencoder: encourages few active latent dimensions.
- Variational autoencoder: learns a probabilistic latent space and can generate samples.
- Contrastive learning: learns representations by pulling related examples together and pushing unrelated examples apart.

## Uses

- Dimensionality reduction.
- Pretraining.
- Anomaly detection.
- Compression.
- Embeddings for retrieval or downstream prediction.
- Generative modeling through VAEs.

## Connections

- Autoencoders connect supervised deep learning to unsupervised representation learning.
- VAEs connect autoencoders to probabilistic latent variable models.
- Contrastive learning connects representation learning to retrieval and modern multimodal models.

## Interview Check

Be able to compare PCA and autoencoders: PCA is linear, deterministic, and easier to interpret; autoencoders can learn nonlinear representations but require more data and care.
