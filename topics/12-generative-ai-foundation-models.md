# Generative AI and Foundation Models

## Core Idea

Generative models learn to produce data: text, images, audio, video, code, structured outputs, or actions. Foundation models are large pretrained models adapted to many downstream tasks.

## Main Families

- Autoregressive models: generate one token or unit at a time.
- Diffusion models: learn to denoise from noise to data.
- VAEs: probabilistic latent variable models with neural encoders and decoders.
- GANs: generator and discriminator trained adversarially.
- Flow models: invertible transformations with tractable likelihoods.

## LLM Concepts

- Tokenization.
- Next-token prediction.
- Pretraining.
- Instruction tuning.
- Preference tuning.
- RAG.
- Tool use.
- Evaluation and alignment.

## Multimodal Models

Modern AI systems often combine text, image, audio, video, and actions through shared representations, contrastive training, cross-attention, or modality-specific encoders.

## Evaluation

Generative evaluation differs from classification:

- Factuality.
- Helpfulness.
- Groundedness.
- Safety.
- Diversity.
- Human preference.
- Task success.

## Connections

- VAEs connect GenAI to probabilistic latent variable modeling.
- Diffusion connects generation to iterative denoising.
- RAG connects LLMs to retrieval.
- RLHF connects GenAI to preference learning and RL.

## Interview Check

Be able to explain why LLM evaluation is hard: many outputs can be acceptable, surface metrics can miss factuality or usefulness, and automated judges need validation against human preferences.
