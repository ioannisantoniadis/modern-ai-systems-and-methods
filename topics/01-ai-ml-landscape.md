# AI and ML Landscape

## Core Idea

Modern AI is a collection of problem framings: prediction, representation learning, probabilistic inference, generation, decision-making, ranking, causal estimation, and production systems. Many techniques overlap, but each was built to answer a different kind of question.

## The Main Axes

- Supervised versus unsupervised versus self-supervised versus reinforcement learning.
- Discriminative versus generative modeling.
- Prediction versus causation.
- Static examples versus sequences and state.
- Offline training versus online decision-making.
- Model quality versus system reliability.

## Where Methods Fit

| Question | Family |
| --- | --- |
| What label should this example get? | Supervised learning |
| What hidden structure explains the data? | Probabilistic and latent variable modeling |
| How do I represent high-dimensional data? | Representation learning |
| What happens next in a sequence? | Sequence and state models |
| What action should I take? | Bandits and reinforcement learning |
| Which item should be shown first? | Retrieval, ranking, recommendation |
| What new content should be generated? | Generative AI |
| Did this action cause an outcome? | Causal inference |
| Will this work reliably in production? | ML systems |

## Common Confusions

- A model can be generative without being a modern GenAI model.
- Deep learning is a function approximation toolkit, not one problem type.
- Embeddings are representations, not an objective by themselves.
- RL is for sequential decisions, not just "learning from feedback."
- Causal inference is about interventions, not higher-accuracy prediction.

## What To Be Able To Explain

- Why the same product problem can be framed as classification, ranking, retrieval, or RL.
- Why uncertainty matters in some tasks and can be incidental in others.
- Why modern AI systems combine multiple model families.
- Why evaluation changes across prediction, generation, ranking, and decision-making.
