# Information Retrieval, Ranking, and Recommenders

## Core Idea

Retrieval finds candidates from a large collection. Ranking orders them. Recommendation personalizes candidate selection and ordering based on user, item, context, and interaction signals.

## Retrieval

- Lexical retrieval: BM25, inverted indexes, exact terms.
- Dense retrieval: embedding similarity and approximate nearest neighbors.
- Hybrid retrieval: combines lexical and semantic matching.

## Ranking

- Pointwise: predict relevance for each item.
- Pairwise: learn preferences between items.
- Listwise: optimize list-level ordering.
- Reranking: apply diversity, freshness, safety, or business constraints.

## Recommenders

- Collaborative filtering.
- Matrix factorization.
- Content-based recommendation.
- Two-tower retrieval.
- Deep ranking.
- Sequential recommendation.
- Bandit-based exploration.

## Metrics

- Recall@K for retrieval.
- MRR for first relevant result.
- NDCG@K for graded ranking quality.
- MAP for precision across ranked results.
- CTR, conversion, retention, satisfaction, diversity, novelty.

## Connections

- Matrix factorization connects recommender systems to latent variable models.
- Dense retrieval connects representation learning to production search.
- Bandits connect recommendation to decision-making under uncertainty.
- RAG connects retrieval to GenAI.

## Interview Check

Be able to debug whether a bad recommendation came from missing candidates, poor ranking, biased labels, over-personalization, cold start, or serving constraints.
