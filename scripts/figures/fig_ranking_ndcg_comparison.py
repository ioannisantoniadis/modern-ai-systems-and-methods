import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, SEQUENTIAL_BLUE, INK, INK_SECONDARY, MUTED, GRIDLINE

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(42)

N_QUERIES = 20
N_DOCS = 10
MAX_GRADE = 3  # relevance grades 0..3

# Each document has a latent "true relevance score" (continuous) and a
# graded relevance label derived from it (0-3), so there is a real notion of
# quality to recover, not just arbitrary labels.
true_scores = rng.normal(size=(N_QUERIES, N_DOCS))
# Map continuous scores to graded labels via fixed quantile-style cuts.
grade_cuts = np.quantile(true_scores.ravel(), [0.4, 0.7, 0.9])
relevance = np.digitize(true_scores, grade_cuts)  # 0,1,2,3

def dcg_at_k(rel_sorted, k):
    rel_k = rel_sorted[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel_k) + 2))
    return np.sum((2.0 ** rel_k - 1.0) * discounts)

def ndcg_curve(relevance_row, ranking_order, ks):
    rel_ranked = relevance_row[ranking_order]
    ideal_ranked = np.sort(relevance_row)[::-1]
    out = np.zeros(len(ks))
    for i, k in enumerate(ks):
        dcg = dcg_at_k(rel_ranked, k)
        idcg = dcg_at_k(ideal_ranked, k)
        out[i] = dcg / idcg if idcg > 0 else 1.0
    return out

ks = np.arange(1, N_DOCS + 1)

# Strategy 1: random ranking (no model at all).
# Strategy 2: perfect relevance-score-sorted ranking (oracle-ish: sorts by
#             the true continuous score that generated the grades).
# Strategy 3: a ranking trained on position-bias-corrupted clicks: relevance
#             signal is corrupted by simulating that only the first few
#             positions of some arbitrary prior ranking ever got seen/clicked,
#             so the "training label" for lower positions is systematically
#             underestimated (never observed => train the surrogate as if it
#             had relevance 0), producing a model ranking that overweights
#             whatever happened to rank high before and misses truly relevant
#             but under-exposed documents.
strategies = {}
scores = {"random": [], "relevance_sorted": [], "position_bias": []}

for q in range(N_QUERIES):
    rel_row = relevance[q]
    score_row = true_scores[q]

    # Random ranking.
    rand_order = rng.permutation(N_DOCS)
    scores["random"].append(ndcg_curve(rel_row, rand_order, ks))

    # Relevance-score-sorted ranking (best possible given the true scores).
    sorted_order = np.argsort(-score_row)
    scores["relevance_sorted"].append(ndcg_curve(rel_row, sorted_order, ks))

    # Position-bias-corrupted: an arbitrary "prior" ranking (e.g. by
    # document id / recency) determines exposure; clicks only ever happen in
    # the top 3 exposed slots, so a naive click-trained relevance estimate
    # is 0 for anything below position 3 in that prior ranking, regardless
    # of true relevance. The resulting ranking sorts by this biased signal.
    prior_order = np.arange(N_DOCS)  # arbitrary fixed prior (e.g. by doc id)
    exposed_rank = np.empty(N_DOCS, dtype=int)
    exposed_rank[prior_order] = np.arange(N_DOCS)
    observed_relevance = np.where(exposed_rank < 3, rel_row, 0)
    # add a little noise so ties break plausibly
    biased_score = observed_relevance + rng.normal(0, 0.01, size=N_DOCS)
    biased_order = np.argsort(-biased_score)
    scores["position_bias"].append(ndcg_curve(rel_row, biased_order, ks))

mean_curves = {k: np.mean(v, axis=0) for k, v in scores.items()}
std_curves = {k: np.std(v, axis=0) for k, v in scores.items()}

fig, ax = plt.subplots(figsize=(7.5, 4.8))

labels = {
    "random": ("Random ranking", CATEGORICAL[7]),
    "position_bias": ("Trained on position-biased clicks", CATEGORICAL[3]),
    "relevance_sorted": ("Sorted by true relevance score", CATEGORICAL[2]),
}

for key, (label, color) in labels.items():
    mean_c = mean_curves[key]
    std_c = std_curves[key]
    ax.plot(ks, mean_c, color=color, linewidth=2.2, marker="o", markersize=4,
             label=label)
    ax.fill_between(ks, mean_c - std_c, mean_c + std_c, color=color, alpha=0.12)

ax.set_xlabel("k")
ax.set_ylabel("nDCG@k")
ax.set_ylim(0, 1.05)
ax.set_xticks(ks)
ax.set_title(f"nDCG@k across {N_QUERIES} queries x {N_DOCS} graded candidates")
ax.legend(loc="lower right", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "ranking_ndcg_comparison.png"
savefig(fig, str(out))
