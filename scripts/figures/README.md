# Figure scripts

Each `fig_*.py` script is a small, self-contained Python program: it computes
something real (on toy or synthetic data — a fitted tree, an MCMC chain, a
gridworld value function, ...) and renders it to a static PNG in
`docs/images/`. Quarto does not execute Python at render time; these images
are pre-generated and checked into the repo like any other asset.

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/figures/requirements.txt
```

Regenerate a single figure:

```bash
python scripts/figures/fig_bias_variance.py
```

Regenerate everything:

```bash
for f in scripts/figures/fig_*.py; do python "$f"; done
```

Every script imports `apply_theme()` and the shared palette from `_theme.py`
so all figures across all chapters look like one system — new figure scripts
should do the same rather than styling matplotlib ad hoc.
