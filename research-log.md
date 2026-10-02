# Research log

Sources consulted for this book and what was checked in each. Add an entry whenever a new source
is used.

## 2026-10-02: audit and citation pass

`docs/references.bib` was a single placeholder; it now holds 30 entries, each generated from its
arXiv abstract page or Crossref record (titles, authors, years copied, not typed). Citations were
added where the frontier chapters (10, 12, 13, 14, 18) first describe a named method, and each
sentence was read against its source.

Corrections made during the pass:

- **DPO "cheaper and more stable"** (Ch 12) → "cheaper and simpler"; whether it matches tuned PPO
  is disputed (Xu et al. 2024, arXiv 2404.10719: "PPO is able to surpass other alignment methods
  in all cases").
- **Speculative decoding "purely to claw back throughput"** (Ch 12) → cuts latency (2–3× in the
  paper) with an unchanged output distribution (Leviathan et al., arXiv 2211.17192).
- **Fairness impossibility** (Ch 18): the text said demographic parity, equalized odds and
  calibration "cannot all be satisfied". Kleinberg et al. (2016) prove the conflict between
  calibration within groups and balanced error rates; Chouldechova (2017) similarly. Reworded to
  state that result, plus the elementary demographic-parity vs. equalized-odds conflict with its
  one-line derivation.
- **Tokenizer vocabulary "32k–200k"** (Ch 12): kept, now with sources (Llama 2 paper: 32k;
  tiktoken `o200k_base`: ~200k).
- **README** said the diffusion figure runs "forward and backward"; the script is forward only.

Confirmed and cited: RLHF pipeline (Christiano 2017; Ouyang 2022), PPO, Q-learning (Watkins &
Dayan 1992), UCB1 O(log T) (Auer et al. 2002), Thompson sampling (1933), VAE, GAN, DDPM, CLIP, RAG,
BPE (Sennrich), Transformer, LLM-as-judge biases (Zheng et al. 2023: position, verbosity,
self-enhancement), ReAct, indirect prompt injection (Greshake et al. 2023), Goodhart variants,
ε-DP (Dwork et al. 2006), equalized odds (Hardt et al. 2016), federated learning (McMahan et al.),
adversarial examples (Szegedy et al.; Goodfellow et al.), split-conformal quantile (Angelopoulos &
Bates 2021).

**Not yet checked:** Chapters 1–9, 11, 15–17 were not sampled.
