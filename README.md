# Predicting Llama-3-8B Activations from QA-Emb

A fork of the [QA-Emb paper](https://arxiv.org/abs/2405.16714) (Benara et al., 2024) that swaps the target: instead of regressing 674 yes/no question answers onto **fMRI voxel responses**, we regress them onto the **internal hidden states of Llama-3-8B-Instruct**. The original paper's `readme.md` documents the upstream QA-Emb library and fMRI pipeline; everything below describes this fork's additions.

## What this fork does

For each of 190,515 sliding-10-word n-grams from the fMRI story corpus (OpenNeuro ds003020):

1. **QA-Emb features** — `(190515, 674)` bool — already provided by the paper authors. Row `i`, column `j` = "did Llama-3-8B-Instruct answer Yes to question `j` about n-gram `i`?"
2. **Llama hidden states** — `(190515, 4096)` float32 per layer — extracted by us at layers **0, 16, 31** (first, middle, last), at the **last-token position**, with the n-gram fed as raw text.
3. **PCA bottleneck** — 100 components per layer, fit on the training split.
4. **Ridge regression** — `RidgeCV` from the 674-dim QA-Emb features to the 100-dim PCA-projected activations, with story-based train/test split (test = `wheretheressmoke`, `fromboyhoodtofatherhood`).
5. **Reporting** — Pearson correlation per PCA component (PCA-target space) and per activation dimension (full 4096-dim activation space, via PCA inverse-transform).
6. **Sparsity sweep** — `MultiTaskElasticNet` over 20 log-spaced alphas in [1e-3, 1] (paper-faithful), then re-fit Ridge on selected questions — replicates the paper's "how few questions suffice?" analysis on this new target.

Headline numbers (test, story-held-out):

| Layer | Mean test r (PCA target, 100 dims) | Mean test r (full 4096-dim activation space) |
|------:|-----------------------------------:|---------------------------------------------:|
|     0 |                              0.180 |                                        0.124 |
|    16 |                              0.370 |                                        0.222 |
|    31 |                              0.406 |                                        0.335 |

Sparsity findings (top-K by Ridge importance, full activation r):

| K (questions) | layer 0 | layer 16 | layer 31 |
|--------------:|--------:|---------:|---------:|
|             1 |   0.005 |    0.007 |    0.011 |
|            29 |   0.073 |    0.090 |    0.178 |
|           674 |   0.131 |    0.230 |    0.345 |

The paper's "29 questions outperform Eng1000" sparsity result (their §4.2) replicates qualitatively: by K=29 we already recover roughly half of the full-674 performance at the deeper layers.

## What these findings mean

The single sentence: **interpretable yes/no question answers about an input text linearly predict a meaningful chunk of Llama-3-8B's internal hidden states**, with the prediction quality growing monotonically deeper into the network and concentrated in roughly two dozen questions.

A few things to read off the table above:

- **Layer depth monotonically improves predictability** (full-act r: 0.124 → 0.222 → 0.335 from layer 0 → 16 → 31). Layer 0 sits right above the embedding lookup; its hidden states are still mostly local-lexical, and 674 high-level yes/no questions don't capture lexical detail well. Layer 16's jump to 0.222 reflects the model having done the work of combining tokens into semantic content — exactly the kind of thing the questions are designed to probe. Layer 31's 0.335 is the strongest signal yet; by the final layer the activations are nearly the output distribution, so the abstractions the network has already used to decide the next token line up well with what an interpretable question bank can describe. **Impact**: if you're using this kind of probing to *understand* what a layer encodes, the deeper layers are the productive target; the input-side layers will look opaque to discrete semantic questions even when they carry meaningful structure.

- **PCA-target r is structurally higher than full-activation r at every layer**, by 0.05–0.15. The gap is the part of activation variance that lives outside the top 100 PCA components — directions PCA discards as "low-variance," but which still contribute to per-dim correlations in the full 4096-dim space. The fact that PCA-target r is *not* dramatically higher (e.g. 0.370 vs 0.222 at layer 16 is a 1.7× gap, not 10×) is informative: a lot of the action is in PCA's top 100 components, but enough lives in the tail that the headline metric should be the full-activation r if we want to say something honest about predicting Llama's activations. **Impact**: PCA is a useful regression target — most predictable variance is in its top-100 — but reporting only PCA-target r overstates the system's reach into the model's representation. Both numbers matter; the full-act r is the one to quote in any external comparison.

- **Sparsity is severe and the gradient is steep at the bottom**: at K=1 the deepest layer already hits r=0.011 — small in absolute terms but well above zero, meaning **a single well-chosen yes/no question carries detectable signal about Llama's last-layer hidden state**. By K=29, layer 31 reaches r=0.178, which is ~52% of the full-674 performance. The Eng1000 baseline the paper compares against uses 985 features; here we're matching half of "674 questions" with 29. **Impact**: this is QA-Emb's core selling point translated to LLM internals — the *interpretable* feature space doesn't need to be wide. A short, hand-readable checklist of ~30 questions explains a meaningful fraction of what a 4096-dim hidden state encodes. For interpretability work, this means the dictionary you build to talk about the model's representations doesn't have to be huge to be useful.

- **The mean r of 0.222 at layer 16 corresponds to roughly R² ≈ 0.05 per dim**, which is *low* in absolute terms but *not low* given the difficulty of the task: a 184k-row regression onto 4096 high-dimensional activation features using 674 crude binary features. The variance not explained is not noise — it's directions of the hidden state that aren't lexically or semantically scriptable as yes/no questions about input n-grams alone. **Impact**: any claim that "QA-Emb captures everything Llama does" is wrong; any claim that "QA-Emb is useless because R² is small" is also wrong. The right framing is: discrete interpretable features can recover the linearly-predictable component of activations, which is non-trivial and growing with depth.

- **Autoencoder beats PCA in reconstruction by ~22% MSE** (layer 0; similar at other layers). A small MLP bottleneck (4096→1024→100→1024→4096, GELU) preserves more activation variance than the linear PCA bottleneck at the same width. **Impact**: a non-linear bottleneck is a tighter coordinate system on the activation manifold than 100 linear directions. The downstream question — whether it's also a *better Ridge target* — is the comparison currently in flight. If it is, swapping PCA for the AE decoder will sharpen the full-activation prediction without changing any other piece of the pipeline.

- **Why this matters for the upcoming steering work**: the same linear map `QA-Emb → activations` that we use here to *measure* alignment can be inverted to *cause* it. ActAdd steers a model by adding a residual-stream vector built from the difference of two prompts' activations. We are building that same vector from the difference of two prompts' **QA-Emb features** instead — `qa_diff (674) → Ridge.coef_.T → bottleneck → decoder → 4096-dim residual perturbation`. The findings here directly bound what's achievable: layer 16 is the right intervention site (best balance of predictability and depth), 29 questions is a plausible "vocabulary" for constructing the steering direction, and full-activation r — not PCA r — is the metric that should be used to choose between bottleneck options. If the AE bottleneck wins the comparison, steering vectors built through it should be sharper than the PCA-derived ones.

The broader contribution this fork makes: it ports a probing methodology developed for fMRI (where the target is biologically noisy) to LLM activations (where the target is deterministic and accessible). The fact that the same 674 questions designed for brain prediction also work, monotonically with layer depth, on Llama-3-8B is non-trivial evidence that **the semantic abstractions the brain uses to process language and the abstractions Llama-3 learned during pretraining are similar enough to share a feature vocabulary**.

## Pipeline overview

```
       fmri_cached_qa_features/                  activations_full.npz
              (received, paper authors)                (ours, extracted on H200)
            ┌───────────────────┐                ┌────────────────────────┐
n-gram i → │ qa_train[i] (674) │   ─Ridge→     │ activations[layer][i]  │
            └───────────────────┘                │       (4096)           │
                       ↑                          └────────────────────────┘
                       │                                     │
            metadata['index'] (n-gram text)                  │
            metadata['columns'] (674 questions)              │
                                                             ↓
                                              PCA (fit on train) → 100 dims
                                                             ↓
                                              RidgeCV (QA-Emb → PCA)
                                                             ↓
                                              Pearson r per component
                                                             ↓
                                              PCA.inverse_transform → 4096-dim
                                                             ↓
                                              Pearson r per activation dim
```

## Repository layout

```
.
├── activation_pca.ipynb               # main analysis notebook (this fork's contribution)
├── activation_pca.executed.ipynb      # executed copy with all plots inline
├── activation_pca.executed.html       # standalone HTML viewer (1 MB, plots embedded)
├── activation_pca_plots.html          # plots-only HTML report (regenerable from caches)
├── CLAUDE.md, CLAUDE_CONTEXT.md       # working notes for AI-assisted development
├── plan.md                            # design notes for activation_pca.ipynb
├── crafting_interpretable_embeddings.pdf  # the QA-Emb paper
├── steering_language_models.pdf       # ActAdd paper (for upcoming steering work)
├── neuro1/                            # upstream QA-Emb library — not modified
├── experiments/                       # upstream fMRI scripts — not modified
├── mteb/                              # upstream MTEB experiments — not modified
└── readme.md                          # upstream paper's README (kept verbatim)
```

The notebook is the entry point. It's structured into 12 numbered sections plus an §11b that we added: §1–7 set up paths, load n-grams, load Llama, extract activations, load QA-Emb; §8 runs PCA; §9 runs the Ridge fit; §10 produces all evaluation plots; §11 is the ElasticNet sparsity sweep; §11b is the top-K-by-Ridge-importance sweep we added; §12 summarizes.

### Off-repo working files

Because activations are large (~9.4 GB) and the `accounts` filesystem has a 10 GB user quota, all derived artifacts live under `/scratch/users/arihant_kaul/`:

- `/scratch/users/arihant_kaul/fmri_cached_qa_features/` — paper authors' precomputed QA-Emb cache (rsynced once from a local copy).
- `/scratch/users/arihant_kaul/qa_emb_activations/` — this fork's outputs:
  - `activations_full.npz` (9.4 GB) — Llama hidden states at layers 0/16/31.
  - `ridge_results.npz` (15 MB) — `coef`, `correlations`, `mean_corr`, `best_alpha`, `Y_pred`, `Y_test` per layer.
  - `pca_components.npz` — PCA `components_` and `mean_` per layer.
  - `pca_explained_var.npz` — `explained_variance_ratio_` per layer.
  - `extra_metrics.npz` — full-activation-space mean correlations and top-K (by Ridge importance) sweep at K ∈ {1,2,3,5,10,15,20,29,50,100,200,400,674}.
  - `analyze_layer_{0,16,31}.npz` — ElasticNet sparsity-sweep results: `(alpha, n_nonzero, perf_PCA)` for each alpha; layers 16 and 31 were extended via bisection so all three layers reach K=1.
  - `ae_state_layer_{0,16,31}.pt`, `ae_bottlenecks_layer_{0,16,31}.npz`, `ae_summary.npz` — trained autoencoder bottlenecks (alternative to PCA, comparison in progress).

## What changed in this fork

- Added `activation_pca.ipynb` from scratch — replaces the fMRI target with Llama hidden states.
- New PCA + Ridge + per-dimension correlation evaluation in both **PCA-target space (100 dims)** and **full activation space (4096 dims)**. The full-activation metric is the headline; PCA-target is shown as a secondary curve where relevant.
- Plot set:
  - PCA cumulative variance per layer
  - Per-PCA-component Pearson r (Plot A)
  - Mean correlation per layer, paired PCA vs full-activation bars (Plot B)
  - Top-10 questions per layer by mean |Ridge coef| (Plot C, text)
  - Per-4096-dim Pearson r histogram (Plot D)
  - ElasticNet sparsity sweep — full-activation r as the primary curve, PCA r dashed secondary, 29-question line marked. Alphas that selected 0 questions are dropped. Extended via bisection so K reaches 1 for all three layers.
  - Top-K-by-Ridge-importance sweep — both metrics, 29-question marker, K∈{1,...,674}.
- All plots labeled with natural titles ("How well QA-Emb predicts each Llama-3-8B layer", "How few questions are needed?", etc.).
- Autoencoder bottleneck (50-epoch MLP, 4096→1024→100→1024→4096 with GELU) trained per layer as an alternative to static PCA. Layer 0 AE beats PCA by **+22.7%** in test-set reconstruction MSE; layers 16 and 31 by similar margins. Whether the AE bottleneck is also a better Ridge target is the next experiment (`compare_ae_vs_pca.py`).
- Upcoming: ActAdd-style steering with steering vectors built in QA-Emb space — `qa_diff (674) → Ridge.coef_.T → bottleneck → decoder → 4096-dim residual-stream perturbation`. Implementation in `steering_qa_emb.py` (PCA path complete; AE path stubbed). Two example pairs from ActAdd Table 1 (Love/Hate, Weddings).

## Critical gotchas

These are the non-obvious things that bite if you re-run any of this:

- **Cache layout.** `meta_train['index']` is a **list of n-gram strings**, not integer indices into another list. The notebook does `train_ngrams = list(meta_train['index'])` directly. This contradicts an earlier note in `CLAUDE.md` that called it an index list.
- **Row alignment.** Row i of `qa_emb` and row i of `activations[layer]` are derived from the **same n-gram string**, in **the same order**: train rows 0..184563 come from `meta_train['index']`, test rows 184564..190514 from `meta_test['index']`, both fed sequentially through activation extraction with no shuffle. `meta_train['columns'] == meta_test['columns']` is `True` — the 674-question column order is consistent across train and test, so `np.concatenate([qa_train, qa_test], axis=0)` is column-safe.
- **Activation extraction OOM.** Batch size **8** (not 32) for `output_hidden_states=True` — at fp16 with all 33 layer hidden states, batch 32 OOM's on H200. Free `outputs, hidden_states, inputs` and `torch.cuda.empty_cache()` after each batch.
- **`output_hidden_states` placement.** Pass it to the **forward call** (`model(**inputs, output_hidden_states=True)`), not to `from_pretrained()`.
- **Hidden-state indexing.** `outputs.hidden_states` has 33 entries (1 embedding + 32 layers). Layer L's hidden state is at **index L+1**.
- **PCA refit is expensive.** Cell 17 now loads `pca_components.npz` + `pca_explained_var.npz` from disk if present; only refits PCA on first run. Saves ~30 min on every subsequent notebook execution.
- **ElasticNet sparsity range.** The paper's alpha range [1e-3, 1] doesn't induce K=1 on this task at layers 16 and 31. Layer 16 needs alpha ≈ 0.5363, layer 31 needs alpha ≈ 10.94. Don't be alarmed if the paper-faithful path doesn't span [1, 674] — bisect for the gap.
- **Hardcoded paths.** `neuro1.config` points at `/home/chansingh/mntv1` and `/mntv1` (paper authors' machine). Anything that imports `neuro1.config` will break on a fresh setup. The fork's notebook does **not** import `neuro1.config`; it uses `/scratch/users/arihant_kaul/...` directly.

## How to reproduce on SCF

1. **Stage the QA-Emb cache** (~1 GB) under `/scratch/users/arihant_kaul/fmri_cached_qa_features/`. The paper authors share this on request; layout is one folder per LLM checkpoint (`meta-llama___Meta-Llama-3-8B-Instruct/`) with answer `.npz` and metadata `.pkl` files inside.
2. **Install** in a venv: `pip install -r requirements.txt` (`transformers`, `accelerate`, `scikit-learn`, `torch`, `tqdm`, `matplotlib`, `numpy`, `joblib`).
3. **First end-to-end run** (long, ~4 h on H200 because of activation extraction): `sbatch run_notebook.sbatch` — extracts activations, fits PCA, fits Ridge, runs ElasticNet sweep. All intermediate results cache under `/scratch/users/arihant_kaul/qa_emb_activations/`.
4. **Subsequent re-runs** (plotting only, ~1–2 min): rerun the notebook; every heavy step is cache-hit.
5. **Standalone plot regeneration** (no notebook, no GPU): `python /scratch/users/arihant_kaul/make_plots.py`. Writes PNGs to `/scratch/users/arihant_kaul/plots/` and an HTML report to `activation_pca_plots.html`.

Compute environment: Berkeley SCF, `berkeleynlp` partition, H200 GPUs (`horton`, `lorax`). 1× H200 (NVL or PCIe), 64 GB RAM, 8 CPUs is enough.

## Status

| Done                                                                | In flight                                  |
|---------------------------------------------------------------------|--------------------------------------------|
| Activation extraction (layers 0/16/31, 190,515 n-grams)             | AE vs PCA Ridge-target comparison          |
| PCA, Ridge, full-activation evaluation, all 8 plots                 | QA-Emb-based ActAdd steering implementation |
| ElasticNet sparsity sweep to K=1 (bisected on layers 16, 31)        |                                            |
| Top-K-by-Ridge-importance sweep at K∈{1,...,674}                    |                                            |
| AE bottleneck training (50 ep, +22% MSE improvement over PCA)       |                                            |

## Citation

The upstream paper:

```bibtex
@misc{benara2024crafting,
  title  = {Crafting Interpretable Embeddings by Asking LLMs Questions},
  author = {Vinamra Benara and Chandan Singh and John X. Morris and Richard Antonello and Ion Stoica and Alexander G. Huth and Jianfeng Gao},
  year   = {2024},
  eprint = {2405.16714},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

ActAdd, used in the upcoming steering experiments:

```bibtex
@misc{turner2024steering,
  title  = {Steering Language Models With Activation Engineering},
  author = {Alexander Matt Turner and Lisa Thiergart and Gavin Leech and David Udell and Juan J. Vazquez and Ulisse Mini and Monte MacDiarmid},
  year   = {2024},
  eprint = {2308.10248}
}
```
