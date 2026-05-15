# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **fork** of the QA-Emb paper code (Benara et al., 2024, "Crafting Interpretable Embeddings by Asking LLMs Questions", arXiv:2405.16714) extended to a new research goal: instead of predicting fMRI brain voxels from QA-Emb features, predict **Llama 3-8B internal hidden-state activations**. It is a class project on a ~2-week deadline.

The original paper's pipeline (predict fMRI voxels) is implemented in the `neuro1/` package + `experiments/` scripts. The fork's new work lives primarily in:
- `activation_pca.ipynb` — the main project notebook (n-grams → Llama 3-8B activations → PCA → ridge from QA-Emb)
- `plan.md` — design notes for that notebook
- `CLAUDE_CONTEXT.md` — **read this first.** It has the precomputed-data layout, expected sample counts, layer-extraction details, train/test split info, and a full list of known gotchas. Treat it as authoritative over this file when they overlap.
- `crafting_interpretable_embeddings.pdf` — the paper itself

## Repository Architecture

Two layers coexist in this repo and should not be conflated:

**1. Upstream paper code (don't modify unless the task is upstream)**
- `neuro1/` — installable package (`pip install -e .` registers it as `neuro1`).
  - `neuro1/data/` — fMRI response loading, story name lists, TextGrid parsing (`textgrid.py`, `story_names.py`, `response_utils.py`).
  - `neuro1/features/` — feature spaces. `qa_embedder.py` is the QA-Emb computation (one forward pass per `(text, question)` pair, comparing P(Yes) vs P(No) from next-token logits). Question lists live in `neuro1/features/questions/` (`v3_boostexamples.json` is the 674-question set).
  - `neuro1/encoding/` — ridge regression with bootstrap CV (`ridge.py`) and evaluation utilities (`eval.py`).
  - `neuro1/config.py` — hardcoded paths (`/home/chansingh/mntv1` or `/mntv1`); paths must be patched for any non-author environment.
- `experiments/` — original training entry points: `00_download_dataset.py` (datalad clone of OpenNeuro ds003020), `02_fit_encoding.py` (the canonical "QA-Emb → fMRI voxel ridge regression" script).
- `notebooks/` — original paper's analysis notebooks (encoding, flatmaps, sparsity).
- `mteb/` — separate experiment using QA-Emb on MTEB benchmarks (largely unrelated to the fork's goal).
- `scripts/` — paper's job launchers / `d3` analysis.

**2. Fork's new work**
- `activation_pca.ipynb` — runs the new pipeline. Important: **do not use `neuro1/`'s pipeline for this notebook.** It pulls TextGrids directly from the OpenNeuro S3 bucket, parses them inline (regex over the word tier — tier index 1, filtering `BAD_WORDS`), and uses HuggingFace `transformers` directly for activation extraction.
- `activation_pca.ipynb` is **Colab-flavored** (mounts Drive, uses Groq API for QA-Emb in addition to / instead of computing locally). For SCF, replace `drive.mount()` and `CACHE_DIR` with a local path.

## Pipeline Specifics (fork's notebook)

1. **N-grams**: sliding 10-word n-grams over story TextGrids. Same train story list as the paper (~80 stories); test stories are exactly `wheretheressmoke` and `fromboyhoodtofatherhood`.
2. **QA-Emb**: 674 yes/no questions from `v3_boostexamples.json`. The paper authors shared **precomputed** answer arrays — use these rather than recomputing. See `CLAUDE_CONTEXT.md` §2 for the exact directory layout (`fmri_cached_qa_features/`), expected shapes (`(184564, 674)` train, `(5951, 674)` test, dtype bool), and the load recipe. The notebook also contains a Groq API fallback.
3. **Activations**: `meta-llama/Meta-Llama-3-8B-Instruct` in float16. Extract the **last token's hidden state** at layers `[0, num_layers // 2, num_layers - 1]` = `[0, 16, 31]`. Index as `outputs.hidden_states[layer_idx + 1]` (index 0 is the embedding layer).
4. **PCA → Ridge**: 100 PCA components per layer, fit on **train only** (story-based split, never random). Then `RidgeCV` from QA-Emb (674-dim) → PCA activations (100-dim). Evaluate per-component Pearson correlation on the held-out test stories.
5. **Feature selection**: `MultiTaskElasticNetCV` to rank questions; compare performance at `[5, 10, 15, 20, 29, 50, 100, 200, 400, 674]` questions (mirrors the paper's "29 questions suffice" finding).

## Critical Gotchas

These are situations where the obvious thing is wrong; **see `CLAUDE_CONTEXT.md` §8 for the full list with reasoning.**

- **Activation extraction OOM**: batch size must be **8** (not 32) because storing all 33 hidden-state tensors is ~1.1 GB at batch 32. After each batch, `del outputs, hidden_states, inputs; torch.cuda.empty_cache()`.
- **`output_hidden_states` placement**: pass it to the **forward call** (`model(**inputs, output_hidden_states=True)`), not to `from_pretrained()`.
- **Hidden state indexing**: `outputs.hidden_states` has 33 entries (embedding + 32 layers); layer N's hidden state is at index `N+1`.
- **Train/test split is story-based, not random**. The split must match across QA-Emb and activations. Test stories are fixed.
- **QA-Emb dtype**: precomputed arrays are bool (`|b1`). Cast to `float32` before regression.
- **N-gram count**: precomputed data has 184,564 train + 5,951 test = 190,515. The notebook's own TextGrid parser produces 141,784 train n-grams (covers a different story subset). The precomputed counts are ground truth — verify alignment between QA-Emb rows and your n-gram texts before training.
- **Hardcoded `neuro1/config.py` paths**: `/mntv1/deep-fMRI` is from the paper authors' machine. Anything that imports `neuro1.config` will break unless those paths exist or are patched.

## Common Commands

```bash
# Install the upstream paper package (only needed if running the original fMRI pipeline)
pip install -e .

# Install fork-specific deps
pip install -r requirements.txt

# Original paper: download fMRI dataset (requires datalad — author notes pip install didn't work, used apt)
python experiments/00_download_dataset.py

# Original paper: fit a single-subject encoding model
python experiments/02_fit_encoding.py --subject UTS03 --feature_space qa_embedder-10
# Fast smoke-test mode (single story, few bootstraps):
python experiments/02_fit_encoding.py --use_test_setup 1
```

There is no test suite, lint config, or build step in this repo. The project's "tests" are running the notebook end-to-end.

## Compute Environment

Three options in declining order of preference (per `CLAUDE_CONTEXT.md` §7):
1. **Berkeley SCF** — has GPUs, no time limits, runs unattended.
2. **Google Colab Pro** — A100 40GB, but disconnects; that's why the notebook checkpoints to Drive every 512 samples.
3. **Local** — no GPU.

The notebook is Colab-shaped by default. For SCF, replace `drive.mount()` and `CACHE_DIR` with a local path; the rest of the logic is portable.
