# Full Project Context for Claude Code

## 1. What This Project Does

We are extending the QA-Emb paper (Benara et al., 2024, "Crafting Interpretable Embeddings by Asking LLMs Questions", arXiv:2405.16714) to predict **Llama 3-8B internal activations** instead of fMRI brain voxels.

**The core idea:** QA-Emb creates interpretable text embeddings where each dimension is the answer to a yes/no question (e.g., "Does the text mention a person?"). The paper showed these 674-dimensional binary vectors can predict fMRI brain responses to narrative text. We test whether the same features can predict what's happening inside the LLM itself.

**Pipeline:**
1. Take ~190k sliding 10-word n-grams from narrative podcast transcripts
2. For each n-gram, we already have precomputed QA-Emb (674-dim binary vectors) from the paper authors
3. Run each n-gram through Llama 3-8B-Instruct and extract hidden state activations at layers 0, 16, 31
4. PCA on activations → 100 components (fit on train only)
5. Ridge regression: QA-Emb (674-dim) → PCA activations (100-dim)
6. Evaluate with Pearson correlation on held-out test stories

**This is a class project** with a 2-week deadline (from ~May 2, 2026). The collaborator is a PhD student in Berkeley NLP. The code will run on Berkeley's Statistics Computing Facility (SCF) or Google Colab Pro.

---

## 2. Precomputed Data (CRITICAL — eliminates the most expensive step)

The paper authors shared their cached QA-Emb features via Google Drive. This data is at:
```
/path/to/fmri_cached_qa_features/
```
(On the user's local machine: `/home/arihant/predict_activations/fmri_cached_activations/fmri_cached_qa_features/`)

### Directory Structure
```
fmri_cached_qa_features/
├── readme.md                           # Usage instructions (see below)
├── questions_list_35_ordered.pkl       # 35 "stable" questions for best fMRI prediction
├── meta-llama___Meta-Llama-3-8B-Instruct/
│   ├── v3_boostexamples_answers_train_numpy.npz   # shape: (184564, 674), dtype: bool
│   ├── v3_boostexamples_answers_test_numpy.npz    # shape: (5951, 674), dtype: bool
│   ├── v3_boostexamples_train_metadata.pkl        # dict with 'index' and 'columns'
│   └── v3_boostexamples_test_metadata.pkl         # dict with 'index' and 'columns'
├── meta-llama___Meta-Llama-3-8B-Instruct-fewshot/ # same structure, (184564, 674) + (5951, 674)
├── meta-llama___Meta-Llama-3-70B-Instruct/        # same structure
├── mistralai___Mistral-7B-Instruct-v0.2/          # same structure
└── gpt4/
    ├── ngrams_metadata.joblib      # {'wordseq_idxs': {story: (start, end)}, ...}
    ├── ngrams_list_total.joblib    # list of all n-gram text strings
    └── [86 question pickle files]  # GPT-4 answers per question
```

### How to Load QA-Emb (from the readme)
```python
import numpy as np
import pickle
import joblib

# Single model (Llama 3-8B):
folder = 'meta-llama___Meta-Llama-3-8B-Instruct'
qa_emb = np.load(f'{folder}/v3_boostexamples_answers_train_numpy.npz')['arr_0']  # (184564, 674) bool
with open(f'{folder}/v3_boostexamples_train_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
# metadata['index'] = sample indices, metadata['columns'] = 674 question strings

# Ensemble (average of 4 models):
models = ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct',
          'meta-llama/Meta-Llama-3-8B-Instruct-fewshot', 'meta-llama/Meta-Llama-3-70B-Instruct']
answers_list = []
for m in models:
    folder = m.replace("/", "___")
    answers_list.append(np.load(f'{folder}/v3_boostexamples_answers_train_numpy.npz')['arr_0'])
qa_emb_ensemble = np.mean(np.array(answers_list), axis=0)  # (184564, 674) float, values in {0, 0.25, 0.5, 0.75, 1.0}
```

### N-gram Texts (from GPT-4 folder)
```python
ngrams_metadata = joblib.load('gpt4/ngrams_metadata.joblib')
ngrams_all = ngrams_metadata['ngrams_list_total']  # list of all n-gram text strings
wordseq_idxs = ngrams_metadata['wordseq_idxs']     # {story_name: (start_idx, end_idx)}
# Example: wordseq_idxs['adollshouse'] = (0, 1523) means ngrams_all[0:1523] are from that story
```

### Critical Alignment Issue
- The precomputed QA-Emb has **184,564 train** + **5,951 test** = **190,515 total** samples
- The `ngrams_list_total` should have entries for all of these
- The metadata `index` field tells you which rows in the QA-Emb array correspond to which n-gram indices
- **You must verify this alignment** when first loading the data — the index in train_metadata might be numeric indices into ngrams_list_total, or it might be something else (story+position identifiers)
- The train/test split is story-based: test stories are 'wheretheressmoke' and 'fromboyhoodtofatherhood'

### Why 184k Samples Instead of 123k?
The paper reports ~123k training samples for individual subjects (each heard 79-82 stories). The cached data has 184k because it covers the **union** of all stories across all 3 subjects. Our notebook should use all of them since we're not subject-specific.

---

## 3. The Paper's Methodology (key details)

**Paper:** "Crafting Interpretable Embeddings by Asking LLMs Questions" (Benara et al., 2024)
**PDF location:** `2405.16714v1.pdf` in the project root

### Key Numbers
- 674 yes/no questions (generated by GPT-4, stored in `v3_boostexamples.json`)
- 3 fMRI subjects, each heard 79-82 training stories + 2 test stories
- Training: ~123k 10-word sliding n-grams per subject (184k total across all subjects)
- Test: ~4.6k n-grams (from stories 'wheretheressmoke' and 'fromboyhoodtofatherhood')
- 100 PCA components for dimensionality reduction
- Ridge regression with cross-validated alpha from 12 log-spaced values between 10 and 10,000
- Evaluation: Pearson correlation per PCA component (or per voxel after inverse PCA)

### QA-Emb Computation (how the paper did it)
- Used **Mistral-7B-Instruct-v0.2** and **Llama-3-8B-Instruct** with 2 prompts each
- Averaged answers across all 4 runs (ensemble) → values in {0, 0.25, 0.5, 0.75, 1.0}
- Each (text, question) pair was an independent forward pass — one question per context window
- Compared P(Yes) vs P(No) from next-token logits
- Ran on 64 AMD MI210 GPUs, took ~4 days total
- The paper also computed GPT-4 answers (86 questions, different set)

### Paper Results (Table A2, page 16)
Mean test correlation for 674 questions:
| Setting | S01 | S02 | S03 | AVG |
|---------|-----|-----|-----|-----|
| Ensemble | 0.084 | 0.124 | 0.141 | 0.116 |
| Llama-3 8B | 0.081 | 0.119 | 0.136 | 0.112 |
| Llama-3 8B fewshot | 0.085 | 0.121 | 0.136 | 0.114 |
| Mistral 7B | 0.076 | 0.108 | 0.132 | 0.105 |

### 35 Stable Questions
The file `questions_list_35_ordered.pkl` contains 35 questions selected for stable fMRI prediction. The paper shows 29 questions can outperform the 985-dim Eng1000 baseline.

---

## 4. Activation Extraction Details

### Model
- `meta-llama/Meta-Llama-3-8B-Instruct` (same model used for QA-Emb answering)
- Load in float16 (no quantization) for exact hidden states
- `device_map='auto'` for GPU placement
- Do NOT pass `output_hidden_states=True` to `from_pretrained()` — only to the forward pass

### Extraction
- Layers to extract: 0 (first), 16 (middle), 31 (last) — i.e., `[0, num_layers // 2, num_layers - 1]`
- Use the **last token's hidden state** per n-gram (has attended to all prior tokens via causal attention)
- `outputs.hidden_states[layer_idx + 1]` (index 0 is the embedding layer, so layer 0's hidden state is at index 1)
- Extract with: `hidden_states[l + 1][torch.arange(batch_size), seq_lengths]` where `seq_lengths = attention_mask.sum(dim=1) - 1`

### Memory Management
- **Batch size 8** (not 32) to avoid OOM — storing all 33 hidden state tensors is ~1.1GB at batch_size=32
- After each batch: `del outputs, hidden_states, inputs; torch.cuda.empty_cache()`
- Checkpoint every 512 samples to Google Drive / disk

### Output
- Per layer: numpy array of shape `(n_samples, 4096)`, dtype float32
- Save as `activations_cached.npz` with keys `layer_0`, `layer_16`, `layer_31`

---

## 5. Downstream Analysis

### PCA
- 100 components per layer
- **Fit on training data only** (avoid leakage)
- Transform both train and test

### Ridge Regression
- `RidgeCV` from scikit-learn
- Alphas: `np.logspace(-1, 4, 20)`
- X = QA-Emb (674-dim), Y = PCA activations (100-dim)
- Story-based train/test split (NOT random split)
- Evaluate: per-component Pearson correlation on test set

### Feature Selection (Elastic Net)
- `MultiTaskElasticNetCV` to rank questions by importance
- Test performance with progressively fewer questions: [5, 10, 15, 20, 29, 50, 100, 200, 400, 674]
- Compare with paper's finding that 29 questions suffice for fMRI

### Visualizations
1. PCA variance explained curves per layer
2. Per-component test correlation bar charts per layer
3. Layer comparison summary (mean correlation bar chart)
4. Top 10 most predictive questions per layer (by ridge weight magnitude)
5. Per-dimension correlation histograms in full 4096-dim space (after inverse PCA)
6. Feature selection curve (performance vs number of questions)

---

## 6. Notebook Structure (`activation_pca_cached.ipynb`)

The notebook has already been created at `/home/arihant/predict_activations/activation_pca_cached.ipynb`. It has 33 cells covering:
1. Setup & installs
2. Google Drive mount
3. Load precomputed QA-Emb from paper authors
4. Extract n-gram texts from ngrams_metadata.joblib
5. Verify alignment between QA-Emb rows and n-gram texts
6. Load Llama 3-8B-Instruct (HuggingFace login)
7. Extract activations (batch_size=8, checkpointing)
8. PCA (100 components, train-only fit)
9. Ridge regression
10. Visualizations and feature selection

**The alignment cells (4-5) need verification** — the metadata index format hasn't been inspected in a live Python environment yet. The code assumes numeric indices into ngrams_list_total, but this needs to be confirmed.

---

## 7. Compute Environment

Options (in order of preference):
1. **Berkeley SCF** — The user has access. Likely has GPUs, no time limits, can run unattended.
2. **Google Colab Pro** — A100 40GB GPU. Subject to disconnects. All intermediate results cached to Google Drive.
3. **Local machine** — No GPU (Arch Linux, Python 3.14).

The notebook is designed for Colab but can be adapted for SCF with minimal changes (remove `drive.mount()`, change `CACHE_DIR` to a local path).

---

## 8. Known Issues and Gotchas

1. **OOM on activation extraction**: Must use batch_size=8, not 32. Must delete outputs after each batch.
2. **output_hidden_states placement**: Only pass to `model(**inputs, output_hidden_states=True)`, NOT to `from_pretrained()`.
3. **Train/test alignment**: The precomputed data has a pre-defined train/test split. The test stories are 'wheretheressmoke' and 'fromboyhoodtofatherhood'. Must use the same split for activations.
4. **Data type**: QA-Emb arrays are boolean (dtype `|b1`). Cast to float32 before regression: `qa_emb.astype(np.float32)`.
5. **N-gram count discrepancy**: Our TextGrid parser produced 141,784 train n-grams vs the paper's 123,203. The precomputed data has 184,564. These differences are because different sets cover different story subsets. Use the precomputed data's sample count as ground truth.
6. **Hidden state indexing**: `outputs.hidden_states` has 33 entries (embedding + 32 layers). Layer 0's hidden state is at index 1, layer 31 is at index 32.

---

## 9. Paper Reference

- **Title:** Crafting Interpretable Embeddings by Asking LLMs Questions
- **Authors:** Vinamra Benara*, Chandan Singh*, et al.
- **Affiliations:** UC Berkeley, Microsoft Research, Cornell, UT Austin
- **arXiv:** 2405.16714v1
- **Code:** https://github.com/csinva/interpretable-embeddings
- **Data:** OpenNeuro ds003020 (fMRI story corpus)
- **PDF in repo:** `2405.16714v1.pdf`
