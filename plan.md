# Plan: Predicting LLM Activations with QA-Emb

## Goal
Replicate the QA-Emb → fMRI pipeline, but predicting **Llama 3-8B hidden-state activations** instead of brain voxels. We want to see if interpretable yes/no question embeddings can predict the internal representations of an LLM.

## Notebook Structure (`activation_pca.ipynb`)

### Cell 1: Setup & Installs
- Install `transformers`, `bitsandbytes`, `accelerate`, `scikit-learn`, `datasets`, `matplotlib`
- Mount Google Drive for caching intermediate results

### Cell 2: Load the 674 Questions
- Embed the questions JSON directly in the notebook (copy from `v3_boostexamples.json`) so the notebook is self-contained
- No dependency on the local repo structure

### Cell 3: Prepare Text Data (~200 n-grams)
- Load a narrative text dataset from HuggingFace (e.g., `wikitext-2-raw-v1` or `bookcorpus`)
- Split into 10-word n-grams, take first ~200
- Optional cell to load the full fMRI story dataset from Google Drive if available

### Cell 4: Load Llama 3-8B-Instruct
- Load `meta-llama/Meta-Llama-3-8B-Instruct` in **4-bit quantization** (fits on T4 Colab GPU ~5GB VRAM)
- Use same model for both activation extraction and QA-Emb answering

### Cell 5: Extract Activations
- For each n-gram, run a forward pass and extract hidden states at:
  - **Layer 0** (first layer — raw token embeddings + first transform)
  - **Layer 16** (middle)
  - **Layer 31** (last layer)
- Use the **last token's hidden state** (4096-dim) as the representation for each n-gram
- Save to Drive as numpy arrays: `activations_layer{N}.npy` shape (200, 4096)

### Cell 6: Compute QA-Emb
- For each of the 200 n-grams, answer all 674 yes/no questions using the loaded Llama 3-8B-Instruct
- Use the instruct prompt template from the codebase (formatted with question + input text)
- Generate 1 token, check if it starts with "Yes" or "No" → binary 0/1
- Result: `qa_emb.npy` shape (200, 674)
- Save to Drive
- This is the bottleneck: 200 × 674 = 134,800 inferences. With batching (~batch size 32), expect ~30-60 min on T4.

### Cell 7: PCA on Activations
- For each layer (0, 16, 31):
  - Run PCA on the (200, 4096) activation matrix
  - Extract top 50 components (reasonable given 200 samples)
  - Save the PCA model and transformed data
  - Plot variance explained curve

### Cell 8: Ridge Regression (QA-Emb → PCA Components)
- For each layer:
  - Split data: ~160 train / ~40 test (80/20 split)
  - Fit ridge regression: 674-dim QA-Emb → 50-dim PCA space
  - Cross-validate lambda (regularization parameter) on training set
  - Evaluate on test set: per-component Pearson correlation

### Cell 9: Evaluation & Visualization
- For each layer:
  - Report mean test correlation across PCA components
  - Plot per-component correlation (bar chart)
  - Compare layers: which layer is most predictable from interpretable questions?
- Feature importance: which questions have the largest regression weights?
- Reconstruct predictions in the full 4096-dim activation space, compute per-dimension correlations

### Cell 10: Analysis
- Compare across layers: are early/middle/late layers differently predictable?
- Identify which questions are most predictive for each layer
- Elastic Net feature selection: how few questions suffice? (mirroring the paper's 29-question result)

## Key Design Decisions
- **4-bit quantization**: Hidden states are still computed in float16 between layers; only weight matrices are quantized. Activations should be representative.
- **Last token pooling**: Matches the autoregressive nature — analogous to the fMRI "brain state reflects all preceding words" setup.
- **50 PCA components (not 100)**: With only 200 samples, 100 components would be 50% of the data. 50 is more conservative.
- **Same model for activations + QA answering**: Simpler (one model in memory). We can note that using a separate model for QA would be more "clean" but this suffices for prototyping.
- **Google Drive caching**: Save activations and QA-Emb vectors so they survive runtime resets. The expensive computation only needs to happen once.

## Expected Outcome
If QA-Emb can predict LLM activations, it would mean the semantic distinctions captured by yes/no questions align with the internal representation geometry of the LLM. This would be a form of interpretability — we could describe what an LLM layer is "thinking about" in terms of human-readable questions.
