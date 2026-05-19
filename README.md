# Selected Course Projects

A collection of three machine learning projects spanning classical methods, deep learning, and LLM efficiency research.

---

## Projects

### 1. Image Classification — Classical ML and Transfer Learning

`image_classification/`

Benchmarks a progression of methods on MNIST (handwritten digits) and a monkey-species dataset (10 classes):

| Method | Dataset | Key result |
|--------|---------|-----------|
| SVM (RBF) + PCA (50 components) | MNIST | 97.1% accuracy |
| Logistic regression from scratch (NumPy softmax) | MNIST | competitive with sklearn |
| Custom CNN | MNIST | 95.48% |
| VGG19 (frozen backbone) | Monkey species | 93.38% |
| VGG19 (fine-tuned) | Monkey species | 97.06% |

The notebook implements logistic regression from scratch using only NumPy (softmax activation, cross-entropy loss, gradient descent), providing a direct comparison against sklearn's L-BFGS solver. The VGG19 section demonstrates why fine-tuning outperforms frozen-feature extraction on a small domain-shift dataset.

**Data:** MNIST loads automatically. Monkey species dataset should be placed in `data/training/training/` and `data/validation/validation/`.

**Files:**
- `image_classification.ipynb` — full pipeline
- `report.pdf` — written report

---

### 2. Face Recognition — Dimensionality Reduction and Kernel Classifiers

`face_recognition/`

Evaluates classical face recognition methods on a 200-subject dataset (neutral, expression, illumination images; 24×21 pixels).

**Two tasks:**
- **Task 1:** 200-class person identification (2 training images per subject)
- **Task 2:** Binary neutral-vs-expression classification (80/20 split)

**Classifiers implemented from scratch:**

| Notebook | Method | Best Task 1 | Best Task 2 |
|----------|--------|------------|------------|
| `PCA.ipynb` | PCA + nearest-centroid | ~75% | — |
| `Bayes.ipynb` | PCA/MDA + Gaussian Bayes (QDA-style, regularized) | ~80% | ~88% |
| `K-NN.ipynb` | PCA/MDA + K-nearest neighbors | ~85% | ~89% |
| `KernelSVM.ipynb` | PCA/MDA + kernel SVM (linear/RBF/polynomial) | ~89% | ~91.25% |
| `MDA.ipynb` | MDA standalone | baseline | ~86% |
| `adaboost-svm.ipynb` | AdaBoost with custom SVM weak learners | ~94% | ~90% |

All notebooks are **self-contained** — each includes `compute_pca`, `compute_mda`, and `separate_train_test_manual` implementations so they run independently.

**Custom SVM implementation:** CVXOPT (quadratic programming) showed numerical instability with high-degree polynomial kernels, motivating a custom **gradient-descent SVM** used throughout the AdaBoost experiments.

**Data:** `data.mat`, `pose.mat`, `illumination.mat` are included in the repository.

**Files:**
- `PCA.ipynb`, `Bayes.ipynb`, `K-NN.ipynb`, `KernelSVM.ipynb`, `MDA.ipynb`, `adaboost-svm.ipynb`
- `report.pdf` — full written report

---

### 3. KIVI: 2-Bit KV-Cache Quantization for LLaMA (7B & 13B)

`kv_cache_kivi/`

Implements and evaluates **KIVI** — a training-free, 2-bit KV-cache quantization scheme applied to LLaMA-2 7B and 13B.

**Key idea:** Replace full-precision (FP16) attention key-value tensors with 2-bit group-quantized representations, keeping a small residual buffer of recent full-precision tokens. No retraining required.

**Implementation:**
- `quantize_per_token` / `quantize_per_channel` — 2-bit asymmetric quantization with configurable group size
- `KIVICache` — quantized KV-cache manager with residual buffer and on-the-fly dequantization
- `KIVILlamaAttention` — drop-in replacement for LLaMA self-attention
- `replace_llama_attention_with_kivi` — applies KIVI to every attention layer in a loaded model

**Benchmarks evaluated:**

| Benchmark | Model | Metric |
|-----------|-------|--------|
| CNN/DailyMail | LLaMA-2 7B | ROUGE-L, BERTScore, token match rate |
| GSM8K | LLaMA-2 13B | Exact match accuracy |
| CoQA | LLaMA-2 7B | Exact match accuracy |

Memory analysis includes theoretical KV-cache reduction (~8× for 2-bit vs FP16) and empirical peak GPU memory profiling.

Pre-computed results are stored in `results/`.

**Note:** This was my individual contribution to a team project on KV-cache efficiency methods. The full team report (which also covers H2O, Streaming-LLM, ZipCache, and StreamingSliding) is included as `report.pdf`.

**Files:**
- `kivi_llama_7b_13b.ipynb` — full KIVI implementation and evaluation
- `results/` — pre-computed JSON results (CNN/DM, GSM8K, CoQA)
- `report.pdf` — team report (CMSC 723)

---

## Repository Structure

```
Selected_projects/
├── image_classification/
│   ├── image_classification.ipynb
│   └── report.pdf
├── face_recognition/
│   ├── PCA.ipynb
│   ├── Bayes.ipynb
│   ├── K-NN.ipynb
│   ├── KernelSVM.ipynb
│   ├── MDA.ipynb
│   ├── adaboost-svm.ipynb
│   ├── data.mat          (200 subjects × 3 images, 24×21 pixels)
│   ├── illumination.mat  (illumination variation subset)
│   ├── pose.mat          (pose variation subset)
│   └── report.pdf
└── kv_cache_kivi/
    ├── kivi_llama_7b_13b.ipynb
    ├── results/
    └── report.pdf
```

---

## Installation

```bash
# Image classification and face recognition
pip install numpy scipy matplotlib scikit-learn tensorflow

# KIVI (requires GPU)
pip install torch transformers datasets rouge-score bert-score
# HuggingFace token required for LLaMA-2 weights
```

---

## Citation

```bibtex
@misc{amirian2025selected,
  author = {Kiyana Amirian},
  title  = {Selected Course Projects: Image Classification, Face Recognition, and KIVI KV-Cache Quantization},
  year   = {2025},
  url    = {https://github.com/kamirian/Selected_projects}
}
```
