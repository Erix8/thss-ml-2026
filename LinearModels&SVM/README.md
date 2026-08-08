# ML2026 HW1: Linear Models & SVM

## Overview

This programming assignment covers fundamental supervised learning methods in two parts:

1. **Linear Models** — data preprocessing, basis function expansion, linear regression with gradient descent, and multiclass softmax classification.
2. **Support Vector Machines** — linear SVM and kernel SVM (RBF) with stochastic subgradient descent for text sentiment classification.

---

## Files

| File | Description |
|------|-------------|
| `linear/start_code.py` | Linear regression (full-batch GD, SGD), multiclass softmax, model selection |
| `linear/softmax_util.py` | Softmax training utility with model selection (early stopping by val loss/acc) |
| `linear/generate_data.py` | Synthetic data generator with configurable nonlinear relationships |
| `linear/train.csv` / `linear/test.csv` | Generated synthetic regression + classification datasets |
| `svm/start_code.py` | Linear SVM and kernel SVM with stochastic subgradient descent |
| `svm/data_train.csv` / `svm/data_val.csv` | Emotion text classification dataset (joy vs sadness) |
| `requirements.txt` | Python dependencies |

---

## 1. Linear Models (`linear/`)

### Problem

Build linear regression and multiclass classification models from scratch:

- **2.1 Data preprocessing:** split data into train/val/test, normalize features to [0,1].
- **2.2 Basis expansion:** construct 9 nonlinear basis functions from x1, x2, x3: `[x1, x2, x3, cos(x1), cos(x2), cos(x3), x1^2, x2^2, x3^2]`.
- **2.3 Linear regression:** implement ridge regression ($L_2$ regularization) with:
  - Full-batch gradient descent + gradient checker.
  - Stochastic gradient descent (SGD) with validation monitoring and hyperparameter tuning.
- **2.4 Multiclass softmax:** train a 4-class softmax classifier using PyTorch with model selection.

### Implementation

**`split_data`** — Splits data by cumulative ratios using `np.split`. Supports optional shuffling with a random seed.

**`feature_normalization`** — Min-max normalization to [0,1]: `(x - train_min) / train_range`. The same transformation (derived from training data only) is used for val and test.

**`build_basis_features`** — Constructs the 9-column design matrix `Phi` using `np.column_stack` from the first 3 raw features. This allows linear regression to fit nonlinear patterns.

**Linear regression with gradient descent:**

| Component | Description |
|-----------|-------------|
| Loss | Ridge: $\frac{1}{2n}\sum (X\theta - y)^2 + \frac{\lambda}{2}\|\theta\|_2^2$ |
| Gradient | $\frac{1}{n}X^T(X\theta - y) + \lambda \theta$ |
| `grad_checker` | Verifies analytical gradient via central finite difference: $\frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon}$ |
| Full-batch GD | `theta = theta - alpha * grad`, iterates over all data each step |
| SGD | Mini-batches with `np.random.choice`, same update rule |

**Multiclass softmax** (`softmax_util.py`):
- Single `nn.Linear` layer (no bias) + `CrossEntropyLoss` + Adam optimizer.
- Model selection by best validation loss or accuracy, with `copy.deepcopy` snapshots.

### Usage

```bash
cd linear
python start_code.py --basis_reg    # run basis expansion + regression (2.1-2.3)
python start_code.py --grad_check   # run gradient checker
python start_code.py --gd_search    # run full-batch GD hyperparameter search
python start_code.py --sgd          # run SGD with batch size sweep
python start_code.py --classification  # run multiclass softmax (2.4)
```

---

## 2. SVM (`svm/`)

### Problem

Implement SVMs for binary text sentiment classification (joy vs sadness):

- **3.1 Data loading & vectorization:** load emotion-labeled text from CSV, convert to TF-IDF feature vectors.
- **3.2 Linear SVM:** train with stochastic subgradient descent on the hinge loss.
- **3.3 Kernel SVM:** RBF (Gaussian) kernel SVM using the representer theorem.

### Implementation

**`load_text_dataset`** — Reads CSV, filters rows by positive/negative emotion labels, maps to {+1, -1}.

**`vectorize`** — Applies `TfidfVectorizer` with English stop words and smooth IDF.

**`gaussian_kernel_matrix`** — Efficiently computes $K(X_1, X_2) = \exp(-\gamma \|X_1 - X_2\|^2)$ via the squared distance trick: $\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2x_i^T x_j$.

**Linear SVM — `linear_svm_subgrad_descent`**

- Hinge loss: $L = \frac{\lambda}{2}\|\theta\|^2 + \frac{1}{n}\sum \max(0, 1 - y_i \cdot \theta^T x_i)$
- Subgradient: only violating samples ($y_i \cdot \theta^T x_i < 1$) contribute.
- Mini-batch SGD with optional validation loss evaluation every `eval_every` iterations.

**Kernel SVM — `kernel_svm_subgrad_descent`**

- Representer theorem: decision function $f(x) = \sum_{j} \alpha_j y_j K(x_j, x) + b$
- Precomputes the full training kernel matrix to avoid redundant computation.
- Regularization term: $\frac{\lambda}{2} (\alpha \odot y)^T K (\alpha \odot y)$
- Bias $b$ is unregularized (gradient = 0 in the regularization term).

### Usage

```bash
cd svm
python start_code.py --linear-svm    # train linear SVM
python start_code.py --kernel-svm    # train kernel SVM (RBF)
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tqdm`.

