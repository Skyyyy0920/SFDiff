# SFDiff Project — Claude Code Instructions

## Project Overview

This project implements **SFDiff (Structure-guided Functional Diffusion)**, a novel generative model
for brain functional connectivity (FC) matrices. The key innovation is replacing the isotropic Gaussian
forward process of standard DDPM with a **SC-guided heat diffusion process**, where the structural
connectivity (SC) Laplacian serves as the physical diffusion kernel.

**Target venue**: NeurIPS  
**Paper title**: "Structure as Prior: Heat Kernel Diffusion for Anatomically Constrained Brain Functional Network Generation"

---

## Repository Layout

```
SFDiff/
├── CLAUDE.md                  ← You are here
├── sfdiff/
│   ├── __init__.py
│   ├── heat_kernel.py         ← SC Laplacian + heat kernel computation
│   ├── diffusion.py           ← Forward/reverse process, noise schedule
│   ├── denoiser.py            ← Denoising network architecture
│   ├── sfdiff.py              ← Top-level SFDiff model
│   └── classifier.py          ← Downstream classifier (latent-space or augmentation-based)
├── data/
│   └── dataset.py             ← Reuses ABCDDataset / PPMIDataset from reference codebase
├── train_diffusion.py         ← Training loop for SFDiff (generative phase)
├── train_classifier.py        ← Training loop for downstream classification
├── sample.py                  ← Sampling / generation script
├── evaluate.py                ← Metrics: MMD, MAE, classification AUC/F1
├── run.py                     ← Multi-seed experiment runner
└── requirements.txt
```

---

## Method: SFDiff

### Core Idea

Standard DDPM forward process: $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I)$

**SFDiff forward process** (SC-guided):
$$q(x_t \mid x_0, L_{SC}) = \mathcal{N}(H_t\, x_0,\; \sigma_t^2\, I), \qquad H_t = e^{-\beta_t L_{SC}}$$

where:
- $L_{SC}$ = symmetric normalized graph Laplacian of the SC matrix
- $H_t$ = graph heat kernel, computed via eigendecomposition of $L_{SC}$:
  $H_t = V \cdot \mathrm{diag}(e^{-\beta_t \lambda_i}) \cdot V^\top$
- $\beta_t$ increases with $t$ following a linear or cosine schedule
- At $t=0$: $H_0 \approx I$ (identity, preserve original FC)
- At $t=T$: $H_T \approx \mathbf{1}\mathbf{1}^\top / N$ (fully diffused along SC)

### Noise Schedule

Linear schedule for $\beta_t$:
$$\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min}), \quad t=1,\ldots,T$$

Suggested defaults: $T=1000$, $\beta_{\min}=0.0001$, $\beta_{\max}=0.5$, $\sigma_t = 0.1$.

### Denoising Network Architecture

Input: noisy FC $x_t \in \mathbb{R}^{N \times N}$, SC graph $G_{SC}$, timestep $t$

```
SC Encoder (Graphormer, reuse from reference codebase)
    G_SC  →  Z_SC  ∈ R^{N × d_model}

FC Encoder (row-wise MLP + positional time embedding)
    x_t   →  Z_FC  ∈ R^{N × d_model}

Cross-Attention Block (FC queries, SC keys/values)
    Q = Z_FC W_Q
    K = Z_SC W_K
    V = Z_SC W_V
    Z_cross = softmax(QK^T / sqrt(d)) V

Self-Attention Block (on Z_cross)
    Z_out = TransformerEncoder(Z_cross)

Output Head (reconstruct N×N noise matrix)
    Z_out → Linear → reshape → eps_hat ∈ R^{N × N}
```

### Training Objective

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \varepsilon \sim \mathcal{N}(0,I)}\left[\left\| \varepsilon - \hat\varepsilon_\theta(H_t x_0 + \sigma_t \varepsilon,\; t,\; G_{SC}) \right\|_F^2\right]$$

**Note**: The FC matrix is symmetric. After generating $\hat\varepsilon$, symmetrize: $\hat\varepsilon \leftarrow (\hat\varepsilon + \hat\varepsilon^\top) / 2$.

### Downstream: Two Modes

**Mode A — Augmentation**: Generate $K=5$ synthetic FC per training sample → feed augmented dataset to the existing DIAL classifier.

**Mode B — Latent Classification**: Freeze SFDiff SC encoder, add MLP head on top of `mean(Z_SC, dim=0)`, fine-tune for classification. This is the primary evaluation mode.

---

## Data Format

Reuse the existing data pipeline. Each sample is a dict:
```python
{
    'SC': np.ndarray,   # (N, N) structural connectivity, non-negative
    'FC': np.ndarray,   # (N, N) functional connectivity, values in [-1, 1]
    'label': int,        # 0 or 1
    'name': str
}
```

Data loading: copy `dial/data.py` → `data/dataset.py`, keep `ABCDDataset`, `PPMIDataset`,
`normalize_sc`, `filter_fc`, `load_data`, `preprocess_labels`, `balance_dataset`, `split_dataset`.

**FC preprocessing for diffusion**: Do NOT apply `filter_fc` thresholding for the generative model.
Use raw FC values, clipped to [-1, 1]. Only apply filtering for downstream classification.

---

## Key Implementation Details

### heat_kernel.py

```python
def compute_sc_laplacian(sc: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized Laplacian of SC matrix."""
    # sc: [N, N], non-negative
    sc = sc.clone()
    sc.fill_diagonal_(0.0)
    degree = sc.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + 1e-8))
    L = torch.eye(N) - D_inv_sqrt @ sc @ D_inv_sqrt
    return L

def compute_heat_kernel(L: torch.Tensor, beta: float) -> torch.Tensor:
    """H = V diag(exp(-beta * lambda)) V^T"""
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    exp_eigenvalues = torch.exp(-beta * eigenvalues)
    H = eigenvectors @ torch.diag(exp_eigenvalues) @ eigenvectors.T
    return H
```

**Performance note**: Eigendecomposition is O(N^3). For N=90 (PPMI) or N=360 (Glasser/ABCD),
precompute and cache eigenvectors per sample at dataset initialization time. Store as `graph.ndata['eigvec']`
and `graph.ndata['eigval']` in the DGL graph. Do NOT recompute during training.

### diffusion.py

```python
class SCGuidedDiffusion:
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.5, sigma=0.1):
        ...
    
    def q_sample(self, x0, t, L_SC):
        """Sample x_t given x_0 and L_SC."""
        beta_t = self.get_beta(t)
        H_t = compute_heat_kernel(L_SC, beta_t)
        mean = H_t @ x0  # [N, N] — apply row-wise
        noise = torch.randn_like(x0)
        return mean + self.sigma * noise, noise
    
    def p_sample(self, model, xt, t, G_SC):
        """One reverse step."""
        eps_hat = model(xt, t, G_SC)
        ...
```

**Important**: `H_t @ x0` means applying the heat kernel to each row of the FC matrix
(treating rows as node signals). This is correct because FC[i, j] represents the
signal between region i and region j, and heat diffusion acts on node signals.

### denoiser.py

- **SC Encoder**: `GraphormerNodeEncoder` from reference codebase (copy from `dial/model.py`)
  with `N`, `d_model=64`, `nhead=4`, `num_layers=2`.
- **Time Embedding**: Sinusoidal, projected to `d_model` via 2-layer MLP.
- **FC Encoder**: `nn.Linear(N, d_model)` applied row-wise + time embedding added.
- **Cross-Attention**: `nn.MultiheadAttention(d_model, nhead, batch_first=True)`.
- **Output**: `nn.Linear(d_model, N)` applied to each of N output tokens → reshape to [N, N] → symmetrize.

---

## Evaluation Metrics

### Generative Quality
- **MMD** (Maximum Mean Discrepancy): between real and generated FC matrices (use RBF kernel)
- **MAE**: element-wise mean absolute error between generated and real FC
- **SC-FC Coupling**: Pearson correlation between generated FC and SC (should be higher than baseline DDPM)

### Downstream Classification (primary)
- AUC, F1, Accuracy, Precision, Recall (5-seed mean ± std)
- Compare: SFDiff (Mode A aug) vs SFDiff (Mode B latent) vs DIAL vs baselines

---

## Coding Standards

- **Language**: Python 3.10+, PyTorch 2.4+
- **Style**: Follow the existing codebase style (type hints, docstrings, argparse)
- **Device handling**: All tensors must respect the `device` argument; no hardcoded `cuda`
- **Seed**: Always accept and propagate `random_seed` and `data_seed` as separate arguments
- **Logging**: Use Python `logging` module, same pattern as existing `setup_logger()`
- **No new heavy dependencies**: reuse existing deps from `requirements.txt`.
  If needed, only add `einops` (already listed).
- **Numerical stability**: Laplacian eigendecomposition may produce near-zero negative eigenvalues
  due to floating point; clamp eigenvalues: `eigenvalues = eigenvalues.clamp(min=0.0)` before
  computing heat kernel.
- **Symmetry enforcement**: After generating FC, always enforce symmetry:
  `fc = (fc + fc.T) / 2` and optionally zero the diagonal.

---

## Reference Codebase

The reference implementation lives in the same parent repo under `dial/`. Key files:
- `dial/model.py`: `GraphormerNodeEncoder`, `MaskedGraphTransformer` — reuse these directly
- `dial/data.py`: All data loading utilities — copy and extend
- `dial/utils.py`: `build_edge_index_from_S`, `laplacian_from_conductance` — useful helpers
- `baselines/models.py`: Baseline classifier heads — reuse `BaselineHead`
- `main.py`: Training loop pattern (logger, seed, metrics, checkpoint saving) — follow this pattern

Do NOT modify files in `dial/` or `baselines/` — only read from them.

---

## Step-by-Step Implementation Order

1. `data/dataset.py` — Copy and adapt data loading; add eigendecomposition caching
2. `sfdiff/heat_kernel.py` — Laplacian + heat kernel + caching utilities
3. `sfdiff/diffusion.py` — `SCGuidedDiffusion` class (forward/reverse process)
4. `sfdiff/denoiser.py` — Denoising network with cross-attention
5. `sfdiff/sfdiff.py` — Top-level model combining all components
6. `train_diffusion.py` — Training loop (generative phase, no labels needed)
7. `sample.py` — Sampling script to generate FC given SC
8. `sfdiff/classifier.py` — Mode B latent classifier
9. `train_classifier.py` — Classification training loop
10. `evaluate.py` — MMD, MAE, SC-FC coupling, classification metrics
11. `run.py` — Multi-seed runner

---

## Common Pitfalls to Avoid

1. **Do not apply heat kernel to the full N×N matrix as a 2D operation**. The correct
   interpretation: for each row $i$ of FC (treated as a length-N node signal), apply $H_t$.
   Equivalently: `x_t_mean = H_t @ x0` where `@` is standard matrix multiply. This is correct.

2. **Eigendecomposition must be cached**. Computing `torch.linalg.eigh` during training
   forward pass will bottleneck training. Cache at dataset init time.

3. **The FC matrix ranges in [-1, 1]**. When computing heat diffusion, the mean $H_t x_0$
   will also be in approximately [-1, 1] (heat kernel is a stochastic matrix-like operator,
   it contracts toward the mean). The added noise is Gaussian with fixed $\sigma$. This is fine.

4. **Do not confuse the two types of Laplacian**: the SC Laplacian $L_{SC}$ used in the
   diffusion forward process (neuroscience prior) vs the task Laplacian $L_T$ used in DIAL's
   energy computation. They are separate and serve different purposes.

5. **Batch handling**: Each sample has its own SC and thus its own $L_{SC}$ and $H_t$.
   Do NOT share a single heat kernel across a batch. Process per-sample, then stack.
