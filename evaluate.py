"""
Evaluation metrics for SFDiff.

Generative quality metrics:
    - MMD (Maximum Mean Discrepancy) with RBF kernel
    - MAE (element-wise Mean Absolute Error)
    - SC-FC Coupling (Pearson correlation on upper triangle)

Classification metrics:
    - AUC, F1, Accuracy, Precision, Recall

Usage:
    python evaluate.py --mode generative --path ./generated/generated_fc.npz
    python evaluate.py --mode classification --path ./results/classifier/.../results.pkl
"""

import os
import argparse
import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from typing import Dict


def compute_mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    Args:
        X: [M, D] real FC matrices (flattened upper triangle).
        Y: [M', D] generated FC matrices (flattened upper triangle).
        gamma: RBF kernel bandwidth parameter.

    Returns:
        MMD^2 estimate (non-negative).
    """
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    m = X.shape[0]
    n = Y.shape[0]

    mmd = (K_XX.sum() / (m * m)
           + K_YY.sum() / (n * n)
           - 2.0 * K_XY.sum() / (m * n))
    return float(max(mmd, 0.0))  # clamp numerical noise


def compute_mae(real_fc: np.ndarray, gen_fc: np.ndarray) -> float:
    """Element-wise Mean Absolute Error between real and generated FC.

    If gen_fc has K times more samples than real_fc (K samples per real),
    the generated samples are averaged per-group before computing MAE.

    Args:
        real_fc: [M, N, N] real FC matrices.
        gen_fc: [M', N, N] generated FC matrices.

    Returns:
        MAE (non-negative scalar).
    """
    if gen_fc.shape[0] != real_fc.shape[0]:
        K = gen_fc.shape[0] // real_fc.shape[0]
        if K > 1:
            gen_fc = gen_fc.reshape(real_fc.shape[0], K, *gen_fc.shape[1:]).mean(axis=1)

    # Truncate to minimum length
    n = min(real_fc.shape[0], gen_fc.shape[0])
    return float(np.abs(real_fc[:n] - gen_fc[:n]).mean())


def compute_sc_fc_coupling(
    sc: np.ndarray,
    gen_fc: np.ndarray,
) -> float:
    """Pearson correlation between SC and generated FC (upper triangle).

    Higher coupling indicates better anatomical constraint. For each sample,
    compute the Pearson correlation between SC and generated FC upper triangle
    elements, then average across samples.

    Args:
        sc: [M, N, N] structural connectivity matrices.
        gen_fc: [M', N, N] generated FC matrices.

    Returns:
        Mean Pearson correlation (in [-1, 1]).
    """
    M = min(sc.shape[0], gen_fc.shape[0])
    N = sc.shape[1]
    triu_idx = np.triu_indices(N, k=1)

    correlations = []
    for i in range(M):
        sc_vec = sc[i][triu_idx]
        fc_vec = gen_fc[i][triu_idx]
        if sc_vec.std() > 1e-8 and fc_vec.std() > 1e-8:
            r, _ = pearsonr(sc_vec, fc_vec)
            if not np.isnan(r):
                correlations.append(r)

    return float(np.mean(correlations)) if correlations else 0.0


def evaluate_generative(npz_path: str) -> Dict[str, float]:
    """Evaluate generative quality from a saved .npz file.

    The .npz file should contain:
        - generated: [M', N, N] generated FC matrices
        - real_fc: [M, N, N] real FC matrices
        - sc: [M, N, N] SC matrices
        - num_samples_per_real: int (K)

    Args:
        npz_path: Path to the generated .npz file.

    Returns:
        Dict with 'mmd', 'mae', 'sc_fc_coupling'.
    """
    data = np.load(npz_path, allow_pickle=True)
    real_fc = data['real_fc']
    gen_fc = data['generated']
    sc = data['sc']

    N = real_fc.shape[1]
    triu_idx = np.triu_indices(N, k=1)

    # Flatten to upper triangle for MMD
    real_flat = np.array([r[triu_idx] for r in real_fc])

    # For MMD, use first M generated samples (one per real)
    K = gen_fc.shape[0] // real_fc.shape[0] if gen_fc.shape[0] > real_fc.shape[0] else 1
    gen_for_mmd = gen_fc[:real_fc.shape[0]]  # first M
    gen_flat = np.array([g[triu_idx] for g in gen_for_mmd])

    # Median heuristic for RBF kernel bandwidth
    n_sample = min(200, real_flat.shape[0])
    dists = euclidean_distances(real_flat[:n_sample], real_flat[:n_sample])
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (median_dist ** 2 + 1e-8)

    metrics = {
        'mmd': compute_mmd_rbf(real_flat, gen_flat, gamma=gamma),
        'mae': compute_mae(real_fc, gen_fc),
        'sc_fc_coupling': compute_sc_fc_coupling(sc, gen_for_mmd),
    }
    return metrics


def evaluate_classification(results_path: str) -> Dict[str, float]:
    """Load classification results from a pickle file.

    Args:
        results_path: Path to results.pkl from train_classifier.py.

    Returns:
        Dict with accuracy, precision, recall, f1, auc.
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    test_final = results.get('test_final', {})
    return {
        'accuracy': test_final.get('accuracy', 0.0),
        'precision': test_final.get('precision', 0.0),
        'recall': test_final.get('recall', 0.0),
        'f1': test_final.get('f1', 0.0),
        'auc': test_final.get('auc', 0.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFDiff Evaluation')
    parser.add_argument('--mode', type=str, choices=['generative', 'classification'],
                        required=True, help='Evaluation mode')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to .npz (generative) or .pkl (classification)')
    args = parser.parse_args()

    if args.mode == 'generative':
        metrics = evaluate_generative(args.path)
        print("=" * 50)
        print("Generative Quality Metrics:")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
    else:
        metrics = evaluate_classification(args.path)
        print("=" * 50)
        print("Classification Metrics:")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
