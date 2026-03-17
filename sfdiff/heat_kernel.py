"""
SC Laplacian computation, eigendecomposition, and heat kernel utilities for SFDiff.

The heat kernel H_t = V diag(exp(-beta_t * lambda)) V^T is the core mathematical
object that defines the SC-guided forward diffusion process. Here V and lambda are
the eigenvectors and eigenvalues of the symmetric normalized Laplacian of the
structural connectivity (SC) matrix.
"""

import torch
from typing import Tuple


def compute_sc_laplacian(sc: torch.Tensor) -> torch.Tensor:
    """Compute the symmetric normalized Laplacian of an SC matrix.

    L = I - D^{-1/2} A D^{-1/2}

    where A is the SC matrix with diagonal zeroed out, and D is the degree matrix.

    Args:
        sc: [N, N] non-negative structural connectivity matrix.

    Returns:
        L: [N, N] symmetric normalized Laplacian. Eigenvalues lie in [0, 2].
    """
    N = sc.shape[0]
    A = sc.clone()
    A.fill_diagonal_(0.0)

    degree = A.sum(dim=1)  # [N]
    D_inv_sqrt = 1.0 / (degree.sqrt() + 1e-8)  # [N]

    # L = I - D^{-1/2} A D^{-1/2}
    # (D^{-1/2} A D^{-1/2})_{ij} = D_inv_sqrt[i] * A[i,j] * D_inv_sqrt[j]
    L_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)  # [N, N]
    L = torch.eye(N, device=sc.device, dtype=sc.dtype) - L_norm

    return L


def precompute_eigen(sc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute eigendecomposition of the SC Laplacian.

    This should be called once at dataset initialization time and the results
    cached per sample. Do NOT call during training forward pass (O(N^3)).

    Args:
        sc: [N, N] normalized SC matrix (non-negative).

    Returns:
        eigval: [N] eigenvalues, clamped to >= 0 for numerical stability.
        eigvec: [N, N] corresponding eigenvectors (columns).
    """
    L = compute_sc_laplacian(sc)
    eigval, eigvec = torch.linalg.eigh(L)
    eigval = eigval.clamp(min=0.0)  # fix floating-point negative near-zero
    return eigval, eigvec


def compute_heat_kernel(
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute heat kernel H = V diag(exp(-beta * lambda)) V^T.

    Args:
        eigval: [N] eigenvalues of L_SC (non-negative).
        eigvec: [N, N] eigenvectors of L_SC.
        beta: Diffusion time parameter (scalar).

    Returns:
        H: [N, N] heat kernel matrix (symmetric, rows sum ~ 1).
    """
    exp_eigval = torch.exp(-beta * eigval)  # [N]
    # H = V @ diag(exp_eigval) @ V^T
    H = eigvec @ (exp_eigval.unsqueeze(-1) * eigvec.T)  # [N, N]
    return H


def compute_heat_kernel_batch(
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Compute heat kernels for a batch with per-sample beta values.

    Args:
        eigval: [B, N] eigenvalues per sample.
        eigvec: [B, N, N] eigenvectors per sample.
        beta: [B] diffusion time per sample, or scalar.

    Returns:
        H: [B, N, N] heat kernel matrices.
    """
    if beta.dim() == 0:
        beta = beta.unsqueeze(0).expand(eigval.shape[0])
    # beta: [B] -> [B, N]
    exp_eigval = torch.exp(-beta.unsqueeze(1) * eigval)  # [B, N]
    # H = V @ diag(exp) @ V^T in batch form
    # eigvec: [B, N, N], exp_eigval: [B, N]
    # (exp_eigval.unsqueeze(-1) * eigvec.transpose(-2, -1)): [B, N, N]
    H = eigvec @ (exp_eigval.unsqueeze(-1) * eigvec.transpose(-2, -1))  # [B, N, N]
    return H


def compute_heat_kernel_inv_batch(
    eigval: torch.Tensor,
    eigvec: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Compute inverse heat kernel H^{-1} = V diag(exp(+beta * lambda)) V^T.

    Used in the reverse diffusion process to estimate x_0 from x_t.

    Args:
        eigval: [B, N] eigenvalues per sample.
        eigvec: [B, N, N] eigenvectors per sample.
        beta: [B] diffusion time per sample, or scalar.

    Returns:
        H_inv: [B, N, N] inverse heat kernel matrices.
    """
    if beta.dim() == 0:
        beta = beta.unsqueeze(0).expand(eigval.shape[0])
    exp_eigval = torch.exp(beta.unsqueeze(1) * eigval)  # [B, N]
    # Clamp to prevent overflow for large eigenvalues * beta
    exp_eigval = exp_eigval.clamp(max=1e6)
    H_inv = eigvec @ (exp_eigval.unsqueeze(-1) * eigvec.transpose(-2, -1))
    return H_inv


if __name__ == '__main__':
    # Quick sanity check
    print("=== heat_kernel.py sanity check ===")
    N = 10
    # Create a random symmetric SC matrix
    sc = torch.rand(N, N)
    sc = (sc + sc.T) / 2
    sc.fill_diagonal_(0.0)
    sc = sc.clamp(min=0.0)

    # Test Laplacian
    L = compute_sc_laplacian(sc)
    print(f"Laplacian shape: {L.shape}")
    eigval, eigvec = precompute_eigen(sc)
    print(f"Eigenvalues range: [{eigval.min():.6f}, {eigval.max():.6f}]")
    assert (eigval >= -1e-7).all(), "Eigenvalues should be non-negative"

    # Reconstruct Laplacian
    L_recon = eigvec @ torch.diag(eigval) @ eigvec.T
    recon_err = (L - L_recon).abs().max().item()
    print(f"Laplacian reconstruction error: {recon_err:.2e}")
    assert recon_err < 1e-5, "Reconstruction error too large"

    # Test heat kernel at beta=0 (should be identity)
    H0 = compute_heat_kernel(eigval, eigvec, beta=0.0)
    identity_err = (H0 - torch.eye(N)).abs().max().item()
    print(f"H(beta=0) vs I error: {identity_err:.2e}")
    assert identity_err < 1e-5, "H(beta=0) should be identity"

    # Test heat kernel symmetry
    H1 = compute_heat_kernel(eigval, eigvec, beta=0.5)
    sym_err = (H1 - H1.T).abs().max().item()
    print(f"H(beta=0.5) symmetry error: {sym_err:.2e}")
    assert sym_err < 1e-6, "Heat kernel should be symmetric"

    # Test row sums (approximately 1 for connected graphs)
    row_sums = H1.sum(dim=1)
    print(f"H(beta=0.5) row sums: mean={row_sums.mean():.4f}, std={row_sums.std():.4f}")

    # Test batch version
    B = 3
    eigval_b = eigval.unsqueeze(0).expand(B, -1)
    eigvec_b = eigvec.unsqueeze(0).expand(B, -1, -1)
    beta_b = torch.tensor([0.0, 0.1, 0.5])
    H_batch = compute_heat_kernel_batch(eigval_b, eigvec_b, beta_b)
    print(f"Batch heat kernel shape: {H_batch.shape}")
    assert H_batch.shape == (B, N, N)

    # Test inverse
    H_inv = compute_heat_kernel_inv_batch(eigval_b, eigvec_b, beta_b)
    # H @ H_inv should be approximately I
    product = H_batch[1] @ H_inv[1]
    inv_err = (product - torch.eye(N)).abs().max().item()
    print(f"H @ H_inv vs I error (beta=0.1): {inv_err:.2e}")
    assert inv_err < 1e-4, "H @ H_inv should be approximately identity"

    print("\nAll sanity checks passed!")
