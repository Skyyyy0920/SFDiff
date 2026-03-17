"""
Top-level SFDiff model combining the denoiser and SC-guided diffusion process.

Provides:
    - forward(batch, device): Training loss computation
    - sample(batch, device): Generate FC matrices from SC
    - get_sc_embedding(batch): Extract SC encoder features for classification
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .diffusion import SCGuidedDiffusion
from .denoiser import SFDiffDenoiser


class SFDiffModel(nn.Module):
    """Top-level SFDiff model.

    Combines the SFDiffDenoiser (neural network) with the SCGuidedDiffusion
    (forward/reverse process) into a unified interface for training and sampling.

    Args:
        N: Number of brain regions (nodes).
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_sc_layers: Number of Graphormer layers in SC encoder.
        num_self_attn_layers: Number of self-attention layers after cross-attention.
        dim_feedforward: Feedforward dimension in transformers.
        dropout: Dropout rate.
        T: Number of diffusion timesteps.
        beta_min: Minimum diffusion time parameter.
        beta_max: Maximum diffusion time parameter.
        sigma: Fixed noise standard deviation.
        schedule: Noise schedule type ('linear' or 'cosine').
        max_degree: Max node degree for Graphormer.
        max_path_len: Max shortest-path length for Graphormer.
    """

    def __init__(
        self,
        N: int,
        d_model: int = 64,
        nhead: int = 4,
        num_sc_layers: int = 2,
        num_self_attn_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        T: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.5,
        sigma: float = 0.1,
        schedule: str = 'linear',
        max_degree: int = 511,
        max_path_len: int = 5,
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.denoiser = SFDiffDenoiser(
            N=N,
            d_model=d_model,
            nhead=nhead,
            num_sc_layers=num_sc_layers,
            num_self_attn_layers=num_self_attn_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )

        self.diffusion = SCGuidedDiffusion(
            T=T,
            beta_min=beta_min,
            beta_max=beta_max,
            sigma=sigma,
            schedule=schedule,
        )

    def forward(
        self,
        batch: Dict,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Training forward pass: compute x0-prediction loss.

        Samples random timesteps, computes noisy FC via forward diffusion,
        and the network directly predicts x_0 (not noise). Loss is element-wise
        MSE averaged over all dimensions for numerical stability.

        Args:
            batch: Dict from dataloader with keys:
                F_raw [B,N,N], eigval [B,N], eigvec [B,N,N],
                node_feat, in_degree, out_degree, path_data, dist, attn_mask.
            device: Target device.

        Returns:
            loss: Scalar, mean squared error on x0 prediction.
        """
        x0 = batch['F_raw'].to(device)       # [B, N, N]
        eigval = batch['eigval'].to(device)   # [B, N]
        eigvec = batch['eigvec'].to(device)   # [B, N, N]
        B = x0.shape[0]

        # Sample random timesteps uniformly from [1, T]
        t = torch.randint(1, self.diffusion.T + 1, (B,), device=device)

        # Forward diffusion: get noisy FC
        x_t, _ = self.diffusion.q_sample(x0, t, eigval, eigvec)

        # Network predicts x0 (not noise)
        x0_hat = self.denoiser(x_t, t, batch)

        # Frobenius mean loss (element-wise MSE, numerically stable & cross-dataset comparable)
        loss = ((x0 - x0_hat) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch: Dict,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Generate FC matrices given SC information.

        Runs the full reverse diffusion process from noise to clean FC.

        Args:
            batch: Dict with SC graph info, eigendecomposition, and Graphormer inputs.
            device: Target device.

        Returns:
            x_0: [B, N, N] generated FC matrices (symmetric, in [-1, 1], zero diagonal).
        """
        self.eval()
        eigval = batch['eigval'].to(device)
        eigvec = batch['eigvec'].to(device)
        B = eigval.shape[0]

        x_0 = self.diffusion.sample_loop(
            model=self.denoiser,
            shape=(B, self.N, self.N),
            eigval=eigval,
            eigvec=eigvec,
            batch=batch,
            device=device,
        )
        return x_0

    def get_sc_embedding(self, batch: Dict) -> torch.Tensor:
        """Extract SC encoder embeddings for Mode B classifier.

        Args:
            batch: Dict with Graphormer inputs.

        Returns:
            Z_SC: [B, N, d_model] node-level SC embeddings.
        """
        return self.denoiser.get_sc_embedding(batch)


if __name__ == '__main__':
    print("=== sfdiff.py sanity check ===")

    B, N = 2, 10
    d_model = 32
    T = 10  # very small for testing
    max_path_len = 5

    model = SFDiffModel(
        N=N, d_model=d_model, nhead=4,
        num_sc_layers=1, num_self_attn_layers=1,
        dim_feedforward=64, dropout=0.1,
        T=T, beta_min=1e-4, beta_max=0.5, sigma=0.1,
    )

    # Symmetric synthetic FC
    F_raw = torch.randn(B, N, N) * 0.3
    F_raw = (F_raw + F_raw.transpose(-2, -1)) / 2.0
    F_raw = F_raw.clamp(-1, 1)

    batch = {
        'F_raw': F_raw,
        'eigval': torch.rand(B, N).clamp(min=0),
        'eigvec': torch.randn(B, N, N),
        'node_feat': torch.randn(B, N, N),
        'in_degree': torch.randint(0, 10, (B, N)),
        'out_degree': torch.randint(0, 10, (B, N)),
        'path_data': torch.randn(B, N, N, max_path_len, 1),
        'dist': torch.randint(0, 5, (B, N, N)),
        'attn_mask': torch.zeros(B, N + 1, N + 1, dtype=torch.bool),
        'labels': torch.zeros(B, 1),
    }

    # --- Verification 1: forward loss is scalar and reasonable ---
    print("\n[Verification 1] forward loss (x0-prediction):")
    loss = model(batch, device='cpu')
    print(f"  Training loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 10.0, f"Loss too large for N={N}: {loss.item():.4f} (expect 0.01~1.0 range)"
    print(f"  Loss in expected range [0, 10): OK")

    # Test backward
    loss.backward()
    print("  Backward pass: OK")

    # --- Verification 2: sample generates valid FC ---
    print("\n[Verification 2] sample output quality:")
    with torch.no_grad():
        generated = model.sample(batch, device='cpu')
    print(f"  Generated FC shape: {generated.shape}")
    assert generated.shape == (B, N, N)

    # Symmetry
    sym_err = (generated - generated.transpose(-2, -1)).abs().max().item()
    print(f"  Symmetry error: {sym_err:.2e} (should be < 1e-5)")
    assert sym_err < 1e-5, f"Symmetry error too large: {sym_err}"

    # Value range
    print(f"  Value range: [{generated.min():.3f}, {generated.max():.3f}]")
    assert generated.min() >= -1.0 and generated.max() <= 1.0, "Values out of [-1, 1]"

    # Zero diagonal
    diag_err = generated.diagonal(dim1=-2, dim2=-1).abs().max().item()
    print(f"  Diagonal max: {diag_err:.2e} (should be 0)")
    assert diag_err < 1e-6, f"Diagonal not zero: {diag_err}"

    # SC embedding
    Z_SC = model.get_sc_embedding(batch)
    print(f"  SC embedding shape: {Z_SC.shape}")
    assert Z_SC.shape == (B, N, d_model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params}")

    print("\nAll sanity checks passed!")
