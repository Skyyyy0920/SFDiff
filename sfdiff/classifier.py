"""
Classifier for SFDiff using SC-FC fused representations.

Passes real FC through the pretrained denoiser (at t=1) to obtain
cross-attention fused SC-FC embeddings, then classifies via MLP head.

Architecture:
    Real FC + SC graph → Denoiser (t=1) → Z_out [B, N, d_model]
    Mean Pool → z [B, d_model]
    MLP Head → logits [B, num_classes]
"""

import torch
import torch.nn as nn
from typing import Dict


class LatentClassifier(nn.Module):
    """Classifier using SC-FC fused representations from pretrained SFDiff.

    The denoiser's cross-attention layers learned to fuse SC structure with
    FC patterns during diffusion training. At classification time, we pass
    the real (clean) FC through the same pipeline to extract a joint
    SC-FC representation, then classify with an MLP head.

    Args:
        sfdiff_model: Pretrained SFDiffModel instance.
        d_model: Model dimension (must match SFDiff's d_model).
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension of the MLP head.
        dropout: Dropout rate in the MLP head.
        freeze_encoder: If True, freeze all denoiser weights (encoder + attention).
    """

    def __init__(
        self,
        sfdiff_model: nn.Module,
        d_model: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Keep the full denoiser (SC encoder + FC encoder + cross/self attention)
        self.denoiser = sfdiff_model.denoiser

        if freeze_encoder:
            for param in self.denoiser.parameters():
                param.requires_grad = False

        self.pool_dropout = nn.Dropout(dropout * 0.5)

        # MLP classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        """Forward pass: fused SC-FC encoding -> mean pool -> classification.

        Args:
            batch: Dict with:
                F_raw [B, N, N]: real FC matrix
                node_feat, in_degree, out_degree, path_data, dist, attn_mask:
                    Graphormer SC graph inputs.

        Returns:
            logits: [B, num_classes] classification logits.
        """
        fc = batch['F_raw']  # [B, N, N] real FC

        # Get fused SC-FC representation from pretrained denoiser
        Z_out = self.denoiser.get_fused_embedding(fc, batch)  # [B, N, d_model]

        # Mean pool over nodes
        graph_repr = Z_out.mean(dim=1)  # [B, d_model]
        graph_repr = self.pool_dropout(graph_repr)

        logits = self.head(graph_repr)  # [B, num_classes]
        return logits


if __name__ == '__main__':
    print("=== classifier.py sanity check ===")

    B, N = 2, 10
    d_model = 32
    max_path_len = 5

    from sfdiff.sfdiff import SFDiffModel
    sfdiff_model = SFDiffModel(
        N=N, d_model=d_model, nhead=4,
        num_sc_layers=1, num_self_attn_layers=1,
        dim_feedforward=64, T=10,
    )

    classifier = LatentClassifier(
        sfdiff_model=sfdiff_model,
        d_model=d_model,
        num_classes=2,
        hidden_dim=64,
        dropout=0.2,
        freeze_encoder=True,
    )

    # Check frozen parameters
    frozen_params = sum(1 for p in classifier.denoiser.parameters() if not p.requires_grad)
    total_denoiser_params = sum(1 for p in classifier.denoiser.parameters())
    print(f"Denoiser: {frozen_params}/{total_denoiser_params} params frozen")
    assert frozen_params == total_denoiser_params

    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

    # Forward pass with real FC
    batch = {
        'F_raw': torch.randn(B, N, N).clamp(-1, 1),
        'node_feat': torch.randn(B, N, N),
        'in_degree': torch.randint(0, 10, (B, N)),
        'out_degree': torch.randint(0, 10, (B, N)),
        'path_data': torch.randn(B, N, N, max_path_len, 1),
        'dist': torch.randint(0, 5, (B, N, N)),
        'attn_mask': torch.zeros(B, N + 1, N + 1, dtype=torch.bool),
    }

    logits = classifier(batch)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (B, 2)

    # Check gradient flow
    loss = logits.sum()
    loss.backward()
    head_has_grad = all(p.grad is not None for p in classifier.head.parameters())
    denoiser_has_grad = any(p.grad is not None for p in classifier.denoiser.parameters())
    print(f"Head gradient: {'OK' if head_has_grad else 'FAILED'}")
    print(f"Denoiser gradient: {'NONE (frozen, correct)' if not denoiser_has_grad else 'HAS GRAD'}")

    print("\nAll sanity checks passed!")
