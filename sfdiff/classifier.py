"""
Mode B latent-space classifier for SFDiff.

Freezes the SC encoder from a pretrained SFDiff model and adds
an MLP classification head on top of mean-pooled SC embeddings.

Architecture:
    Frozen SC Encoder -> Z_SC [B, N, d_model]
    Mean Pool -> Z_graph [B, d_model]
    MLP Head -> logits [B, num_classes]
"""

import torch
import torch.nn as nn
from typing import Dict


class LatentClassifier(nn.Module):
    """Mode B latent-space classifier using frozen SFDiff SC encoder.

    Extracts the SC encoder from a pretrained SFDiff model, freezes its
    parameters, and trains only a lightweight MLP classification head
    on the mean-pooled node embeddings.

    Args:
        sfdiff_model: Pretrained SFDiffModel instance.
        d_model: Model dimension (must match SFDiff's d_model).
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension of the MLP head.
        dropout: Dropout rate in the MLP head.
        freeze_encoder: If True, freeze SC encoder weights.
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

        # Extract the SC encoder from the pretrained SFDiff denoiser
        self.sc_encoder = sfdiff_model.denoiser.sc_encoder

        if freeze_encoder:
            for param in self.sc_encoder.parameters():
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
        """Forward pass: SC encoding -> mean pool -> classification.

        Args:
            batch: Dict with Graphormer inputs:
                node_feat [B, N, N], in_degree [B, N], out_degree [B, N],
                path_data [B, N, N, max_path_len, 1], dist [B, N, N],
                attn_mask [B, N+1, N+1] (optional).

        Returns:
            logits: [B, num_classes] classification logits.
        """
        # SC encoder (possibly frozen)
        Z_SC = self.sc_encoder(
            node_feat=batch['node_feat'],
            in_degree=batch['in_degree'],
            out_degree=batch['out_degree'],
            path_data=batch['path_data'],
            dist=batch['dist'],
            attn_mask=batch.get('attn_mask'),
        )  # [B, N, d_model]

        # Mean pool over nodes
        graph_repr = Z_SC.mean(dim=1)  # [B, d_model]
        graph_repr = self.pool_dropout(graph_repr)

        logits = self.head(graph_repr)  # [B, num_classes]
        return logits


if __name__ == '__main__':
    # Quick sanity check
    print("=== classifier.py sanity check ===")

    B, N = 2, 10
    d_model = 32
    max_path_len = 5

    # Create a mock SFDiff model
    from sfdiff.sfdiff import SFDiffModel
    sfdiff_model = SFDiffModel(
        N=N, d_model=d_model, nhead=4,
        num_sc_layers=1, num_self_attn_layers=1,
        dim_feedforward=64, T=10,
    )

    # Build classifier
    classifier = LatentClassifier(
        sfdiff_model=sfdiff_model,
        d_model=d_model,
        num_classes=2,
        hidden_dim=64,
        dropout=0.2,
        freeze_encoder=True,
    )

    # Check frozen parameters
    frozen_params = sum(1 for p in classifier.sc_encoder.parameters() if not p.requires_grad)
    total_encoder_params = sum(1 for p in classifier.sc_encoder.parameters())
    print(f"SC encoder: {frozen_params}/{total_encoder_params} params frozen")
    assert frozen_params == total_encoder_params, "All encoder params should be frozen"

    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

    # Forward pass
    batch = {
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

    # Check gradient flow (only through head)
    loss = logits.sum()
    loss.backward()
    head_has_grad = all(
        p.grad is not None for p in classifier.head.parameters()
    )
    encoder_has_grad = any(
        p.grad is not None for p in classifier.sc_encoder.parameters()
    )
    print(f"Head gradient flow: {'OK' if head_has_grad else 'FAILED'}")
    print(f"Encoder gradient: {'NONE (correct)' if not encoder_has_grad else 'HAS GRAD (unexpected)'}")

    print("\nAll sanity checks passed!")
