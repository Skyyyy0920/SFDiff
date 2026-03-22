"""
Classifier for SFDiff using dual-branch SC + FC representations.

Extracts SC and FC embeddings independently from the pretrained denoiser,
concatenates the mean-pooled representations, and classifies via MLP head.

Architecture:
    SC graph → SC Encoder (pretrained) → mean pool → z_sc [B, d]
    Real FC  → FC Encoder (pretrained) → mean pool → z_fc [B, d]
    concat(z_sc, z_fc) → [B, 2d] → MLP Head → logits [B, num_classes]
"""

import torch
import torch.nn as nn
from typing import Dict


class LatentClassifier(nn.Module):
    """Dual-branch classifier using pretrained SC and FC encoders.

    Both encoders are initialized from diffusion pretraining. The SC encoder
    learned structural graph representations; the FC encoder learned to project
    FC row vectors into the same latent space. Their concatenated mean-pooled
    features provide complementary structural and functional information.

    Args:
        sfdiff_model: Pretrained SFDiffModel instance.
        d_model: Model dimension (must match SFDiff's d_model).
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension of the MLP head.
        dropout: Dropout rate in the MLP head.
        freeze_encoder: If True, freeze SC and FC encoder weights.
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
        self.d_model = d_model

        # Extract pretrained encoders from the denoiser
        self.sc_encoder = sfdiff_model.denoiser.sc_encoder
        self.fc_encoder = sfdiff_model.denoiser.fc_encoder
        self.time_embed = sfdiff_model.denoiser.time_embed

        if freeze_encoder:
            for param in self.sc_encoder.parameters():
                param.requires_grad = False
            for param in self.fc_encoder.parameters():
                param.requires_grad = False
            for param in self.time_embed.parameters():
                param.requires_grad = False

        self.pool_dropout = nn.Dropout(dropout * 0.5)

        # MLP classification head: input is concat of z_sc and z_fc
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        """Forward pass: dual-branch encoding -> concat -> classification.

        Args:
            batch: Dict with:
                F_raw [B, N, N]: real FC matrix
                node_feat, in_degree, out_degree, path_data, dist, attn_mask:
                    Graphormer SC graph inputs.

        Returns:
            logits: [B, num_classes] classification logits.
        """
        fc = batch['F_raw']  # [B, N, N]
        B = fc.shape[0]

        # SC branch: Graphormer on SC graph
        Z_SC = self.sc_encoder(
            node_feat=batch['node_feat'],
            in_degree=batch['in_degree'],
            out_degree=batch['out_degree'],
            path_data=batch['path_data'],
            dist=batch['dist'],
            attn_mask=batch.get('attn_mask'),
        )  # [B, N, d_model]

        # FC branch: row-wise projection of real FC (t=0, no noise)
        t = torch.zeros(B, device=fc.device, dtype=torch.long)
        t_emb = self.time_embed(t)  # [B, d_model]
        Z_FC = self.fc_encoder(fc, t_emb)  # [B, N, d_model]

        # Mean pool each branch
        z_sc = Z_SC.mean(dim=1)  # [B, d_model]
        z_fc = Z_FC.mean(dim=1)  # [B, d_model]

        # Concatenate and classify
        z = torch.cat([z_sc, z_fc], dim=-1)  # [B, 2 * d_model]
        z = self.pool_dropout(z)

        logits = self.head(z)  # [B, num_classes]
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
    frozen_sc = sum(1 for p in classifier.sc_encoder.parameters() if not p.requires_grad)
    total_sc = sum(1 for p in classifier.sc_encoder.parameters())
    frozen_fc = sum(1 for p in classifier.fc_encoder.parameters() if not p.requires_grad)
    total_fc = sum(1 for p in classifier.fc_encoder.parameters())
    print(f"SC encoder: {frozen_sc}/{total_sc} frozen")
    print(f"FC encoder: {frozen_fc}/{total_fc} frozen")

    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in classifier.parameters())
    print(f"Trainable: {trainable}, Total: {total}")

    # Forward pass
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

    loss = logits.sum()
    loss.backward()
    head_has_grad = all(p.grad is not None for p in classifier.head.parameters())
    print(f"Head gradient: {'OK' if head_has_grad else 'FAILED'}")

    print("\nAll sanity checks passed!")
