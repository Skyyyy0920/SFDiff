"""
Denoising network for SFDiff.

Architecture:
    SC Encoder: GraphormerNodeEncoder (from reference dial/model.py)
        G_SC -> Z_SC [B, N, d_model]

    FC Encoder: row-wise MLP + sinusoidal time embedding
        x_t -> Z_FC [B, N, d_model]

    Cross-Attention: FC queries, SC keys/values
        Z_cross = softmax(Q K^T / sqrt(d)) V  (with residual)

    Self-Attention: TransformerEncoder (2 layers)
        Z_out = TransformerEncoder(Z_cross)

    Output Head: Linear -> reshape -> [B, N, N] -> symmetrize
"""

import os
import sys
import math
import torch
import torch.nn as nn
from typing import Optional, Dict

# Add reference codebase to path for GraphormerNodeEncoder import
_REF_ROOT = os.environ.get(
    'SFDIFF_REF_PATH',
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing',
        'AFR-Net'
    ))
)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from dial.model import GraphormerNodeEncoder


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding, standard in diffusion models.

    Maps integer timesteps to continuous d_model-dimensional vectors using
    sinusoidal positional encoding, then projects through a 2-layer MLP.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] integer timesteps.

        Returns:
            emb: [B, d_model] time embeddings.
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class FCEncoder(nn.Module):
    """Row-wise FC encoder that projects noisy FC rows and adds time embedding.

    Each row of the N x N FC matrix is treated as a length-N signal and
    projected to d_model dimensions, then the time embedding is added.
    """

    def __init__(self, N: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(N, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, N, N] noisy FC matrix.
            t_emb: [B, d_model] time embedding.

        Returns:
            Z_FC: [B, N, d_model]
        """
        Z = self.proj(x_t)  # [B, N, d_model] — row-wise projection
        Z = Z + t_emb.unsqueeze(1)  # broadcast time embedding to all N tokens
        return Z


class SFDiffDenoiser(nn.Module):
    """Denoising network for SFDiff.

    Predicts the noise eps_hat given noisy FC x_t, timestep t, and SC graph.
    The SC graph is processed by GraphormerNodeEncoder (from reference codebase)
    and the noisy FC is processed by a row-wise MLP with time embedding.
    Cross-attention fuses FC queries with SC keys/values, followed by
    self-attention and a linear output head.

    Args:
        N: Number of brain regions (nodes).
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_sc_layers: Number of Graphormer layers in SC encoder.
        num_self_attn_layers: Number of self-attention layers after cross-attention.
        dim_feedforward: Feedforward dimension in transformers.
        dropout: Dropout rate.
        max_degree: Max node degree for Graphormer degree encoding.
        max_path_len: Max shortest-path length for Graphormer path encoding.
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
        max_degree: int = 511,
        max_path_len: int = 5,
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model

        # SC Encoder (reused from reference codebase)
        self.sc_encoder = GraphormerNodeEncoder(
            N=N,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_sc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # FC Encoder
        self.fc_encoder = FCEncoder(N, d_model, dropout)

        # Cross-Attention: FC queries, SC keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)

        # Self-Attention block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_self_attn_layers,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Output head: project each of N tokens to N values -> [B, N, N]
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, N),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Predict noise eps_hat.

        Args:
            x_t: [B, N, N] noisy FC matrix.
            t: [B] timestep indices.
            batch: Dict with Graphormer SC graph inputs:
                node_feat [B, N, N], in_degree [B, N], out_degree [B, N],
                path_data [B, N, N, max_path_len, 1], dist [B, N, N],
                attn_mask [B, N+1, N+1] (optional).

        Returns:
            eps_hat: [B, N, N] predicted noise (symmetrized).
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, d_model]

        # SC Encoder (Graphormer on SC graph)
        Z_SC = self.sc_encoder(
            node_feat=batch['node_feat'],
            in_degree=batch['in_degree'],
            out_degree=batch['out_degree'],
            path_data=batch['path_data'],
            dist=batch['dist'],
            attn_mask=batch.get('attn_mask'),
        )  # [B, N, d_model]

        # FC Encoder (row-wise projection of noisy FC + time embedding)
        Z_FC = self.fc_encoder(x_t, t_emb)  # [B, N, d_model]

        # Cross-Attention: FC queries, SC keys/values (with pre-norm and residual)
        Z_cross, _ = self.cross_attn(
            query=self.cross_norm_q(Z_FC),
            key=self.cross_norm_kv(Z_SC),
            value=Z_SC,
        )
        Z_cross = Z_FC + Z_cross  # residual connection

        # Self-Attention
        Z_out = self.self_attn(Z_cross)
        Z_out = self.self_attn_norm(Z_out)  # [B, N, d_model]

        # Output head: [B, N, d_model] -> [B, N, N]
        eps_hat = self.output_head(Z_out)  # [B, N, N]

        # Symmetrize predicted noise
        eps_hat = (eps_hat + eps_hat.transpose(-2, -1)) / 2.0

        return eps_hat

    def get_sc_embedding(self, batch: dict) -> torch.Tensor:
        """Extract SC encoder embeddings (for Mode B classifier).

        Args:
            batch: Dict with Graphormer inputs.

        Returns:
            Z_SC: [B, N, d_model] node-level SC embeddings.
        """
        with torch.no_grad():
            Z_SC = self.sc_encoder(
                node_feat=batch['node_feat'],
                in_degree=batch['in_degree'],
                out_degree=batch['out_degree'],
                path_data=batch['path_data'],
                dist=batch['dist'],
                attn_mask=batch.get('attn_mask'),
            )
        return Z_SC

    def get_fused_embedding(self, fc: torch.Tensor, batch: dict) -> torch.Tensor:
        """Extract SC-FC fused embeddings by passing real FC through the denoiser.

        Uses t=1 (near-identity) so the FC encoder sees essentially clean FC.
        The cross-attention and self-attention layers fuse SC and FC information
        into a joint representation, reusing representations learned during
        diffusion training.

        Args:
            fc: [B, N, N] real FC matrix.
            batch: Dict with Graphormer SC graph inputs.

        Returns:
            Z_out: [B, N, d_model] fused SC-FC node embeddings.
        """
        B = fc.shape[0]
        # t=1: minimal diffusion, FC encoder sees near-clean input
        t = torch.ones(B, device=fc.device, dtype=torch.long)
        t_emb = self.time_embed(t)  # [B, d_model]

        # SC Encoder
        Z_SC = self.sc_encoder(
            node_feat=batch['node_feat'],
            in_degree=batch['in_degree'],
            out_degree=batch['out_degree'],
            path_data=batch['path_data'],
            dist=batch['dist'],
            attn_mask=batch.get('attn_mask'),
        )  # [B, N, d_model]

        # FC Encoder
        Z_FC = self.fc_encoder(fc, t_emb)  # [B, N, d_model]

        # Cross-Attention: FC queries, SC keys/values
        Z_cross, _ = self.cross_attn(
            query=self.cross_norm_q(Z_FC),
            key=self.cross_norm_kv(Z_SC),
            value=Z_SC,
        )
        Z_cross = Z_FC + Z_cross  # residual

        # Self-Attention
        Z_out = self.self_attn(Z_cross)
        Z_out = self.self_attn_norm(Z_out)  # [B, N, d_model]

        return Z_out


if __name__ == '__main__':
    # Quick sanity check (requires DGL for GraphormerNodeEncoder)
    print("=== denoiser.py sanity check ===")

    B, N = 2, 10
    d_model = 32
    max_path_len = 5

    model = SFDiffDenoiser(
        N=N, d_model=d_model, nhead=4,
        num_sc_layers=1, num_self_attn_layers=1,
        dim_feedforward=64, dropout=0.1,
    )

    # Synthetic inputs
    x_t = torch.randn(B, N, N)
    t = torch.randint(1, 100, (B,))
    batch = {
        'node_feat': torch.randn(B, N, N),
        'in_degree': torch.randint(0, 10, (B, N)),
        'out_degree': torch.randint(0, 10, (B, N)),
        'path_data': torch.randn(B, N, N, max_path_len, 1),
        'dist': torch.randint(0, 5, (B, N, N)),
        'attn_mask': torch.zeros(B, N + 1, N + 1, dtype=torch.bool),
    }

    # Forward pass
    eps_hat = model(x_t, t, batch)
    print(f"eps_hat shape: {eps_hat.shape}")
    assert eps_hat.shape == (B, N, N), f"Expected {(B, N, N)}, got {eps_hat.shape}"

    # Check symmetry
    sym_err = (eps_hat - eps_hat.transpose(-2, -1)).abs().max().item()
    print(f"Symmetry error: {sym_err:.2e}")
    assert sym_err < 1e-6, "Output should be symmetric"

    # Check gradient flow
    loss = eps_hat.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradient flow: {'OK' if has_grad else 'FAILED'}")
    assert has_grad, "Gradients should flow to all parameters"

    # Check SC embedding extraction
    Z_SC = model.get_sc_embedding(batch)
    print(f"Z_SC shape: {Z_SC.shape}")
    assert Z_SC.shape == (B, N, d_model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    print("\nAll sanity checks passed!")
