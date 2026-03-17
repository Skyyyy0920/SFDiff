"""
SC-guided diffusion process for SFDiff.

Forward process: q(x_t | x_0, L_SC) = N(H_t x_0 H_t, sigma^2 I)
  where H_t = exp(-beta_t L_SC) is the graph heat kernel.
  Bilateral multiplication preserves FC matrix symmetry.

Reverse process: x0-prediction — the network directly predicts x_0,
  avoiding numerically unstable inverse heat kernel computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .heat_kernel import compute_heat_kernel_batch


class SCGuidedDiffusion:
    """SC-guided diffusion process for brain FC matrices.

    The forward process replaces standard Gaussian diffusion with bilateral heat
    diffusion on the structural connectivity graph:
        x_t = H_t @ x_0 @ H_t + sigma * epsilon_sym
    where H_t is the graph heat kernel and epsilon_sym is symmetrized Gaussian noise.

    Args:
        T: Number of diffusion timesteps.
        beta_min: Minimum diffusion time parameter.
        beta_max: Maximum diffusion time parameter.
        sigma: Fixed noise standard deviation.
        schedule: 'linear' or 'cosine' schedule for beta_t.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.5,
        sigma: float = 0.1,
        schedule: str = 'linear',
    ):
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sigma = sigma
        self.schedule = schedule

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute beta_t from the noise schedule.

        Args:
            t: [B] or scalar, timestep indices in [1, T].

        Returns:
            beta_t: [B] or scalar, diffusion time parameters.
        """
        t_float = t.float()
        if self.schedule == 'linear':
            return self.beta_min + (t_float / self.T) * (self.beta_max - self.beta_min)
        elif self.schedule == 'cosine':
            s = t_float / self.T
            return self.beta_min + (self.beta_max - self.beta_min) * (
                1.0 - torch.cos(s * math.pi / 2.0)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eigval: torch.Tensor,
        eigvec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: sample x_t given x_0 and SC eigendecomposition.

        x_t = H_t @ x_0 @ H_t + sigma * noise_sym

        Bilateral multiplication (H_t @ x_0 @ H_t) preserves the symmetry of
        the FC matrix, as opposed to single-sided H_t @ x_0 which breaks it.
        The noise is also symmetrized to maintain overall symmetry.

        Args:
            x0: [B, N, N] original FC matrices (symmetric).
            t: [B] timestep indices (1..T).
            eigval: [B, N] precomputed eigenvalues.
            eigvec: [B, N, N] precomputed eigenvectors.

        Returns:
            x_t: [B, N, N] noisy FC matrix (symmetric).
            noise: [B, N, N] the sampled symmetric Gaussian noise.
        """
        beta_t = self.get_beta(t)  # [B]
        H_t = compute_heat_kernel_batch(eigval, eigvec, beta_t)  # [B, N, N]

        # Bilateral heat diffusion preserves FC symmetry
        mean = H_t @ x0 @ H_t  # [B, N, N]

        # Symmetrize noise so x_t stays symmetric
        noise = torch.randn_like(x0)  # [B, N, N]
        noise = (noise + noise.transpose(-2, -1)) / 2.0

        x_t = mean + self.sigma * noise

        return x_t, noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eigval: torch.Tensor,
        eigvec: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """One reverse diffusion step (x0-prediction).

        The network directly predicts x_0, avoiding numerically unstable
        inverse heat kernel computation:
            x_0_hat = model(x_t, t, batch)
            x_{t-1} = H_{t-1} @ x_0_hat @ H_{t-1} + sigma * z_sym  (for t > 1)
            x_{t-1} = x_0_hat                                         (for t = 1)

        Args:
            model: Denoiser network (returns x0 prediction).
            x_t: [B, N, N] current noisy FC.
            t: [B] current timestep.
            eigval: [B, N] eigenvalues.
            eigvec: [B, N, N] eigenvectors.
            batch: Dict with Graphormer inputs for SC encoder.

        Returns:
            x_prev: [B, N, N] denoised FC at t-1.
        """
        # Network directly predicts x0
        x0_hat = model(x_t, t, batch)  # [B, N, N]
        x0_hat = (x0_hat + x0_hat.transpose(-2, -1)) / 2.0  # symmetrize
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        t_prev = t - 1
        is_last = (t_prev == 0).float().view(-1, 1, 1)  # [B, 1, 1]

        beta_prev = self.get_beta(t_prev.clamp(min=1))
        H_prev = compute_heat_kernel_batch(eigval, eigvec, beta_prev)  # [B, N, N]

        # Bilateral heat diffusion mean at t-1
        mean_prev = H_prev @ x0_hat @ H_prev  # [B, N, N]

        # Symmetric noise (not added at t=1)
        z = torch.randn_like(x_t)
        z = (z + z.transpose(-2, -1)) / 2.0
        x_prev = mean_prev + self.sigma * z

        # At t=1 (t_prev=0), return x_0_hat directly without noise
        x_prev = is_last * x0_hat + (1.0 - is_last) * x_prev

        return x_prev

    @torch.no_grad()
    def sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int],
        eigval: torch.Tensor,
        eigvec: torch.Tensor,
        batch: dict,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Full reverse sampling loop from t=T to t=1.

        Args:
            model: Denoiser network.
            shape: (B, N, N) output shape.
            eigval: [B, N] eigenvalues.
            eigvec: [B, N, N] eigenvectors.
            batch: Dict with Graphormer inputs.
            device: Target device.

        Returns:
            x_0: [B, N, N] generated FC matrices (symmetric, in [-1, 1]).
        """
        B, N, _ = shape

        # Start from symmetric pure noise at t=T
        x_t = torch.randn(B, N, N, device=device)
        x_t = self.sigma * (x_t + x_t.transpose(-2, -1)) / 2.0

        for step in reversed(range(1, self.T + 1)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, eigval, eigvec, batch)

        # Final symmetrization and clamping
        x_0 = (x_t + x_t.transpose(-2, -1)) / 2.0
        x_0 = x_0.clamp(-1.0, 1.0)

        # Zero diagonal
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)  # [1, N, N]
        x_0 = x_0 * mask

        return x_0


if __name__ == '__main__':
    print("=== diffusion.py sanity check ===")

    B, N = 2, 10
    T = 50  # small T for testing

    # Create synthetic eigendecomposition
    from sfdiff.heat_kernel import precompute_eigen
    sc = torch.rand(N, N)
    sc = (sc + sc.T) / 2
    sc.fill_diagonal_(0.0)
    sc = sc.clamp(min=0.0)

    eigval, eigvec = precompute_eigen(sc)
    eigval_b = eigval.unsqueeze(0).expand(B, -1).clone()
    eigvec_b = eigvec.unsqueeze(0).expand(B, -1, -1).clone()

    # Create symmetric synthetic FC
    x0 = torch.randn(B, N, N) * 0.3
    x0 = (x0 + x0.transpose(-2, -1)) / 2.0
    x0 = x0.clamp(-1.0, 1.0)

    diffusion = SCGuidedDiffusion(T=T, beta_min=1e-4, beta_max=0.5, sigma=0.1)

    # --- Bug 1 verification: q_sample preserves symmetry ---
    print("\n[Bug 1] q_sample symmetry verification:")
    t_mid = torch.full((B,), 25, dtype=torch.long)
    x_t, noise = diffusion.q_sample(x0, t_mid, eigval_b, eigvec_b)

    x_t_sym_err = (x_t - x_t.transpose(-2, -1)).abs().max().item()
    noise_sym_err = (noise - noise.transpose(-2, -1)).abs().max().item()
    print(f"  x_t symmetry error: {x_t_sym_err:.2e} (should be < 1e-5)")
    assert x_t_sym_err < 1e-5, f"x_t symmetry error too large: {x_t_sym_err}"
    print(f"  noise symmetry error: {noise_sym_err:.2e} (should be < 1e-5)")
    assert noise_sym_err < 1e-5, f"noise symmetry error too large: {noise_sym_err}"

    # Test at t=1 (near identity heat kernel)
    t_low = torch.ones(B, dtype=torch.long)
    x_t_low, _ = diffusion.q_sample(x0, t_low, eigval_b, eigvec_b)
    diff_low = (x_t_low - x0).abs().mean().item()
    print(f"  q_sample at t=1: mean diff from x0 = {diff_low:.4f} (should be ~sigma={diffusion.sigma})")

    # Shape checks
    assert x_t.shape == (B, N, N), f"Expected {(B, N, N)}, got {x_t.shape}"
    assert noise.shape == (B, N, N)
    print(f"  q_sample output shape: {x_t.shape}")

    # --- Bug 2 verification: p_sample with mock x0-prediction model ---
    print("\n[Bug 2] p_sample (x0-prediction) verification:")

    class MockX0Model(nn.Module):
        """Mock model that returns clamped x_t as x0 prediction."""
        def forward(self, x_t, t, batch):
            return x_t.clamp(-1.0, 1.0)

    mock_model = MockX0Model()
    t_test = torch.full((B,), 10, dtype=torch.long)
    x_t_test, _ = diffusion.q_sample(x0, t_test, eigval_b, eigvec_b)

    x_prev = diffusion.p_sample(mock_model, x_t_test, t_test, eigval_b, eigvec_b, batch={})
    print(f"  p_sample output shape: {x_prev.shape} (expected {(B, N, N)})")
    assert x_prev.shape == (B, N, N)

    p_sym_err = (x_prev - x_prev.transpose(-2, -1)).abs().max().item()
    print(f"  p_sample symmetry error: {p_sym_err:.2e} (should be < 1e-5)")
    assert p_sym_err < 1e-5, f"p_sample symmetry error too large: {p_sym_err}"

    # Test at t=1 (should return x0_hat without noise)
    t_one = torch.ones(B, dtype=torch.long)
    x_prev_last = diffusion.p_sample(mock_model, x_t_test, t_one, eigval_b, eigvec_b, batch={})
    p_last_sym_err = (x_prev_last - x_prev_last.transpose(-2, -1)).abs().max().item()
    print(f"  p_sample at t=1 symmetry error: {p_last_sym_err:.2e}")
    assert p_last_sym_err < 1e-5

    print("\nAll sanity checks passed!")
