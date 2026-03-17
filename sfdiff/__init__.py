"""SFDiff: Structure-guided Functional Diffusion for brain connectivity generation."""

# Pure-torch modules (no DGL dependency) — always available
from .heat_kernel import (
    compute_sc_laplacian,
    precompute_eigen,
    compute_heat_kernel,
    compute_heat_kernel_batch,
    compute_heat_kernel_inv_batch,
)
from .diffusion import SCGuidedDiffusion

# DGL-dependent modules — lazy import to avoid breaking pure-torch usage
def __getattr__(name):
    if name == "SFDiffDenoiser":
        from .denoiser import SFDiffDenoiser
        return SFDiffDenoiser
    if name == "SFDiffModel":
        from .sfdiff import SFDiffModel
        return SFDiffModel
    if name == "LatentClassifier":
        from .classifier import LatentClassifier
        return LatentClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
