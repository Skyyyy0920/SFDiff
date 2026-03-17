"""
Data loading and preprocessing for SFDiff.

Extends the reference BrainDatasetBase with:
1. Eigendecomposition caching (eigvec, eigval) for heat kernel computation
2. Raw FC (unfiltered, clipped to [-1, 1]) for diffusion training
3. Filtered FC for Graphormer SC encoder node features

Reuses utilities from dial/data.py without modification.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Add reference codebase to path for dial imports
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

from dial.data import (
    normalize_sc,
    filter_fc,
    build_graph,
    BrainDatasetBase,
    load_data,
    preprocess_labels,
    balance_dataset,
    split_dataset,
)
from dgl import shortest_dist
from sfdiff.heat_kernel import precompute_eigen


class SFDiffDataset(BrainDatasetBase):
    """Brain connectivity dataset for SFDiff diffusion training (ABCD-style).

    Key differences from ABCDDataset:
    1. FC is NOT filtered for the diffusion target — raw values clipped to [-1, 1].
    2. Precomputes and caches SC Laplacian eigendecomposition per sample.
    3. Filtered FC is still used as node features for the Graphormer SC encoder.
    """

    def __init__(
        self,
        data_list: List[Dict],
        device: str = 'cpu',
        max_degree: int = 511,
        max_path_len: int = 5,
        filter_fc_for_graph: bool = True,
    ):
        """
        Args:
            data_list: List of dicts with keys 'SC', 'FC', 'label', 'name'.
            device: Device for tensor storage.
            max_degree: Max node degree for Graphormer degree encoding.
            max_path_len: Max shortest-path length for Graphormer path encoding.
            filter_fc_for_graph: If True, use filtered FC as Graphormer node features.
                                 If False, use raw FC (clipped to [-1, 1]).
        """
        samples = [
            self._prepare_graph_sample(
                item,
                max_path_len=max_path_len,
                filter_fc_for_graph=filter_fc_for_graph,
            )
            for item in tqdm(data_list, desc='Preparing SFDiff samples')
        ]
        super().__init__(
            samples=samples,
            device=device,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )

    @staticmethod
    def _prepare_graph_sample(
        item: Dict,
        max_path_len: int = 5,
        filter_fc_for_graph: bool = True,
    ) -> Dict:
        """Prepare a single sample with eigendecomposition caching.

        Args:
            item: Dict with 'SC', 'FC', 'label', 'name' (numpy arrays).
            max_path_len: Max shortest-path length for Graphormer.
            filter_fc_for_graph: Whether to filter FC for Graphormer node features.

        Returns:
            Dict with S, F, F_raw, label, name, graph, eigval, eigvec.
        """
        S = normalize_sc(item['SC'])

        # Raw FC for diffusion (clipped to [-1, 1], NOT filtered)
        F_raw = torch.from_numpy(
            np.clip(np.asarray(item['FC'], dtype=np.float32), -1.0, 1.0)
        )

        # FC for Graphormer node features (optionally filtered)
        if filter_fc_for_graph:
            F_graph = filter_fc(item['FC'])
        else:
            F_graph = F_raw.clone()

        label = torch.tensor(item['label'], dtype=torch.long)
        name = item['name']

        # Build DGL graph using SC topology, FC as node features
        graph = build_graph(S, F_graph)
        spd, path = shortest_dist(graph, root=None, return_paths=True)
        graph.ndata['spd'] = spd
        graph.ndata['path'] = path

        # Precompute eigendecomposition of SC Laplacian (CRITICAL for performance)
        eigval, eigvec = precompute_eigen(S)

        return {
            'S': S,
            'F': F_graph,       # filtered FC for Graphormer input
            'F_raw': F_raw,     # raw FC for diffusion forward process
            'label': label,
            'name': name,
            'graph': graph,
            'eigval': eigval,   # [N], eigenvalues of L_SC, clamped >= 0
            'eigvec': eigvec,   # [N, N], eigenvectors of L_SC
        }

    def __getitem__(self, idx: int) -> Dict:
        """Return a single sample with all fields."""
        sample = self.samples[idx]
        return {
            'S': sample['S'].to(self.device),
            'F': sample['F'].to(self.device),
            'F_raw': sample['F_raw'].to(self.device),
            'label': sample['label'].to(self.device),
            'name': sample['name'],
            'graph': sample['graph'],
            'eigval': sample['eigval'].to(self.device),
            'eigvec': sample['eigvec'].to(self.device),
        }

    def collate(self, samples: List[Dict]) -> Dict:
        """Extend base collation with F_raw, eigval, eigvec."""
        batch = super().collate(samples)
        batch['F_raw'] = torch.stack([s['F_raw'] for s in samples])
        batch['eigval'] = torch.stack([s['eigval'] for s in samples])
        batch['eigvec'] = torch.stack([s['eigvec'] for s in samples])
        return batch


class SFDiffPPMIDataset(BrainDatasetBase):
    """PPMI dataset variant for SFDiff with eigendecomposition caching.

    Follows the reference PPMIDataset behavior:
    - Does NOT apply normalize_sc or filter_fc (raw values used directly).
    - Loads from pickle with keys 'scn', 'fcn', 'labels'.
    """

    def __init__(
        self,
        data_path: str,
        device: str = 'cpu',
        max_degree: int = 511,
        max_path_len: int = 5,
        name_prefix: str = 'ppmi',
    ):
        """
        Args:
            data_path: Path to PPMI pickle file.
            device: Device for tensor storage.
            max_degree: Max node degree for Graphormer.
            max_path_len: Max shortest-path length for Graphormer.
            name_prefix: Prefix for sample names.
        """
        data_list = self._load_ppmi_split(data_path, name_prefix=name_prefix)
        samples = [
            self._prepare_graph_sample(item, max_path_len=max_path_len)
            for item in tqdm(data_list, desc='Preparing PPMI samples')
        ]
        super().__init__(
            samples=samples,
            device=device,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )

    @staticmethod
    def _prepare_graph_sample(item: Dict, max_path_len: int = 5) -> Dict:
        """Prepare a single PPMI sample (no normalize/filter, matching reference)."""
        S = torch.from_numpy(np.asarray(item['SC'], dtype=np.float32))
        F_mat = torch.from_numpy(np.asarray(item['FC'], dtype=np.float32))
        F_raw = F_mat.clamp(-1.0, 1.0)

        label = torch.tensor(item['label'], dtype=torch.long)
        name = item['name']

        graph = build_graph(S, F_mat)
        spd, path = shortest_dist(graph, root=None, return_paths=True)
        graph.ndata['spd'] = spd
        graph.ndata['path'] = path

        # Precompute eigendecomposition
        eigval, eigvec = precompute_eigen(S)

        return {
            'S': S,
            'F': F_mat,
            'F_raw': F_raw,
            'label': label,
            'name': name,
            'graph': graph,
            'eigval': eigval,
            'eigvec': eigvec,
        }

    @staticmethod
    def _load_ppmi_split(data_path: str, name_prefix: str = 'ppmi') -> List[Dict]:
        """Load a PPMI pickle split file."""
        print(f"[PPMI] loading from {data_path} ...")
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)

        required_keys = ('scn', 'fcn', 'labels')
        missing = [k for k in required_keys if k not in raw]
        if missing:
            raise KeyError(f"PPMI pickle missing keys: {missing}")

        scn = np.asarray(raw['scn'])
        fcn = np.asarray(raw['fcn'])
        labels = np.asarray(raw['labels'])

        if scn.shape[0] != fcn.shape[0] or scn.shape[0] != labels.shape[0]:
            raise ValueError(
                f"PPMI data mismatch - scn: {scn.shape}, fcn: {fcn.shape}, labels: {labels.shape}"
            )

        samples: List[Dict] = []
        for idx, (sc, fc, label) in enumerate(zip(scn, fcn, labels)):
            samples.append({
                'SC': sc,
                'FC': fc,
                'label': int(label),
                'name': f"{name_prefix}_{idx}",
            })

        print(f"[PPMI] loaded {len(samples)} samples")
        return samples

    def __getitem__(self, idx: int) -> Dict:
        """Return a single sample with all fields."""
        sample = self.samples[idx]
        return {
            'S': sample['S'].to(self.device),
            'F': sample['F'].to(self.device),
            'F_raw': sample['F_raw'].to(self.device),
            'label': sample['label'].to(self.device),
            'name': sample['name'],
            'graph': sample['graph'],
            'eigval': sample['eigval'].to(self.device),
            'eigvec': sample['eigvec'].to(self.device),
        }

    def collate(self, samples: List[Dict]) -> Dict:
        """Extend base collation with F_raw, eigval, eigvec."""
        batch = super().collate(samples)
        batch['F_raw'] = torch.stack([s['F_raw'] for s in samples])
        batch['eigval'] = torch.stack([s['eigval'] for s in samples])
        batch['eigvec'] = torch.stack([s['eigvec'] for s in samples])
        return batch


if __name__ == '__main__':
    # Quick sanity check with synthetic data
    print("=== dataset.py sanity check ===")

    N = 10
    num_samples = 4

    # Create synthetic data
    data_list = []
    for i in range(num_samples):
        sc = np.random.rand(N, N).astype(np.float32)
        sc = (sc + sc.T) / 2
        np.fill_diagonal(sc, 0.0)
        sc = np.maximum(sc, 0.0)

        fc = np.random.randn(N, N).astype(np.float32) * 0.5
        fc = (fc + fc.T) / 2
        np.fill_diagonal(fc, 1.0)
        fc = np.clip(fc, -1.0, 1.0)

        data_list.append({
            'SC': sc,
            'FC': fc,
            'label': i % 2,
            'name': f'test_{i}',
        })

    # Create dataset
    dataset = SFDiffDataset(data_list, device='cpu', filter_fc_for_graph=True)
    print(f"Dataset size: {len(dataset)}")

    # Check single sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"S shape: {sample['S'].shape}")
    print(f"F shape: {sample['F'].shape}")
    print(f"F_raw shape: {sample['F_raw'].shape}")
    print(f"eigval shape: {sample['eigval'].shape}")
    print(f"eigvec shape: {sample['eigvec'].shape}")
    print(f"eigval range: [{sample['eigval'].min():.6f}, {sample['eigval'].max():.6f}]")
    assert (sample['eigval'] >= 0).all(), "Eigenvalues should be non-negative"
    assert sample['F_raw'].min() >= -1.0 and sample['F_raw'].max() <= 1.0, "F_raw should be in [-1, 1]"

    # Verify eigendecomposition reconstructs Laplacian
    from sfdiff.heat_kernel import compute_sc_laplacian
    L = compute_sc_laplacian(sample['S'])
    L_recon = sample['eigvec'] @ torch.diag(sample['eigval']) @ sample['eigvec'].T
    recon_err = (L - L_recon).abs().max().item()
    print(f"Laplacian reconstruction error: {recon_err:.2e}")
    assert recon_err < 1e-4, "Reconstruction error too large"

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate)
    batch = next(iter(loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"F_raw batch shape: {batch['F_raw'].shape}")
    print(f"eigval batch shape: {batch['eigval'].shape}")
    print(f"eigvec batch shape: {batch['eigvec'].shape}")
    print(f"node_feat batch shape: {batch['node_feat'].shape}")
    print(f"in_degree batch shape: {batch['in_degree'].shape}")
    print(f"path_data batch shape: {batch['path_data'].shape}")
    print(f"dist batch shape: {batch['dist'].shape}")
    print(f"attn_mask batch shape: {batch['attn_mask'].shape}")

    assert batch['F_raw'].shape == (2, N, N)
    assert batch['eigval'].shape == (2, N)
    assert batch['eigvec'].shape == (2, N, N)

    print("\nAll sanity checks passed!")
