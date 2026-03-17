"""
Sampling script for SFDiff.

Loads a trained SFDiff model and generates synthetic FC matrices
for each sample in the dataset, saving results to a .npz file.

Usage:
    python sample.py --model_path ./results/diffusion/.../best_model.pth \
                     --data_path ./data/PPMI/test_data.pkl \
                     --task PPMI --num_samples 5
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add reference codebase to path
_REF_ROOT = os.environ.get(
    'SFDIFF_REF_PATH',
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'Uncovering Latent Communication Patterns in Brain Networks via Adaptive Flow Routing',
        'AFR-Net'
    ))
)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from sfdiff.sfdiff import SFDiffModel
from data.dataset import SFDiffDataset, SFDiffPPMIDataset
from dial.data import load_data, preprocess_labels, balance_dataset, split_dataset


def _move_batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensor values in batch to the target device."""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def sample(args: argparse.Namespace):
    """Generate synthetic FC matrices from a trained SFDiff model.

    Args:
        args: Parsed command-line arguments.
    """
    # ---- Load dataset ----
    if args.task == 'PPMI':
        dataset = SFDiffPPMIDataset(args.data_path, device='cpu')
    else:
        data_dict = load_data(args.data_path)
        processed_dict = preprocess_labels(data_dict, task=args.task)
        balanced_dict = balance_dataset(
            processed_dict, ratio=1.0, random_seed=args.data_seed
        )
        _, test_data = split_dataset(
            balanced_dict, test_size=0.3, random_seed=args.data_seed
        )
        dataset = SFDiffDataset(test_data, device='cpu')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
    )

    # Determine N
    sample_item = dataset[0]
    N = sample_item['S'].shape[0]
    print(f"Number of brain regions (N): {N}")
    print(f"Dataset size: {len(dataset)}")

    # ---- Load model ----
    model = SFDiffModel(
        N=N,
        d_model=args.d_model,
        nhead=args.nhead,
        num_sc_layers=args.num_sc_layers,
        num_self_attn_layers=args.num_self_attn_layers,
        dim_feedforward=args.dim_feedforward,
        T=args.T,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        sigma=args.sigma,
        schedule=args.schedule,
    ).to(args.device)

    state_dict = torch.load(args.model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from: {args.model_path}")

    # ---- Generate samples ----
    all_generated = []
    all_real_fc = []
    all_sc = []
    all_labels = []
    all_names = []

    for batch in tqdm(loader, desc='Generating FC matrices'):
        batch = _move_batch_to_device(batch, args.device)

        # Generate K synthetic FC per sample
        for k in range(args.num_samples):
            generated = model.sample(batch, device=args.device)  # [B, N, N]
            all_generated.append(generated.cpu().numpy())

        all_real_fc.append(batch['F_raw'].cpu().numpy())
        all_sc.append(batch['S'].cpu().numpy())
        if 'labels' in batch:
            all_labels.append(batch['labels'].cpu().numpy().flatten())
        all_names.extend(batch.get('names', []))

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'generated_fc.npz')

    save_dict = {
        'generated': np.concatenate(all_generated, axis=0),
        'real_fc': np.concatenate(all_real_fc, axis=0),
        'sc': np.concatenate(all_sc, axis=0),
        'names': np.array(all_names, dtype=object),
        'num_samples_per_real': args.num_samples,
    }
    if all_labels:
        save_dict['labels'] = np.concatenate(all_labels, axis=0)

    np.savez(output_path, **save_dict)
    print(f"\nSaved {save_dict['generated'].shape[0]} generated FC matrices to: {output_path}")
    print(f"  Generated shape: {save_dict['generated'].shape}")
    print(f"  Real FC shape: {save_dict['real_fc'].shape}")
    print(f"  K samples per real: {args.num_samples}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for sampling."""
    parser = argparse.ArgumentParser(description='SFDiff Sampling / Generation')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained SFDiff model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset (PPMI pickle or ABCD data_dict)')
    parser.add_argument('--task', type=str, default='PPMI',
                        choices=['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD',
                                 'Eat', 'ADHD', 'ODD', 'Cond', 'PTSD',
                                 'ADHD_ODD_Cond', 'PPMI'])
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for generated samples')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of synthetic FC to generate per real sample')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--data_seed', type=int, default=20010920,
                        help='Data seed for consistent splits')

    # Model architecture args (must match training)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_sc_layers', type=int, default=2)
    parser.add_argument('--num_self_attn_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_min', type=float, default=1e-4)
    parser.add_argument('--beta_max', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'cosine'])
    parser.add_argument('--device', type=str, default='cuda')

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    sample(args)
