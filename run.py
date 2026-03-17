"""
Multi-seed experiment runner for SFDiff.

Runs the full pipeline:
    Phase 1: Train SFDiff diffusion model (once, shared across seeds)
    Phase 2: Train Mode B classifier with multiple seeds, report mean +/- std

Usage:
    python run.py --task PPMI --ppmi_train_path ./data/PPMI/train_data.pkl \
                  --ppmi_test_path ./data/PPMI/test_data.pkl \
                  --seeds 0 1 2 3 4

    # Skip diffusion training if already trained:
    python run.py --task PPMI --pretrained_path ./results/diffusion/.../best_model.pth \
                  --seeds 0 1 2 3 4
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Dict, List

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

from train_diffusion import main as train_diffusion, build_arg_parser as build_diff_parser
from train_classifier import main as train_classifier

FIXED_DATA_SEED = 20010920
TEST_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def aggregate_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across seeds with mean and std.

    Args:
        results: List of metric dicts from classifier training.

    Returns:
        Dict mapping metric name to {mean, std}.
    """
    summary = {}
    for key in TEST_METRIC_KEYS:
        values = np.array([res[key] for res in results if key in res], dtype=np.float64)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[key] = {
            "mean": float(values.mean()),
            "std": std,
        }
    return summary


def build_run_parser() -> argparse.ArgumentParser:
    """Build argument parser extending diffusion and classifier args."""
    base_parser = build_diff_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Multi-seed SFDiff experiment runner",
        parents=[base_parser],
    )

    # Multi-seed arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='List of random seeds for classifier training')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Skip diffusion training, use this pretrained model')

    # Classifier arguments
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--no_freeze_encoder', dest='freeze_encoder', action='store_false')
    parser.add_argument('--cls_epochs', type=int, default=50)
    parser.add_argument('--cls_lr', type=float, default=5e-4)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-3)
    parser.add_argument('--cls_hidden_dim', type=int, default=128)
    parser.add_argument('--cls_dropout', type=float, default=0.2)

    return parser


def main():
    """Run the full SFDiff experiment pipeline."""
    parser = build_run_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    # Fix data seed for consistent splits
    if args.data_seed != FIXED_DATA_SEED:
        print(f"[info] Overriding data_seed from {args.data_seed} to "
              f"fixed {FIXED_DATA_SEED} for consistent splits.")
    args.data_seed = FIXED_DATA_SEED

    if not args.seeds:
        parser.error("Please provide at least one seed via --seeds.")

    # ====== Phase 1: Train diffusion model (once) ======
    if args.pretrained_path is None:
        print("=" * 80)
        print("Phase 1: Training SFDiff diffusion model")
        print("=" * 80)
        diff_result = train_diffusion(args)
        pretrained_path = os.path.join(diff_result['run_dir'], 'best_model.pth')
        print(f"\nDiffusion training complete. Model saved to: {pretrained_path}")
    else:
        pretrained_path = args.pretrained_path
        print(f"Using pretrained model: {pretrained_path}")

    # ====== Phase 2: Train classifier with multiple seeds ======
    test_results: List[Dict] = []

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"Phase 2: Classifier seed {seed} (data seed fixed at {FIXED_DATA_SEED})")
        print("=" * 80)

        # Build classifier args from run args
        cls_args = argparse.Namespace(**vars(args))
        cls_args.random_seed = seed
        cls_args.pretrained_path = pretrained_path

        result = train_classifier(cls_args)
        test_metrics = result['test_final']
        test_results.append(test_metrics)

        print(
            f"  [Seed {seed}] "
            f"Acc={test_metrics['accuracy']:.4f}, "
            f"Prec={test_metrics['precision']:.4f}, "
            f"Rec={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics['auc']:.4f}"
        )

    # ====== Aggregate results ======
    summary = aggregate_metrics(test_results)

    print("\n" + "=" * 80)
    print(f"SFDiff Results ({len(test_results)} seeds, data_seed={FIXED_DATA_SEED}):")
    print("=" * 80)
    for key in TEST_METRIC_KEYS:
        stats = summary.get(key)
        if stats:
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
