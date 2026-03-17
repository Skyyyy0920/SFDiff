"""
Training script for SFDiff generative diffusion model.

This is unsupervised generative training: given SC (structural connectivity),
learn to generate anatomically plausible FC (functional connectivity) matrices.
Labels are NOT used during this phase.

Usage:
    python train_diffusion.py --task PPMI --ppmi_train_path ./data/PPMI/train_data.pkl
    python train_diffusion.py --task OCD --data_path /data/data_dict.pkl
"""

import os
import sys
import random
import logging
import argparse
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from typing import Dict

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

LOGGER_NAME = "sfdiff_diffusion"


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: str) -> logging.Logger:
    """Configure file and console logging."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def _move_batch_to_device(batch: Dict, device: str) -> Dict:
    """Move all tensor values in batch to the target device."""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def train_epoch(
    model: SFDiffModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
) -> float:
    """Train one epoch of diffusion model.

    Args:
        model: SFDiffModel instance.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        device: Device string.
        epoch: Current epoch index (0-based).
        num_epochs: Total number of epochs.

    Returns:
        avg_loss: Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    count = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch + 1}/{num_epochs}", leave=False)
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)

        optimizer.zero_grad()
        loss = model(batch, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = batch['F_raw'].shape[0]
        total_loss += loss.item() * B
        count += B
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(count, 1)


@torch.no_grad()
def validate_epoch(
    model: SFDiffModel,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Compute validation loss.

    Args:
        model: SFDiffModel instance.
        dataloader: Validation DataLoader.
        device: Device string.

    Returns:
        avg_loss: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        loss = model(batch, device=device)
        B = batch['F_raw'].shape[0]
        total_loss += loss.item() * B
        count += B

    return total_loss / max(count, 1)


def main(args: argparse.Namespace) -> Dict:
    """Main training function for SFDiff diffusion model.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict with 'run_dir' and 'best_loss'.
    """
    set_seed(args.random_seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, 'diffusion', run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, 'train.log'))
    logger.info("SFDiff Diffusion Training")
    logger.info(f"Args: {vars(args)}")

    # ---- Load data ----
    if args.task == 'PPMI':
        full_dataset = SFDiffPPMIDataset(args.ppmi_train_path, device='cpu')
    else:
        data_dict = load_data(args.data_path)
        processed_dict = preprocess_labels(data_dict, task=args.task)
        balanced_dict = balance_dataset(
            processed_dict, ratio=args.balance_ratio, random_seed=args.data_seed
        )
        train_data, _ = split_dataset(
            balanced_dict, test_size=args.test_size, random_seed=args.data_seed
        )
        full_dataset = SFDiffDataset(train_data, device='cpu')

    # Split into train/val for monitoring
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_size))
    n_train = n_total - n_val

    indices = list(range(n_total))
    rng = np.random.default_rng(args.data_seed)
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=full_dataset.collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=full_dataset.collate,
    )

    # Determine N from first sample
    sample = full_dataset[0]
    N = sample['S'].shape[0]
    logger.info(f"Number of brain regions (N): {N}")
    logger.info(f"Train samples: {n_train}, Val samples: {n_val}")

    # ---- Build model ----
    model = SFDiffModel(
        N=N,
        d_model=args.d_model,
        nhead=args.nhead,
        num_sc_layers=args.num_sc_layers,
        num_self_attn_layers=args.num_self_attn_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        T=args.T,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        sigma=args.sigma,
        schedule=args.schedule,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # ---- Training loop ----
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, args.device, epoch, args.num_epochs
        )
        val_loss = validate_epoch(model, val_loader, args.device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr:.2e}"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
            }, ckpt_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            logger.info(f"  *** Saved best model (val_loss={best_val_loss:.6f}) ***")

    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))

    # Save training config
    with open(os.path.join(run_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    logger.info(f"Training complete! Best val loss: {best_val_loss:.6f}")
    logger.info(f"Models saved to: {run_dir}")

    return {'run_dir': run_dir, 'best_loss': best_val_loss}


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build argument parser for diffusion training."""
    parser = argparse.ArgumentParser(
        description='SFDiff Diffusion Training',
        add_help=add_help,
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/data_dict.pkl',
                        help='Path to ABCD data pickle')
    parser.add_argument('--task', type=str, default='PPMI',
                        choices=['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD',
                                 'Eat', 'ADHD', 'ODD', 'Cond', 'PTSD',
                                 'ADHD_ODD_Cond', 'PPMI'],
                        help='Task/dataset name')
    parser.add_argument('--ppmi_train_path', type=str, default='./data/PPMI/train_data.pkl',
                        help='Path to PPMI train pickle')
    parser.add_argument('--ppmi_test_path', type=str, default='./data/PPMI/test_data.pkl',
                        help='Path to PPMI test pickle')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set fraction (for ABCD)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set fraction (from training set)')
    parser.add_argument('--balance_ratio', type=float, default=1.0,
                        help='Positive/negative ratio after balancing')
    parser.add_argument('--data_seed', type=int, default=20010920,
                        help='Random seed for data splitting')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for model initialization')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_sc_layers', type=int, default=2,
                        help='Number of Graphormer layers in SC encoder')
    parser.add_argument('--num_self_attn_layers', type=int, default=2,
                        help='Number of self-attention layers after cross-attention')
    parser.add_argument('--dim_feedforward', type=int, default=128,
                        help='Feedforward dimension in transformers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Diffusion arguments
    parser.add_argument('--T', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_min', type=float, default=1e-4,
                        help='Minimum diffusion time parameter')
    parser.add_argument('--beta_max', type=float, default=0.5,
                        help='Maximum diffusion time parameter')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Fixed noise standard deviation')
    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Noise schedule type')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    main(args)
