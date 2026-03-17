"""
Training script for Mode B latent-space classifier.

Loads a pretrained SFDiff model, freezes the SC encoder, and trains
an MLP classification head for brain disease classification.

Usage:
    python train_classifier.py \
        --pretrained_path ./results/diffusion/.../best_model.pth \
        --task PPMI --ppmi_train_path ./data/PPMI/train_data.pkl \
        --ppmi_test_path ./data/PPMI/test_data.pkl
"""

import os
import sys
import random
import logging
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
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

from sfdiff.sfdiff import SFDiffModel
from sfdiff.classifier import LatentClassifier
from data.dataset import SFDiffDataset, SFDiffPPMIDataset
from dial.data import load_data, preprocess_labels, balance_dataset, split_dataset

LOGGER_NAME = "sfdiff_classifier"


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


def _compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[float],
) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = 0.0
    return metrics


def train_epoch(
    model: LatentClassifier,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
) -> Dict:
    """Train one classification epoch.

    Returns:
        Dict with loss, accuracy, precision, recall, f1, auc.
    """
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    count = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch + 1}/{num_epochs}", leave=False)
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        labels = batch['labels'].squeeze(-1).long()

        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = labels.shape[0]
        total_loss += loss.item() * B
        count += B

        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / max(count, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: LatentClassifier,
    dataloader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate classifier on a dataloader.

    Returns:
        Dict with accuracy, precision, recall, f1, auc,
        predictions, probabilities, labels, names, confusion_matrix.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_names: List[str] = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        labels = batch['labels'].squeeze(-1).long()

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_names.extend(batch.get('names', []))

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    metrics['predictions'] = all_preds
    metrics['probabilities'] = all_probs
    metrics['labels'] = all_labels
    metrics['names'] = all_names
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    return metrics


def main(args: argparse.Namespace) -> Dict:
    """Main classifier training function.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict with 'run_dir', 'best_auc', 'test_final'.
    """
    set_seed(args.random_seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, 'classifier', run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, 'train.log'))
    logger.info("SFDiff Mode B Classifier Training")
    logger.info(f"Args: {vars(args)}")

    # ---- Load data ----
    if args.task == 'PPMI':
        train_dataset = SFDiffPPMIDataset(args.ppmi_train_path, device='cpu')
        test_dataset = SFDiffPPMIDataset(args.ppmi_test_path, device='cpu')

        # Split train into train/val
        n_total = len(train_dataset)
        labels = [train_dataset[i]['label'].item() for i in range(n_total)]
        train_idx, val_idx = train_test_split(
            list(range(n_total)),
            test_size=args.val_size,
            random_state=args.data_seed,
            stratify=labels,
        )
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_dataset.collate,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            collate_fn=train_dataset.collate,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=test_dataset.collate,
        )
    else:
        data_dict = load_data(args.data_path)
        processed_dict = preprocess_labels(data_dict, task=args.task)
        balanced_dict = balance_dataset(
            processed_dict, ratio=args.balance_ratio, random_seed=args.data_seed
        )
        train_data, test_data = split_dataset(
            balanced_dict, test_size=args.test_size, random_seed=args.data_seed
        )

        # Split train into train/val
        train_labels = [item['label'] for item in train_data]
        train_data_final, val_data = train_test_split(
            train_data,
            test_size=args.val_size,
            random_state=args.data_seed,
            stratify=train_labels,
        )

        train_dataset = SFDiffDataset(train_data_final, device='cpu')
        val_dataset = SFDiffDataset(val_data, device='cpu')
        test_dataset = SFDiffDataset(test_data, device='cpu')

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_dataset.collate,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=val_dataset.collate,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=test_dataset.collate,
        )

    # Determine N
    if args.task == 'PPMI':
        sample_item = train_dataset[0]
    else:
        sample_item = train_dataset[0]
    N = sample_item['S'].shape[0]
    logger.info(f"Number of brain regions (N): {N}")

    # ---- Load pretrained SFDiff ----
    sfdiff_model = SFDiffModel(
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
    ).to(args.device)

    state_dict = torch.load(args.pretrained_path, map_location=args.device, weights_only=True)
    sfdiff_model.load_state_dict(state_dict)
    logger.info(f"Loaded pretrained SFDiff from: {args.pretrained_path}")

    # ---- Build classifier ----
    classifier = LatentClassifier(
        sfdiff_model=sfdiff_model,
        d_model=args.d_model,
        num_classes=2,
        hidden_dim=args.cls_hidden_dim,
        dropout=args.cls_dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(args.device)

    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in classifier.parameters())
    logger.info(f"Classifier - trainable: {trainable_params:,}, total: {total_params:,}")

    # Only optimize unfrozen parameters
    params_to_optimize = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params_to_optimize, lr=args.cls_lr, weight_decay=args.cls_weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cls_epochs, eta_min=1e-6
    )

    # ---- Training loop ----
    best_val_auc = 0.0
    best_test_metrics = {}

    for epoch in range(args.cls_epochs):
        train_metrics = train_epoch(
            classifier, train_loader, optimizer, args.device, epoch, args.cls_epochs
        )
        val_metrics = evaluate(classifier, val_loader, args.device)
        test_metrics = evaluate(classifier, test_loader, args.device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{args.cls_epochs} - "
            f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, auc={train_metrics['auc']:.4f} | "
            f"Val: acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1']:.4f}, "
            f"auc={val_metrics['auc']:.4f} | "
            f"Test: acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
            f"auc={test_metrics['auc']:.4f}"
        )

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_test_metrics = {
                k: v for k, v in test_metrics.items()
                if k in ('accuracy', 'precision', 'recall', 'f1', 'auc')
            }
            torch.save(
                classifier.state_dict(),
                os.path.join(run_dir, 'best_classifier.pth'),
            )
            logger.info(f"  *** Best val AUC={best_val_auc:.4f} ***")

    # ---- Final evaluation ----
    # Load best model
    classifier.load_state_dict(
        torch.load(os.path.join(run_dir, 'best_classifier.pth'), map_location=args.device, weights_only=True)
    )
    final_test = evaluate(classifier, test_loader, args.device)

    logger.info("=" * 60)
    logger.info("Final Test Results (best val AUC checkpoint):")
    for key in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        logger.info(f"  {key}: {final_test[key]:.4f}")
    logger.info(f"  Confusion matrix: {final_test['confusion_matrix']}")

    # Save results
    results = {
        'test_final': {
            k: v for k, v in final_test.items()
            if k in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix')
        },
        'best_val_auc': best_val_auc,
        'args': vars(args),
    }
    with open(os.path.join(run_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to: {run_dir}")

    return {
        'run_dir': run_dir,
        'best_auc': best_val_auc,
        'test_final': results['test_final'],
    }


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build argument parser for classifier training."""
    parser = argparse.ArgumentParser(
        description='SFDiff Mode B Classifier Training',
        add_help=add_help,
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/data_dict.pkl')
    parser.add_argument('--task', type=str, default='PPMI',
                        choices=['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD',
                                 'Eat', 'ADHD', 'ODD', 'Cond', 'PTSD',
                                 'ADHD_ODD_Cond', 'PPMI'])
    parser.add_argument('--ppmi_train_path', type=str, default='./data/PPMI/train_data.pkl')
    parser.add_argument('--ppmi_test_path', type=str, default='./data/PPMI/test_data.pkl')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--balance_ratio', type=float, default=1.0)
    parser.add_argument('--data_seed', type=int, default=20010920)
    parser.add_argument('--random_seed', type=int, default=42)

    # Pretrained model
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained SFDiff model weights')

    # SFDiff model architecture (must match pretrained)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_sc_layers', type=int, default=2)
    parser.add_argument('--num_self_attn_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_min', type=float, default=1e-4)
    parser.add_argument('--beta_max', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.1)

    # Classifier arguments
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze SC encoder weights')
    parser.add_argument('--no_freeze_encoder', dest='freeze_encoder', action='store_false')
    parser.add_argument('--cls_epochs', type=int, default=50,
                        help='Number of classifier training epochs')
    parser.add_argument('--cls_lr', type=float, default=5e-4,
                        help='Classifier learning rate')
    parser.add_argument('--cls_weight_decay', type=float, default=1e-3)
    parser.add_argument('--cls_hidden_dim', type=int, default=128,
                        help='Hidden dimension of classifier MLP head')
    parser.add_argument('--cls_dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    main(args)
