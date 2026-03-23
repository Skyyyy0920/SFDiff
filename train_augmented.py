"""
Mode A: Data augmentation with SFDiff + DIAL classifier.

Pipeline:
    1. Load training data, generate K synthetic FC per sample using pretrained SFDiff
    2. Build augmented training dataset (original + synthetic)
    3. Train DIAL classifier on augmented data
    4. Evaluate on original test set

Usage:
    python train_augmented.py \
        --pretrained_path ./results/diffusion/.../best_model.pth \
        --task OCD --data_path ./data/data_dict.pkl \
        --num_aug 5 --device cuda
"""

import os
import sys
import copy
import random
import logging
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_fn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

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
from dial.model import DIALModel
from dial.data import (
    load_data, preprocess_labels, balance_dataset, split_dataset,
    normalize_sc, filter_fc, build_graph, BrainDatasetBase,
)
from dgl import shortest_dist

LOGGER_NAME = "sfdiff_augmented"


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


def _compute_metrics(labels, preds, probs) -> Dict[str, float]:
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


# ============================================================
# Generation: produce synthetic FC for training samples
# ============================================================

def generate_augmented_samples(
    sfdiff_model: SFDiffModel,
    train_dataset: SFDiffDataset,
    num_aug: int,
    device: str,
    batch_size: int = 8,
) -> List[Dict]:
    """Generate synthetic FC matrices for each training sample.

    Args:
        sfdiff_model: Pretrained SFDiff model.
        train_dataset: Training dataset with SC, FC, labels.
        num_aug: Number of synthetic FC to generate per sample.
        device: Device for generation.
        batch_size: Batch size for generation.

    Returns:
        List of augmented sample dicts (original format: SC, FC, label, name).
    """
    sfdiff_model.eval()

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=train_dataset.collate,
    )

    augmented_samples = []
    sample_idx = 0

    for batch in tqdm(loader, desc='Generating augmented FC'):
        batch = _move_batch_to_device(batch, device)
        B = batch['S'].shape[0]

        for k in range(num_aug):
            with torch.no_grad():
                fc_syn = sfdiff_model.sample(batch, device=device)  # [B, N, N]

            # Convert each sample back to raw format
            for i in range(B):
                sc_np = batch['S'][i].cpu().numpy()
                fc_np = fc_syn[i].cpu().numpy()
                fc_np = np.clip(fc_np, -1.0, 1.0)

                # Get original label and name
                orig_idx = sample_idx + i
                orig_sample = train_dataset.samples[orig_idx]
                label = orig_sample['label'].item() if isinstance(orig_sample['label'], torch.Tensor) else orig_sample['label']
                name = orig_sample['name']

                augmented_samples.append({
                    'SC': sc_np,
                    'FC': fc_np,
                    'label': label,
                    'name': f'{name}_aug{k}',
                })

        sample_idx += B

    return augmented_samples


# ============================================================
# DIAL-compatible dataset from raw sample dicts
# ============================================================

class AugmentedDataset(BrainDatasetBase):
    """Dataset for augmented samples, compatible with DIAL model input.

    Builds DGL graphs with filtered FC as node features (matching DIAL's
    expected input format).
    """

    def __init__(
        self,
        data_list: List[Dict],
        device: str = 'cpu',
        max_degree: int = 511,
        max_path_len: int = 5,
        is_ppmi: bool = False,
    ):
        samples = [
            self._prepare_sample(item, max_path_len, is_ppmi)
            for item in tqdm(data_list, desc='Building augmented dataset')
        ]
        super().__init__(
            samples=samples,
            device=device,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )

    @staticmethod
    def _prepare_sample(item: Dict, max_path_len: int, is_ppmi: bool) -> Dict:
        """Prepare a single sample for DIAL-compatible format."""
        if is_ppmi:
            S = torch.from_numpy(np.asarray(item['SC'], dtype=np.float32))
            F_mat = torch.from_numpy(np.asarray(item['FC'], dtype=np.float32))
        else:
            S = normalize_sc(item['SC'])
            F_mat = filter_fc(item['FC'])

        label = torch.tensor(item['label'], dtype=torch.long)
        name = item['name']

        graph = build_graph(S, F_mat)
        spd, path = shortest_dist(graph, root=None, return_paths=True)
        graph.ndata['spd'] = spd
        graph.ndata['path'] = path

        return {
            'S': S,
            'F': F_mat,
            'label': label,
            'name': name,
            'graph': graph,
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            'S': sample['S'].to(self.device),
            'F': sample['F'].to(self.device),
            'label': sample['label'].to(self.device),
            'name': sample['name'],
            'graph': sample['graph'],
        }

    def collate(self, samples: List[Dict]) -> Dict:
        batch = super().collate(samples)
        return batch


# ============================================================
# DIAL training / evaluation (mirrors reference main.py)
# ============================================================

def train_dial_epoch(
    model: DIALModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
) -> Dict:
    """Train one epoch of DIAL classifier."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    count = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch + 1}/{num_epochs}", leave=False)
    for batch in pbar:
        labels = batch['labels'].to(device).squeeze(-1).long()
        B = labels.shape[0]

        optimizer.zero_grad()
        y_pred, loss = model(
            node_feat=batch['node_feat'].to(device),
            in_degree=batch['in_degree'].to(device),
            out_degree=batch['out_degree'].to(device),
            path_data=batch['path_data'].to(device),
            dist=batch['dist'].to(device),
            attn_mask=batch['attn_mask'].to(device),
            S=batch['S'].to(device),
            F=batch['F'].to(device),
            y=labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * B
        count += B

        probs = torch.softmax(y_pred.detach(), dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / max(count, 1)
    return metrics


@torch.no_grad()
def evaluate_dial(
    model: DIALModel,
    dataloader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate DIAL classifier."""
    model.eval()
    all_preds, all_labels, all_probs, all_names = [], [], [], []

    for batch in dataloader:
        labels = batch['labels'].to(device).squeeze(-1).long()
        y_pred, _, _ = model.inference(
            node_feat=batch['node_feat'].to(device),
            in_degree=batch['in_degree'].to(device),
            out_degree=batch['out_degree'].to(device),
            path_data=batch['path_data'].to(device),
            dist=batch['dist'].to(device),
            attn_mask=batch['attn_mask'].to(device),
            S=batch['S'].to(device),
            F=batch['F'].to(device),
        )
        probs = torch.softmax(y_pred, dim=1)
        all_preds.extend(y_pred.argmax(dim=1).cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_names.extend(batch.get('names', []))

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    return metrics


# ============================================================
# Main pipeline
# ============================================================

def main(args: argparse.Namespace) -> Dict:
    """Mode A augmentation pipeline.

    Returns:
        Dict with test metrics.
    """
    set_seed(args.random_seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, 'augmented', run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, 'train.log'))
    logger.info("SFDiff Mode A: Augmentation + DIAL Classifier")
    logger.info(f"Args: {vars(args)}")

    # ---- Load data ----
    if args.task == 'PPMI':
        train_sfdiff = SFDiffPPMIDataset(args.ppmi_train_path, device='cpu')
        test_sfdiff = SFDiffPPMIDataset(args.ppmi_test_path, device='cpu')
        is_ppmi = True

        # Original training samples as raw dicts
        orig_train_list = SFDiffPPMIDataset._load_ppmi_split(args.ppmi_train_path)
    else:
        data_dict = load_data(args.data_path)
        processed_dict = preprocess_labels(data_dict, task=args.task)
        balanced_dict = balance_dataset(
            processed_dict, ratio=args.balance_ratio, random_seed=args.data_seed
        )
        train_data_all, test_data = split_dataset(
            balanced_dict, test_size=args.test_size, random_seed=args.data_seed
        )
        train_sfdiff = SFDiffDataset(train_data_all, device='cpu')
        is_ppmi = False
        orig_train_list = train_data_all

    N = train_sfdiff[0]['S'].shape[0]
    logger.info(f"N={N}, train={len(train_sfdiff)}, num_aug={args.num_aug}")

    # ---- Load pretrained SFDiff ----
    sfdiff_model = SFDiffModel(
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

    state_dict = torch.load(args.pretrained_path, map_location=args.device, weights_only=True)
    sfdiff_model.load_state_dict(state_dict)
    logger.info(f"Loaded SFDiff from: {args.pretrained_path}")

    # ---- Generate augmented samples ----
    aug_samples = generate_augmented_samples(
        sfdiff_model=sfdiff_model,
        train_dataset=train_sfdiff,
        num_aug=args.num_aug,
        device=args.device,
        batch_size=args.gen_batch_size,
    )
    logger.info(f"Generated {len(aug_samples)} synthetic samples")

    # Combine original + augmented
    combined_list = list(orig_train_list) + aug_samples
    random.shuffle(combined_list)
    logger.info(f"Combined training set: {len(orig_train_list)} orig + {len(aug_samples)} aug = {len(combined_list)}")

    # ---- Split combined into train/val ----
    combined_labels = [item['label'] if isinstance(item['label'], int)
                       else int(item['label']) for item in combined_list]
    train_list, val_list = train_test_split(
        combined_list,
        test_size=args.val_size,
        random_state=args.data_seed,
        stratify=combined_labels,
    )

    # Build DIAL-compatible datasets
    train_dataset = AugmentedDataset(train_list, device='cpu', is_ppmi=is_ppmi)
    val_dataset = AugmentedDataset(val_list, device='cpu', is_ppmi=is_ppmi)

    if args.task == 'PPMI':
        test_list = SFDiffPPMIDataset._load_ppmi_split(args.ppmi_test_path)
        test_dataset = AugmentedDataset(test_list, device='cpu', is_ppmi=is_ppmi)
    else:
        test_dataset = AugmentedDataset(test_data, device='cpu', is_ppmi=is_ppmi)

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

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ---- Build DIAL model ----
    dial_model = DIALModel(
        N=N,
        d_model=args.dial_d_model,
        nhead=args.dial_nhead,
        num_node_layers=args.dial_num_node_layers,
        num_graph_layers=args.dial_num_graph_layers,
        dim_feedforward=args.dial_dim_feedforward,
        num_classes=2,
        task='classification',
        dropout=args.dial_dropout,
    ).to(args.device)

    total_params = sum(p.numel() for p in dial_model.parameters())
    logger.info(f"DIAL model params: {total_params:,}")

    optimizer = optim.Adam(
        dial_model.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cls_epochs, eta_min=1e-6
    )

    # ---- Training loop ----
    best_val_auc = 0.0
    best_test_metrics = {}

    for epoch in range(args.cls_epochs):
        train_metrics = train_dial_epoch(
            dial_model, train_loader, optimizer, args.device, epoch, args.cls_epochs
        )
        val_metrics = evaluate_dial(dial_model, val_loader, args.device)
        test_metrics = evaluate_dial(dial_model, test_loader, args.device)
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
                if k in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix')
            }
            torch.save(
                dial_model.state_dict(),
                os.path.join(run_dir, 'best_dial.pth'),
            )
            logger.info(f"  *** Best val AUC={best_val_auc:.4f} ***")

    # ---- Final evaluation ----
    dial_model.load_state_dict(
        torch.load(os.path.join(run_dir, 'best_dial.pth'), map_location=args.device, weights_only=True)
    )
    final_test = evaluate_dial(dial_model, test_loader, args.device)

    logger.info("=" * 60)
    logger.info("Final Test Results (Mode A augmentation + DIAL):")
    for key in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        logger.info(f"  {key}: {final_test[key]:.4f}")
    logger.info(f"  Confusion matrix: {final_test['confusion_matrix']}")

    results = {
        'test_final': {
            k: v for k, v in final_test.items()
            if k in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix')
        },
        'best_val_auc': best_val_auc,
        'num_orig': len(orig_train_list),
        'num_aug': len(aug_samples),
        'args': vars(args),
    }
    with open(os.path.join(run_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to: {run_dir}")
    return results


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build argument parser for Mode A augmented training."""
    parser = argparse.ArgumentParser(
        description='SFDiff Mode A: Augmentation + DIAL Classifier',
        add_help=add_help,
    )

    # Data
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

    # SFDiff pretrained model
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_sc_layers', type=int, default=2)
    parser.add_argument('--num_self_attn_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--beta_min', type=float, default=1e-4)
    parser.add_argument('--beta_max', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--schedule', type=str, default='linear')

    # Augmentation
    parser.add_argument('--num_aug', type=int, default=5,
                        help='Number of synthetic FC per training sample')
    parser.add_argument('--gen_batch_size', type=int, default=8,
                        help='Batch size for SFDiff generation')

    # DIAL classifier
    parser.add_argument('--dial_d_model', type=int, default=64)
    parser.add_argument('--dial_nhead', type=int, default=4)
    parser.add_argument('--dial_num_node_layers', type=int, default=2)
    parser.add_argument('--dial_num_graph_layers', type=int, default=2)
    parser.add_argument('--dial_dim_feedforward', type=int, default=128)
    parser.add_argument('--dial_dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--cls_epochs', type=int, default=100)
    parser.add_argument('--cls_lr', type=float, default=1e-4)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    main(args)
