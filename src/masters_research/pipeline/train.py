import json
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch_geometric_temporal.signal import temporal_signal_split
try:
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
except Exception:
    Explainer = None
    GNNExplainer = None
    ModelConfig = None
from masters_research.pipeline.integrated_gradients import integrated_gradients

from masters_research.data import TemporalDataLoader
from masters_research.pipeline.dataset import create_temporal_dataset
from masters_research.models import TemporalSpatialNodeClassifier, SimpleGCNClassifier
from masters_research.pipeline.temporal_quality import run_temporal_quality_audit, compare_splits
from masters_research.pipeline import training_curves
import seaborn as sns


def _model_type_from_cfg(cfg_run: dict) -> str:
    """Map a run's hyperparameter flags to one of the 4 canonical model type names.

    Mirrors the branch order in `_train_one_fold` so the folder layout matches the
    architecture actually instantiated. Kept in sync with
    pipeline/model_type_report.py::_classify_run.
    """
    model_type = cfg_run.get("model_type")
    if model_type == "simple_gcn" or bool(cfg_run.get("use_simple_gcn", False)):
        return "SimpleGCN"
    if model_type == "temporal_attention" or bool(cfg_run.get("use_temporal_attention", False)):
        return "TemporalAttentionGNN"
    if bool(cfg_run.get("use_attention", False)):
        return "TemporalGAT_GRU"
    return "TemporalGCN_GRU"


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional logit adjustment and ordinal penalty.

    Base loss: standard focal cross-entropy (gamma=0 reduces to CE) with class weights
    and label smoothing — designed for imbalanced / mode-collapsing regimes.

    Optional additions (both off by default; enable via config):

    * `logit_adjustment` (Menon et al. 2020, "Long-tail learning via logit adjustment"):
      When `log_prior` is provided, the model's raw logits are shifted by
      `log_prior` before the softmax used inside the CE. This teaches the network
      to output *balanced* posteriors regardless of the training prior, correcting
      for temporal drift between train/val/test class frequencies. At inference
      the raw (un-adjusted) logits are used, giving a properly calibrated classifier.

    * `ordinal_lambda > 0`: adds an expected-ordinal-distance penalty
      `E_p[|c - y|]` on top of the focal CE. For an ordered target (low < med < high)
      this penalises "predicted 0 when true is 2" more than "predicted 1 when true is 2",
      so the model respects the class ordering during optimisation.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        log_prior: torch.Tensor | None = None,
        ordinal_lambda: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        # register_buffer keeps the tensor on the same device as the module and
        # excludes it from parameters(). Store None too for the "off" case.
        if log_prior is not None:
            self.register_buffer("log_prior", log_prior.detach().clone())
        else:
            self.log_prior = None
        self.ordinal_lambda = ordinal_lambda

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Logit adjustment: shift logits by log P(y=c) so the CE optimises the
        # balanced posterior. Raw `inputs` are still returned upstream for
        # inference / metric computation.
        if self.log_prior is not None:
            adj_inputs = inputs + self.log_prior.to(inputs.device).unsqueeze(0)
        else:
            adj_inputs = inputs

        ce = nn.functional.cross_entropy(
            adj_inputs, targets, weight=self.weight, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        loss = focal.mean()

        if self.ordinal_lambda > 0.0:
            # Expected ordinal distance from the true class under the (adjusted) softmax.
            # For each sample, sum_c |c - y| * P(c | x).  Uses adj_inputs so the penalty
            # is computed on the same distribution the CE is minimising.
            probs = torch.softmax(adj_inputs, dim=-1)
            num_classes = probs.size(-1)
            class_ids = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
            dist = (class_ids.unsqueeze(0) - targets.unsqueeze(1).to(probs.dtype)).abs()
            expected_dist = (probs * dist).sum(dim=-1).mean()
            loss = loss + self.ordinal_lambda * expected_dist

        return loss


def entropy_regularization(logits: torch.Tensor) -> torch.Tensor:
    """Compute negative entropy of predictions to encourage diverse class predictions.
    
    When entropy is low, the model is confidently predicting one class (bad - mode collapse).
    When entropy is high, predictions are spread across classes (good - diverse).
    
    Returns: negative entropy (minimize this to maximize entropy / diversity)
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return -entropy  # Negative because we want to MAXIMIZE entropy


class _ExplainWrapper(nn.Module):
    """Thin wrapper used only for GNNExplainer / captum.
    The GRU models now return (log_probs, hidden) tuples; this wrapper strips the
    hidden state so external explainability tools receive a plain log-probability tensor.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self._m = model

    def forward(self, x, edge_index, edge_weight=None):
        result = self._m(x, edge_index, edge_weight)
        return result[0] if isinstance(result, tuple) else result


def load_config(config_path: Path) -> dict:
    """Load JSON config and return dict."""
    cfg = {}
    try:
        with config_path.open("r") as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
    return cfg


def compute_prediction_entropy(logits: torch.Tensor) -> float:
    """Compute mean entropy of predictions (higher = more diverse, lower = collapse).
    
    Returns entropy in bits (log base 2). For 3 classes:
    - Maximum entropy: log2(3) ≈ 1.58 (uniform predictions)
    - Minimum entropy: 0 (all predictions same class)
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return float(entropy)


# ==================== DIAGNOSTIC FUNCTIONS ====================

def diagnose_batchnorm_stability(model, prev_bn_stats=None):
    """Check BatchNorm running statistics stability.
    
    Returns:
        dict with 'mean_drift', 'var_stability', 'status', and current stats
    """
    current_stats = {}
    mean_drifts = []
    var_values = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            running_mean = module.running_mean.detach().cpu().numpy()
            running_var = module.running_var.detach().cpu().numpy()
            
            current_stats[name] = {
                'mean': running_mean.copy(),
                'var': running_var.copy()
            }
            
            var_values.append(running_var.mean())
            
            # Check drift from previous epoch
            if prev_bn_stats and name in prev_bn_stats:
                drift = np.abs(running_mean - prev_bn_stats[name]['mean']).mean()
                mean_drifts.append(drift)
    
    if not current_stats:
        return {'mean_drift': 0, 'var_stability': 1.0, 'status': 'NO_BN', 'stats': {}}
    
    avg_drift = np.mean(mean_drifts) if mean_drifts else 0
    avg_var = np.mean(var_values) if var_values else 1.0
    
    # Determine status
    if avg_drift > 0.1:
        status = 'UNSTABLE'
    elif avg_drift > 0.05:
        status = 'MODERATE'
    else:
        status = 'STABLE'
    
    return {
        'mean_drift': avg_drift,
        'var_mean': avg_var,
        'status': status,
        'stats': current_stats
    }


def diagnose_loss_components(model, criterion, data_snapshot, entropy_weight=0.0):
    """Compute individual loss components to detect conflicts.
    
    Returns:
        dict with 'focal', 'entropy_term', 'total', 'confidence'
    """
    model.eval()
    with torch.no_grad():
        result = model(data_snapshot.x, data_snapshot.edge_index, data_snapshot.edge_attr)
        out = result[0] if isinstance(result, tuple) else result
        targets = data_snapshot.y.clone().to(torch.long)
        labelled = targets >= 0
        out, targets = out[labelled], targets[labelled]
        
        # Focal/CE loss component
        focal_loss = criterion(out, targets).item()
        
        # Entropy regularization component (if used)
        entropy_term = 0.0
        if entropy_weight > 0:
            entropy_term = entropy_weight * entropy_regularization(out).item()
        
        # Model confidence (max probability)
        probs = torch.softmax(out, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Check if predictions are near uniform (sign of over-regularization)
        uniformity = (probs.std(dim=-1).mean().item())  # Low std = uniform
        
    model.train()
    
    return {
        'focal': focal_loss,
        'entropy_term': entropy_term,
        'total': focal_loss + entropy_term,
        'confidence': confidence,
        'uniformity': uniformity
    }


def diagnose_f1_variance(f1_history, window=5):
    """Analyze F1 variance over sliding window to detect oscillation.
    
    Returns:
        dict with 'mean', 'std', 'trend', 'status'
    """
    if len(f1_history) < window:
        return {'mean': 0, 'std': 0, 'trend': 0, 'status': 'INSUFFICIENT_DATA'}
    
    recent = f1_history[-window:]
    mean_f1 = np.mean(recent)
    std_f1 = np.std(recent)
    
    # Calculate trend (positive = improving, negative = degrading)
    if len(f1_history) >= window + 5:
        prev_window = f1_history[-(window+5):-5]
        trend = mean_f1 - np.mean(prev_window)
    else:
        trend = 0
    
    # Determine status
    if std_f1 > 0.15:
        status = 'HIGH_OSCILLATION'
    elif std_f1 > 0.08:
        status = 'MODERATE_OSCILLATION'
    else:
        status = 'STABLE'
    
    if trend < -0.1 and status == 'STABLE':
        status = 'DEGRADING'
    
    return {
        'mean': mean_f1,
        'std': std_f1,
        'trend': trend,
        'status': status
    }


def print_diagnostics(epoch, bn_diag, loss_diag, f1_diag, fold_label):
    """Print formatted diagnostic summary."""
    print(f"  [{fold_label}] === DIAGNOSTICS Epoch {epoch} ===")
    
    # BatchNorm status
    bn_status = bn_diag['status']
    bn_emoji = '✅' if bn_status == 'STABLE' else ('⚠️' if bn_status == 'MODERATE' else '❌')
    print(f"    BatchNorm: {bn_emoji} {bn_status} (drift={bn_diag['mean_drift']:.4f}, var_mean={bn_diag.get('var_mean', 0):.4f})")
    
    # Loss components
    conf_status = '✅' if loss_diag['confidence'] > 0.5 else ('⚠️' if loss_diag['confidence'] > 0.4 else '❌')
    print(f"    Loss: Focal={loss_diag['focal']:.4f}, Entropy={loss_diag['entropy_term']:.4f}, "
          f"Total={loss_diag['total']:.4f}")
    print(f"    Confidence: {conf_status} {loss_diag['confidence']:.3f} (uniformity={loss_diag['uniformity']:.3f})")
    
    # F1 stability
    if f1_diag['status'] != 'INSUFFICIENT_DATA':
        f1_emoji = '✅' if f1_diag['status'] == 'STABLE' else ('⚠️' if 'MODERATE' in f1_diag['status'] else '❌')
        print(f"    F1 Window: {f1_emoji} {f1_diag['status']} (mean={f1_diag['mean']:.4f}, std={f1_diag['std']:.4f}, trend={f1_diag['trend']:+.4f})")
    
    # Recommendations
    recommendations = []
    if bn_status == 'UNSTABLE':
        recommendations.append('Consider LayerNorm instead of BatchNorm')
    if loss_diag['confidence'] < 0.4:
        recommendations.append('Over-regularized: reduce entropy_reg or label_smoothing')
    if f1_diag.get('status') == 'HIGH_OSCILLATION':
        recommendations.append('LR too high: add decay or reduce LR')
    if f1_diag.get('status') == 'DEGRADING':
        recommendations.append('Model degrading: consider early stopping sooner')
    
    if recommendations:
        print(f"    💡 RECOMMENDATIONS: {'; '.join(recommendations)}")
    print()


def train_epoch(model, optimizer, criterion, train_loader, grad_clip=None, entropy_weight=0.0, classifier_l2=0.0):
    """Train for one epoch with optional entropy regularization and classifier L2 penalty.
    
    entropy_weight: coefficient for entropy regularization (0.0 = disabled).
                   Higher values encourage more diverse predictions, preventing mode collapse.
    classifier_l2: L2 penalty weight on classifier layer only (separate from optimizer weight_decay).
                  Helps prevent overconfident predictions.
    
    Returns:
        dict with 'loss', 'grad_norm', 'entropy' for monitoring
    """
    model.train()
    total_loss = 0
    total_grad_norm = 0
    total_entropy = 0
    num_batches = 0
    loader_len = len(list(train_loader))
    hidden = None  # GRU hidden state carried across snapshots within the epoch
    # Reset temporal attention history buffer at start of each epoch
    if hasattr(model, "reset_history"):
        model.reset_history()
    for snapshot in train_loader:
        optimizer.zero_grad()
        result = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, hidden=hidden)
        out, hidden = result if isinstance(result, tuple) else (result, None)
        if hidden is not None:
            hidden = hidden.detach()  # truncated BPTT — don't backprop through prior snapshots
        targets_long = snapshot.y.clone().to(torch.long)
        # Nodes labelled -1 have no drought index for the target month (index
        # warm-up, or the station missing from the raw record). They stay in
        # the graph so neighbours can still aggregate their features, but they
        # must not contribute a loss term — there is no truth to compare to.
        labelled = targets_long >= 0
        if not bool(labelled.any()):
            continue
        out_l, targets_long = out[labelled], targets_long[labelled]
        loss = criterion(out_l, targets_long)
        
        # Track prediction entropy for collapse detection
        batch_entropy = compute_prediction_entropy(out_l)
        total_entropy += batch_entropy
        
        # Add entropy regularization to prevent mode collapse
        # When model collapses to predicting one class, entropy is low
        # This term encourages spreading predictions across classes
        if entropy_weight > 0:
            entropy_loss = entropy_regularization(out_l)
            loss = loss + entropy_weight * entropy_loss
        
        # Add L2 penalty on classifier layer (separate from weight_decay)
        # This helps prevent overconfident predictions by penalizing large classifier weights
        if classifier_l2 > 0 and hasattr(model, 'classifier'):
            l2_penalty = sum(p.pow(2).sum() for p in model.classifier.parameters())
            loss = loss + classifier_l2 * l2_penalty
        
        loss.backward()
        
        # Monitor gradient norm before clipping
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm
        
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / loader_len if loader_len > 0 else total_loss
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    avg_entropy = total_entropy / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'grad_norm': avg_grad_norm,
        'entropy': avg_entropy
    }


def test_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    hidden = None  # GRU hidden state carried across snapshots within the evaluation pass
    # Reset temporal attention history buffer at start of evaluation
    if hasattr(model, "reset_history"):
        model.reset_history()
    with torch.no_grad():
        for snapshot in loader:
            result = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, hidden=hidden)
            out, hidden = result if isinstance(result, tuple) else (result, None)
            preds = out.argmax(dim=1)
            probs = out.exp()  # since model outputs log_softmax
            # Same masking as training: unlabelled nodes are predicted (the
            # forward pass needs the full graph) but never scored.
            labels = snapshot.y.to(torch.long)
            labelled = (labels >= 0).cpu().numpy()
            all_preds.extend(preds.cpu().numpy()[labelled])
            all_labels.extend(labels.cpu().numpy()[labelled])
            all_probs.extend(probs.cpu().numpy()[labelled])
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def generate_param_grid(cfg_in: dict):
    list_keys = [k for k, v in cfg_in.items() if isinstance(v, list) and len(v) > 0]
    if not list_keys:
        return [cfg_in.copy()]
    pools = [cfg_in[k] for k in list_keys]
    grids = []
    for vals in itertools.product(*pools):
        c = cfg_in.copy()
        for k, v in zip(list_keys, vals):
            c[k] = v
        grids.append(c)
    return grids


def _print_class_balance_for_dataset(name: str, dataset, num_classes: int):
    """Print simple class counts and percentages over all snapshots in a dataset."""
    all_labels = []

    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    for i in range(n):
        y = dataset[i].y
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        y = np.asarray(y)
        if y.size:
            all_labels.append(y.astype(int))

    if not all_labels:
        print(f"No labels found for {name}; skipping class balance.")
        return

    labels = np.concatenate(all_labels, axis=0)
    n_unlabelled = int((labels < 0).sum())
    labels = labels[labels >= 0]          # sentinel -1 = no drought index
    if labels.size == 0:
        print(f"No labels found for {name}; skipping class balance.")
        return
    counts = np.bincount(labels, minlength=num_classes)
    total = labels.size

    print("\n" + "=" * 50)
    print(f"CLASS BALANCE - {name}")
    print("=" * 50)
    print(f"Total labeled samples: {total}"
          + (f"  ({n_unlabelled} unlabelled nodes masked out)" if n_unlabelled else ""))
    for cls in range(num_classes):
        count = int(counts[cls])
        pct = (100.0 * count / total) if total > 0 else 0.0
        print(f"Class {cls}: {count} samples ({pct:.2f}%)")
    print("=" * 50)

def _save_confusion_matrix(cm: np.ndarray, classes: list, out_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _save_per_class_bars(per_class_f1: np.ndarray, classes: list, out_path: Path):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=classes, y=per_class_f1)
    plt.xlabel('Class')
    plt.ylabel('F1-score')
    plt.title('Per-class F1')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _save_per_run_boxplot(per_class_f1: np.ndarray, out_path: Path):
    # Box-and-whisker of F1 distribution across classes for a single run
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=per_class_f1)
    sns.stripplot(y=per_class_f1, color='black', alpha=0.6)
    plt.ylabel('F1-score (across classes)')
    plt.title('F1 Distribution Across Classes (Run)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: list, out_path_combined: Path, out_dir: Path):
    # Multiclass One-vs-Rest ROC curves
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # Combined plot
    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_val = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={auc_val:.3f})")
        # Individual plot
        ind_path = out_dir / f"roc_curve_class_{classes[i]}.png"
        plt_ind = plt.figure(figsize=(5,4))
        ax = plt_ind.gca()
        ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
        ax.plot([0,1],[0,1],'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - Class {classes[i]}')
        ax.legend()
        plt_ind.tight_layout()
        plt_ind.savefig(ind_path)
        plt.close(plt_ind)
    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_combined)
    plt.close()

def _compute_and_save_detailed_metrics(run_id: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, num_classes: int, run_dir: Path, run_plots_dir: Path):
    classes = list(range(num_classes))
    # Core metrics
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision, recall, f1_per_class, support = precision_recall_fscore_support(labels, preds, labels=classes, average=None, zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro')

    # Per-run directories are supplied by the caller and already scoped by model type
    # (e.g. save_dir/<model_type>/<run_id>/ and plots_dir/<model_type>/<run_id>/).
    run_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=classes)
    cm_path_img = run_plots_dir / f"{run_id}_confusion_matrix.png"
    cm_path_csv = run_dir / f"{run_id}_confusion_matrix.csv"
    _save_confusion_matrix(cm, classes, cm_path_img)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_path_csv, index_label='true\\pred')

    # Per-class metrics CSV
    per_class_df = pd.DataFrame({
        'class': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1_per_class,
        'support': support,
    })
    per_class_csv = run_dir / f"{run_id}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # Per-class bar chart
    per_class_plot = run_plots_dir / f"{run_id}_per_class_f1.png"
    _save_per_class_bars(f1_per_class, classes, per_class_plot)

    # Per-run box-and-whisker plot of F1 across classes
    per_run_box_plot = run_plots_dir / f"{run_id}_per_class_f1_box.png"
    _save_per_run_boxplot(f1_per_class, per_run_box_plot)

    # ROC / AUC (One-vs-Rest) if possible
    roc_auc_macro = None
    try:
        y_true_bin = label_binarize(labels, classes=classes)
        roc_auc_macro = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
        roc_combined = run_plots_dir / f"{run_id}_roc_curves.png"
        _save_roc_curves(labels, probs, classes, roc_combined, run_plots_dir)
    except Exception as e:
        print(f"Skipping ROC/AUC generation: {e}")

    # Write a summary metrics CSV
    summary = {
        'run_id': run_id,
        'split': 'test',
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'roc_auc_macro_ovr': roc_auc_macro if roc_auc_macro is not None else '',
    }
    summary_csv = run_dir / f"{run_id}_summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"Saved detailed metrics for {run_id} to {run_dir} and plots to {run_plots_dir}")


def _generate_summary_plots(results_csv: Path, plots_dir: Path):
    """Generate summary graphs from grid_results.csv into plots_dir."""
    if not results_csv.exists():
        print(f"No results CSV at {results_csv}; skipping summary plots.")
        return
    try:
        df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"Failed to read results CSV: {e}")
        return

    # Support both old ('f1') and new ('val_f1') column names
    f1_col = 'val_f1' if 'val_f1' in df.columns else 'f1'

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Bar plot of final F1 by run_id
    try:
        plt.figure(figsize=(12, 6))
        order = df.sort_values(f1_col, ascending=False)['run_id'] if 'run_id' in df else None
        sns.barplot(data=df, x='run_id', y=f1_col, order=order)
        plt.xticks(rotation=45, ha='right')
        plt.title('Final F1 by Run')
        plt.tight_layout()
        out_path = plots_dir / 'f1_by_run.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Saved summary plot: {out_path}")
    except Exception as e:
        print(f"Failed to generate F1 by run plot: {e}")

    # 2) Optional heatmap if hyperparams present in params JSON
    try:
        if 'params' in df:
            # Expand params JSON into columns
            params_expanded = df['params'].apply(lambda s: pd.json_normalize(json.loads(s)) if isinstance(s, str) else pd.json_normalize(s))
            params_df = pd.concat(params_expanded.tolist(), ignore_index=True)
            merged = pd.concat([df.drop(columns=['params']), params_df], axis=1)

            # Only proceed if we have variation in lr and hidden_channels
            if 'lr' in merged.columns and 'hidden_channels' in merged.columns:
                # Ensure scalar values (not lists)
                merged['lr'] = merged['lr'].apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else v)
                merged['hidden_channels'] = merged['hidden_channels'].apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else v)
                pivot = merged.pivot_table(index='hidden_channels', columns='lr', values=f1_col, aggfunc='max')
                if pivot.size > 0:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
                    plt.title('F1 Heatmap: hidden_channels vs lr (max across runs)')
                    plt.tight_layout()
                    out_path = plots_dir / 'f1_heatmap_hidden_vs_lr.png'
                    plt.savefig(out_path)
                    plt.close()
                    print(f"Saved summary plot: {out_path}")
    except Exception as e:
        print(f"Failed to generate hyperparam heatmap: {e}")

    # 3) Box-and-whisker of per-class F1 across runs
    try:
        # Collect per-class metrics from all runs. With the models/<model_type>/<run_id>/
        # layout, per-class CSVs live several levels deep — use rglob.
        models_dir = plots_dir.parent / 'models' if plots_dir.parent.name == 'artifacts' else plots_dir.parent
        per_class_files = list(models_dir.rglob('*_per_class_metrics.csv'))
        if per_class_files:
            all_rows = []
            for f in per_class_files:
                try:
                    df_pc = pd.read_csv(f)
                    # infer run_id from filename stem
                    run_id = f.stem.replace('_per_class_metrics', '')
                    df_pc['run_id'] = run_id
                    all_rows.append(df_pc[['run_id', 'class', 'f1']])
                except Exception:
                    continue
            if all_rows:
                agg = pd.concat(all_rows, ignore_index=True)
                plt.figure(figsize=(10, 5))
                sns.boxplot(data=agg, x='class', y='f1')
                sns.stripplot(data=agg, x='class', y='f1', color='black', alpha=0.4)
                plt.xlabel('Class')
                plt.ylabel('F1-score')
                plt.title('Per-class F1 Distribution Across Runs')
                plt.tight_layout()
                out_path = plots_dir / 'f1_per_class_box_across_runs.png'
                plt.savefig(out_path)
                plt.close()
                print(f"Saved summary plot: {out_path}")
    except Exception as e:
        print(f"Failed to generate per-class F1 boxplot across runs: {e}")


def _dataset_class_counts(dataset, num_classes: int) -> np.ndarray:
    """Collect class counts across every snapshot in a temporal dataset.

    Used by the class_weight_source machinery so we can compute inverse-frequency
    weights (or priors for logit adjustment) from an arbitrary dataset — the fold's
    train split, the outer trainval pool, or the full dataset — rather than being
    locked to the train fold.
    """
    all_labels = []
    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    for i in range(n):
        y = dataset[i].y
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        all_labels.append(np.asarray(y).astype(int))
    if not all_labels:
        return np.zeros(num_classes, dtype=float)
    concat = np.concatenate(all_labels)
    concat = concat[concat >= 0]        # drop unlabelled nodes (sentinel -1)
    if concat.size == 0:
        return np.zeros(num_classes, dtype=float)
    return np.bincount(concat, minlength=num_classes).astype(float)


def _train_one_fold(
    fold_label: str,
    train_dataset,
    val_dataset,
    cfg_run: dict,
    cfg: dict,
    device,
    run_models_dir: Path,
    num_classes_run: int,
    trainval_dataset=None,
    run_plots_dir: Path | None = None,
) -> dict:
    """
    Create a fresh model, run the training loop with early stopping, reload the best
    checkpoint, evaluate on val_dataset, save model state, and return a results dict.

    Returned dict keys:
        model, best_model_path, model_save_path, best_f1,
        val_acc, val_f1, val_preds, val_labels, val_probs, duration, history.
    """
    train_loader = TemporalDataLoader(train_dataset)
    val_loader   = TemporalDataLoader(val_dataset)

    # --- Model ---
    # model_type takes precedence: "simple_gcn", "temporal_gru", "temporal_attention"
    model_type = cfg_run.get("model_type", cfg.get("model_type", None))
    use_attention = bool(cfg_run.get("use_attention", cfg.get("use_attention", False)))
    use_temporal_attention = bool(cfg_run.get("use_temporal_attention", cfg.get("use_temporal_attention", False)))
    use_simple_gcn = bool(cfg_run.get("use_simple_gcn", cfg.get("use_simple_gcn", False)))
    
    # Override boolean flags if model_type is explicitly set
    if model_type == "simple_gcn":
        use_simple_gcn, use_temporal_attention = True, False
    elif model_type == "temporal_gru":
        use_simple_gcn, use_temporal_attention = False, False
    elif model_type == "temporal_attention":
        use_simple_gcn, use_temporal_attention = False, True
    
    _gru_layers   = int(cfg_run.get("gru_layers",    cfg.get("gru_layers",    1)))
    _input_norm   = bool(cfg_run.get("input_norm",   cfg.get("input_norm",    True)))
    _in_channels  = int(cfg_run.get("in_channels", cfg.get("in_channels", 10)))
    _hidden_channels = int(cfg_run.get("hidden_channels", cfg.get("hidden_channels", 16)))
    _dropout = float(cfg_run.get("dropout", cfg.get("dropout", 0.3)))
    _gcn_layers = int(cfg_run.get("gcn_layers", cfg.get("gcn_layers", 3)))
    
    if use_simple_gcn:
        # Simple GCN model WITHOUT GRU - tests if recurrent component causes collapse
        print(f"  [{fold_label}] Using SimpleGCNClassifier (no GRU, {_gcn_layers} GCN layers, dropout={_dropout})")
        model = SimpleGCNClassifier(
            in_channels=_in_channels,
            hidden_channels=_hidden_channels,
            num_classes=num_classes_run,
            dropout=_dropout,
            input_norm=_input_norm,
            num_layers=_gcn_layers,
        ).to(device)
    elif use_temporal_attention:
        # New temporal attention model with history buffer
        from masters_research.models.temporal_attention_gnn import TemporalAttentionGNN
        _history_len = int(cfg_run.get("temporal_history_len", cfg.get("temporal_history_len", 10)))
        _temporal_heads = int(cfg_run.get("temporal_heads", cfg.get("temporal_heads", 4)))
        # Attention-specific stability knobs. attention_gate_init=0 (ReZero) is the
        # default because it fixes the class-0<->class-2 sign-flip we observed
        # across seeds. attention_dropout defaults to the network's dropout if
        # unset; overriding lets you regularise the attention layer more heavily
        # than the rest of the model.
        _attn_gate_init = float(cfg_run.get("attention_gate_init", cfg.get("attention_gate_init", 0.0)))
        _attn_dropout_cfg = cfg_run.get("attention_dropout", cfg.get("attention_dropout", None))
        _attn_dropout = float(_attn_dropout_cfg) if _attn_dropout_cfg is not None else None
        print(f"  [{fold_label}] Using TemporalAttentionGNN "
              f"(history_len={_history_len}, heads={_temporal_heads}, "
              f"dropout={_dropout}, attn_dropout={_attn_dropout}, "
              f"attn_gate_init={_attn_gate_init})")
        model = TemporalAttentionGNN(
            in_channels=_in_channels,
            hidden_channels=_hidden_channels,
            num_classes=num_classes_run,
            history_len=_history_len,
            temporal_heads=_temporal_heads,
            gru_layers=_gru_layers,
            dropout=_dropout,
            input_norm=_input_norm,
            use_gat=use_attention,
            gat_heads=int(cfg_run.get("gat_heads", cfg.get("gat_heads", 4))),
            attention_dropout=_attn_dropout,
            attention_gate_init=_attn_gate_init,
        ).to(device)
    elif use_attention:
        try:
            from masters_research.models.temporal_spatial_attention import TemporalSpatialAttentionNodeClassifier
            model = TemporalSpatialAttentionNodeClassifier(
                in_channels=_in_channels,
                hidden_channels=_hidden_channels,
                num_classes=num_classes_run,
                heads=int(cfg_run.get("gat_heads", cfg.get("gat_heads", 4))),
                gru_layers=_gru_layers,
                dropout=_dropout,
                input_norm=_input_norm,
            ).to(device)
        except Exception:
            print(f"  [{fold_label}] Attention model unavailable; falling back to baseline GCN.")
            model = TemporalSpatialNodeClassifier(
                in_channels=_in_channels,
                hidden_channels=_hidden_channels,
                num_classes=num_classes_run,
                gru_layers=_gru_layers,
                dropout=_dropout,
                input_norm=_input_norm,
            ).to(device)
    else:
        model = TemporalSpatialNodeClassifier(
            in_channels=_in_channels,
            hidden_channels=_hidden_channels,
            num_classes=num_classes_run,
            gru_layers=_gru_layers,
            dropout=_dropout,
            input_norm=_input_norm,
        ).to(device)

    # --- Optimizer ---
    opt_name = str(cfg_run.get("optimizer", cfg.get("optimizer", "adam"))).lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(cfg_run.get("lr", cfg.get("lr", 0.001))),
            momentum=float(cfg_run.get("momentum", cfg.get("momentum", 0.9))),
            weight_decay=float(cfg_run.get("weight_decay", cfg.get("weight_decay", 5e-4))),
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(cfg_run.get("lr", cfg.get("lr", 0.001))),
            weight_decay=float(cfg_run.get("weight_decay", cfg.get("weight_decay", 5e-4))),
        )

    # --- Class weights (source configurable via class_weight_source) ---
    # "train"    → weights from this fold's train split (legacy default).
    # "trainval" → weights from the outer trainval pool if available. Preferred when
    #              per-fold inner splits show large class-prior drift (val≠train),
    #              since the trainval pool's distribution is more representative of
    #              the eventual test-time prior.
    # "uniform"  → equal weights (i.e. no reweighting). Useful when combined with
    #              logit_adjustment, which corrects for prior shift directly.
    weight_source = str(cfg_run.get("class_weight_source", cfg.get("class_weight_source", "train"))).lower()
    if weight_source == "trainval" and trainval_dataset is not None:
        source_ds = trainval_dataset
    elif weight_source == "uniform":
        source_ds = None
    else:
        source_ds = train_dataset

    if source_ds is None:
        _counts = np.ones(num_classes_run, dtype=float)
    else:
        _counts = _dataset_class_counts(source_ds, num_classes_run)
    _w = 1.0 / np.maximum(_counts, 1)
    _w = (_w / _w.sum()) * num_classes_run
    class_weights = torch.tensor(_w, dtype=torch.float).to(device)
    print(f"  [{fold_label}] Class weights (source={weight_source}): {[f'{v:.3f}' for v in _w]}")

    # --- Logit adjustment prior (Menon et al. 2020) ---
    # If enabled, we shift training-time logits by log(P(y=c)) so the CE optimises
    # the *balanced* posterior. Prior is estimated from the same dataset chosen
    # for class weights (usually train or trainval).
    use_logit_adjustment = bool(cfg_run.get("logit_adjustment", cfg.get("logit_adjustment", False)))
    log_prior_tensor = None
    if use_logit_adjustment:
        prior_ds = source_ds if source_ds is not None else train_dataset
        prior_counts = _dataset_class_counts(prior_ds, num_classes_run)
        prior_counts = np.maximum(prior_counts, 1)  # avoid log(0)
        prior = prior_counts / prior_counts.sum()
        log_prior_tensor = torch.tensor(np.log(prior), dtype=torch.float, device=device)
        print(f"  [{fold_label}] Logit adjustment enabled — log-prior: {[f'{v:.3f}' for v in np.log(prior)]}")

    # --- Loss function ---
    use_focal   = bool(cfg_run.get("focal_loss",  cfg.get("focal_loss",  True)))
    focal_gamma = float(cfg_run.get("focal_gamma", cfg.get("focal_gamma", 2.0)))
    label_smoothing = float(cfg_run.get("label_smoothing", cfg.get("label_smoothing", 0.0)))
    entropy_weight = float(cfg_run.get("entropy_reg", cfg.get("entropy_reg", 0.0)))
    classifier_l2 = float(cfg_run.get("classifier_l2", cfg.get("classifier_l2", 0.0)))
    use_ordinal = bool(cfg_run.get("ordinal_loss", cfg.get("ordinal_loss", False)))
    ordinal_lambda = float(cfg_run.get("ordinal_lambda", cfg.get("ordinal_lambda", 0.5))) if use_ordinal else 0.0

    if use_focal or use_ordinal or use_logit_adjustment:
        # FocalLoss now carries both logit adjustment and the ordinal penalty, so
        # we take this branch whenever any of the three is on. gamma=0 recovers CE
        # when the caller opted in for adjustment/ordinal only.
        effective_gamma = focal_gamma if use_focal else 0.0
        criterion = FocalLoss(
            gamma=effective_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing,
            log_prior=log_prior_tensor,
            ordinal_lambda=ordinal_lambda,
        )
        active = []
        if use_focal:
            active.append(f"focal(gamma={effective_gamma})")
        if use_logit_adjustment:
            active.append("logit_adjustment")
        if use_ordinal:
            active.append(f"ordinal(lambda={ordinal_lambda})")
        print(f"  [{fold_label}] Loss = FocalLoss[{', '.join(active) or 'ce'}] "
              f"label_smoothing={label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"  [{fold_label}] Using weighted CrossEntropyLoss (label_smoothing={label_smoothing})")
    
    if entropy_weight > 0:
        print(f"  [{fold_label}] Entropy regularization enabled (weight={entropy_weight})")
    if classifier_l2 > 0:
        print(f"  [{fold_label}] Classifier L2 penalty enabled (weight={classifier_l2})")

    # --- Scheduler ---
    num_epochs      = int(cfg_run.get("epochs", cfg.get("epochs", 200)))  # Needed for scheduler setup
    scheduler_cfg = cfg_run.get("scheduler", cfg.get("scheduler", None))
    scheduler = None
    if isinstance(scheduler_cfg, dict):
        _sname = scheduler_cfg.get("name", "")
        if _sname == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 10)),
                gamma=float(scheduler_cfg.get("gamma", 0.5)),
            )
            print(f"  [{fold_label}] StepLR scheduler (step_size={scheduler_cfg.get('step_size', 10)}, gamma={scheduler_cfg.get('gamma', 0.5)})")
        elif _sname == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 10)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
            )
            print(f"  [{fold_label}] ReduceLROnPlateau scheduler "
                  f"(factor={scheduler_cfg.get('factor', 0.5)}, patience={scheduler_cfg.get('patience', 10)})")
        elif _sname == "CosineAnnealingLR":
            # Cosine annealing: smoothly decays LR from initial to eta_min over T_max epochs
            # Helps prevent overfitting by reducing LR as training progresses
            T_max = int(scheduler_cfg.get("T_max", num_epochs))
            eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
            print(f"  [{fold_label}] CosineAnnealingLR scheduler (T_max={T_max}, eta_min={eta_min})")

    # --- Training loop ---
    print(f"  [{fold_label}] Starting training...")
    start_time      = time.time()
    best_f1         = -1.0
    # Raw (unsmoothed) val F1 at the epoch whose checkpoint was retained.
    # best_f1 may hold a smoothed selection score, so keep the raw value
    # for any comparison that must stay on the same scale (e.g. SWA).
    best_raw_f1     = -1.0
    best_model_path = None
    eval_interval   = int(cfg_run.get("eval_interval",         cfg.get("eval_interval",         1)))
    min_delta       = float(cfg_run.get("early_stopping_min_delta", cfg.get("early_stopping_min_delta", 0.0)))
    patience        = int(cfg_run.get("early_stopping_patience",    cfg.get("early_stopping_patience",    0)))
    patience_counter = 0
    grad_clip       = cfg_run.get("grad_clip", cfg.get("grad_clip", None))
    last_val_f1     = -1.0
    
    # Learning rate warmup - start at LR/3, linearly ramp to full LR over warmup_epochs
    warmup_epochs   = int(cfg_run.get("warmup_epochs", cfg.get("warmup_epochs", 0)))
    base_lr         = float(cfg_run.get("lr", cfg.get("lr", 0.001)))
    warmup_start_lr = base_lr / 3.0  # More aggressive warmup start (was /10)
    if warmup_epochs > 0:
        # Set initial LR to warmup_start_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
        print(f"  [{fold_label}] LR warmup enabled: {warmup_start_lr:.6f} → {base_lr:.6f} over {warmup_epochs} epochs")
    
    # SWA (Stochastic Weight Averaging) - helps escape local minima by averaging weights
    # from multiple points in the training trajectory, finding flatter minima that generalize better
    use_swa = cfg_run.get("use_swa", cfg.get("use_swa", False))
    swa_start_epoch = int(cfg_run.get("swa_start_epoch", cfg.get("swa_start_epoch", 20)))
    swa_lr = float(cfg_run.get("swa_lr", cfg.get("swa_lr", 0.0005)))
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"  [{fold_label}] SWA enabled: starts at epoch {swa_start_epoch}, swa_lr={swa_lr}")

    # Collapse detection parameters
    collapse_entropy_threshold = 0.3  # Below this entropy (bits), consider it collapse
    collapse_lr_reduction = 0.5       # Reduce LR by this factor when collapse detected
    collapse_detected_count = 0       # Track consecutive low-entropy epochs
    collapse_patience = 3             # Consecutive low-entropy epochs before reducing LR
    grad_norm_spike_threshold = 10.0  # Gradient norm above this triggers adaptive clipping
    baseline_grad_norm = None         # Running average of gradient norms
    adaptive_grad_clip = grad_clip    # May be reduced if gradients spike
    collapse_recovery_count = 0       # How many times we've tried to recover
    max_collapse_recoveries = 3       # Max recovery attempts before giving up
    
    # Diagnostic tracking
    prev_bn_stats = None              # Track BatchNorm stats between epochs
    f1_history = []                   # Track F1 for variance analysis
    diagnostic_interval = 5           # Print diagnostics every N epochs

    # Learning-curve history (see pipeline/training_curves.py). The extra
    # train-set pass is inference-only, so it does not alter the trajectory.
    record_curves = bool(cfg_run.get("record_training_curves", cfg.get("record_training_curves", True)))
    curve_history = []
    selected_epoch = None
    
    for epoch in range(1, num_epochs + 1):
        # Learning rate warmup: linearly increase LR from warmup_start_lr to base_lr
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        train_stats = train_epoch(model, optimizer, criterion, train_loader, grad_clip=adaptive_grad_clip, entropy_weight=entropy_weight, classifier_l2=classifier_l2)
        loss = train_stats['loss']
        epoch_grad_norm = train_stats['grad_norm']
        epoch_entropy = train_stats['entropy']
        
        # Update baseline gradient norm (exponential moving average)
        if baseline_grad_norm is None:
            baseline_grad_norm = epoch_grad_norm
        else:
            baseline_grad_norm = 0.9 * baseline_grad_norm + 0.1 * epoch_grad_norm
        
        # Gradient spike detection and adaptive clipping
        if baseline_grad_norm > 0 and epoch_grad_norm > grad_norm_spike_threshold * baseline_grad_norm:
            old_clip = adaptive_grad_clip if adaptive_grad_clip else float('inf')
            adaptive_grad_clip = max(0.1, (adaptive_grad_clip or 1.0) * 0.5)
            print(f"  [{fold_label}] ⚠️ Gradient spike detected (norm={epoch_grad_norm:.2f}, baseline={baseline_grad_norm:.2f}). "
                  f"Reducing grad_clip: {old_clip:.2f} → {adaptive_grad_clip:.2f}")
        
        # Collapse detection based on entropy
        if epoch_entropy < collapse_entropy_threshold:
            collapse_detected_count += 1
            if collapse_detected_count >= collapse_patience and collapse_recovery_count < max_collapse_recoveries:
                # Reduce learning rate to recover from collapse
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * collapse_lr_reduction
                collapse_recovery_count += 1
                print(f"  [{fold_label}] 🚨 COLLAPSE DETECTED (entropy={epoch_entropy:.3f} < {collapse_entropy_threshold}, "
                      f"count={collapse_detected_count}). Reducing LR by {collapse_lr_reduction}x (recovery #{collapse_recovery_count})")
                collapse_detected_count = 0  # Reset counter after intervention
        else:
            collapse_detected_count = max(0, collapse_detected_count - 1)  # Slowly recover counter
        
        do_eval = (epoch % eval_interval == 0) or (epoch == num_epochs)
        if do_eval:
            if record_curves:
                tr_preds, tr_labels, tr_probs = test_model(model, train_loader)
            preds, labels, probs = test_model(model, val_loader)
            acc = accuracy_score(labels, preds)
            f1  = f1_score(labels, preds, average="weighted")
            f1_history.append(f1)  # Track for variance analysis
            if record_curves:
                curve_history.append(training_curves.history_row(
                    epoch, tr_labels, tr_preds, tr_probs, labels, preds, probs,
                    objective=float(loss), lr=float(optimizer.param_groups[0]['lr']),
                ))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [{fold_label}] Epoch {epoch:03d}/{num_epochs} | "
                  f"Loss: {loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f} | "
                  f"LR: {current_lr:.6f} | Entropy: {epoch_entropy:.3f} | GradNorm: {epoch_grad_norm:.2f}")
            
            # Run diagnostics every N epochs or when performance drops significantly
            run_diagnostics = (epoch % diagnostic_interval == 0) or \
                             (len(f1_history) > 1 and f1 < f1_history[-2] - 0.1)
            
            if run_diagnostics:
                # Get one validation snapshot for loss component analysis
                val_snapshot = next(iter(val_loader))
                
                bn_diag = diagnose_batchnorm_stability(model, prev_bn_stats)
                loss_diag = diagnose_loss_components(model, criterion, val_snapshot, entropy_weight)
                f1_diag = diagnose_f1_variance(f1_history, window=5)
                
                print_diagnostics(epoch, bn_diag, loss_diag, f1_diag, fold_label)
                
                prev_bn_stats = bn_diag['stats']  # Save for next comparison
            
            last_val_f1 = f1

            # --- Checkpoint selection criterion ---
            # Selecting the single best-scoring epoch out of ~60 evaluations on a
            # small validation set takes the maximum of many noisy estimates, which
            # is optimistically biased: it partly selects validation noise rather
            # than genuine skill. Measured effect on this project: GNN runs showed a
            # +0.10..+0.14 val->test gap while the non-checkpoint-selecting baselines
            # showed ~0.00. Averaging the last `selection_window` evaluations before
            # comparing suppresses that noise, so the retained checkpoint reflects a
            # sustained level rather than a lucky epoch.
            # Set selection_window = 1 to recover the original best-epoch behaviour.
            _sel_window = int(cfg_run.get("selection_window", cfg.get("selection_window", 5)))
            if _sel_window > 1:
                selection_score = float(np.mean(f1_history[-_sel_window:]))
            else:
                selection_score = f1

            if selection_score > best_f1 + min_delta:
                best_f1          = selection_score
                best_raw_f1      = f1
                selected_epoch   = epoch
                patience_counter = 0
                if cfg_run.get("save_best", cfg.get("save_best", True)):
                    run_models_dir.mkdir(parents=True, exist_ok=True)
                    best_model_path = run_models_dir / f"best_{fold_label}.pth"
                    torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f"  [{fold_label}] Early stopping at epoch {epoch} "
                          f"(best Val F1: {best_f1:.4f}).")
                    break
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if last_val_f1 >= 0:
                    scheduler.step(last_val_f1)
            else:
                scheduler.step()
        
        # SWA: update averaged model after warmup period
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    duration = time.time() - start_time
    print(f"  [{fold_label}] Training finished in {duration:.1f}s  (best Val F1: {best_f1:.4f}).")

    # Update batch normalization statistics for SWA model
    if use_swa and swa_model is not None:
        print(f"  [{fold_label}] Updating SWA batch norm statistics...")
        # Create a simple loader for BN update (iterate through training data)
        try:
            update_bn(train_loader, swa_model, device=device)
        except Exception as e:
            print(f"  [{fold_label}] SWA BN update skipped (no BN layers or error: {e})")
        
        # Compare SWA model vs best checkpoint
        swa_preds, swa_labels, swa_probs = test_model(swa_model, val_loader)
        swa_f1 = float(f1_score(swa_labels, swa_preds, average="weighted"))
        print(f"  [{fold_label}] SWA Val F1: {swa_f1:.4f} "
              f"(vs best checkpoint raw: {best_raw_f1:.4f})")
        
        # Use SWA model if it's better
        if swa_f1 > best_raw_f1:
            print(f"  [{fold_label}] Using SWA model (better than checkpoint)")
            # Copy SWA weights to the main model
            model.load_state_dict(swa_model.module.state_dict())
            best_f1 = swa_f1
            best_raw_f1 = swa_f1
            selected_epoch = None  # SWA average, not a single epoch
        elif best_model_path and Path(best_model_path).exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    elif best_model_path and Path(best_model_path).exists():
        # Reload best checkpoint so the returned model has the best weights
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # Final val evaluation
    val_preds, val_labels, val_probs = test_model(model, val_loader)
    val_acc = float(accuracy_score(val_labels, val_preds))
    val_f1  = float(f1_score(val_labels, val_preds, average="weighted"))

    # Persist full model state dict in a per-fold subfolder under the per-run models dir
    fold_run_dir = run_models_dir / fold_label
    fold_run_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = fold_run_dir / f"temporal_spatial_classifier_state_{fold_label}.pth"
    torch.save(model.state_dict(), model_save_path)

    if record_curves and curve_history:
        _swa_used = use_swa and selected_epoch is None
        training_curves.save_run_history(
            curve_history, run_models_dir, run_plots_dir,
            title=f"{type(model).__name__} - {fold_label}"
                  + (" (final model = SWA average)" if _swa_used else ""),
            selected_step=selected_epoch,
        )

    return {
        "model":           model,
        "best_model_path": best_model_path,
        "model_save_path": model_save_path,
        "best_f1":         best_f1 if best_f1 >= 0 else val_f1,
        "val_acc":         val_acc,
        "val_f1":          val_f1,
        "val_preds":       val_preds,
        "val_labels":      val_labels,
        "val_probs":       val_probs,
        "duration":        duration,
        "history":         curve_history,
    }


def main(base_path: Path, config_path: Path = None, cv_fold: int | None = None,
         n_outer_folds: int = 5, run_reports: bool = True):
    """
    Main function to run the training and evaluation pipeline.
    
    Parameters
    ----------
    base_path : Path
        Project root directory
    config_path : Path, optional
        Path to config JSON file. If None, defaults to configs/default.json
    cv_fold : int, optional
        When set, this invocation evaluates one fold of the rotating outer
        cross-validation instead of the single fixed hold-out. The fold index
        selects which months are held out AND which months the scalers, seasonal
        statistics and bin edges are fit on, so each fold stays leak-free.
        Driven by outer_cv.py; leave as None for the ordinary single-split run.
    n_outer_folds : int
        Total number of outer folds, used with cv_fold.
    run_reports : bool
        Set False for individual CV folds so the cross-run aggregate reports are
        produced once over the pooled results rather than once per fold.
    """
    base_path = Path(base_path).resolve()
    if config_path is None:
        config_path = base_path / "configs" / "default.json"
    else:
        config_path = Path(config_path).resolve()
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)

    try:
        # Build parameter grid from config
        grid = generate_param_grid(cfg) or [cfg]

        # Output directory and results file
        # Resolve model and plots directories
        default_models = base_path / "src" / "masters_research" / "artifacts" / "models"
        default_plots = base_path / "src" / "masters_research" / "artifacts" / "plots"
        save_dir = Path(cfg.get("save_dir", default_models))
        plots_dir = Path(cfg.get("plots_dir", default_plots))
        save_dir = save_dir if save_dir.is_absolute() else (base_path / save_dir)
        plots_dir = plots_dir if plots_dir.is_absolute() else (base_path / plots_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / "grid_results.csv"
        header = [
            "run_id",
            "model_type",
            "params",
            "val_accuracy",
            "val_f1",
            "final_test_accuracy",
            "final_test_f1",
            "model_path",
            "best_model_path",
            "run_models_dir",
            "run_plots_dir",
            "best_f1",
            "duration_seconds",
        ]

        # Load data from configured path (processedData.csv by default)
        default_data_path = base_path / "Dataset" / "TransientData" / "processedData.csv"
        _data_path = cfg.get("data_path", str(default_data_path))
        data_path = Path(_data_path) if Path(_data_path).is_absolute() else (base_path / _data_path)
        
        print(f"\nLoading data from {data_path}...")
        final_df = pd.read_csv(data_path)
        print(f"Data loaded: {len(final_df)} rows, {len(final_df['TimeStamp'].unique())} timestamps.")

        # Plot nodes per timestamp
        if cfg.get("plot_nodes_per_timestamp", True):
            print("\nGenerating plot of nodes per timestamp...")
            nodes_per_timestamp = final_df.groupby("TimeStamp")["Compno_x"].nunique()
            plt.figure(figsize=(15, 7))
            try:
                nodes_per_timestamp.plot(kind="bar")
            except Exception as e:
                print(f"  Warning: pandas plotting backend unavailable ({e}). Falling back to matplotlib bar plot.")
                plt.bar(nodes_per_timestamp.index.astype(str), nodes_per_timestamp.values)
            plt.title("Number of Nodes per Timestamp")
            plt.xlabel("Timestamp")
            plt.ylabel("Number of Nodes")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = plots_dir / "nodes_per_timestamp.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}\n")

        # Temporal data quality audit (run once, before any modelling)
        if cfg.get("temporal_quality_audit", True):
            run_temporal_quality_audit(
                final_df,
                plots_dir=plots_dir,
                target_col="Target_x",
                node_col="Compno_x",
            )

        # Create dataset with optional seasonal normalization and lag features
        seasonal_norm = cfg.get("seasonal_normalization", True)
        add_lag_features = cfg.get("add_lag_features", True)
        
        # Get num_classes for target discretization (handle list from grid search)
        _nc = cfg.get("num_classes", 5)
        n_classes_for_dataset = int(_nc[0] if isinstance(_nc, list) else _nc)
        
        dataset = create_temporal_dataset(
            final_df,
            seasonal_normalization=seasonal_norm,
            add_lag_features=add_lag_features,
            n_classes=n_classes_for_dataset,
            # Leak-free preprocessing: fit seasonal stats, scaler, and target
            # bin edges on the trainval slice only. These two args must mirror
            # the outer split used later so the mask aligns row-for-row.
            test_holdout_ratio=float(cfg.get("test_holdout_ratio", 0.2)),
            stratified_split=bool(cfg.get("stratified_split", False)),
            # 1 = one-month-ahead forecasting (default); 0 = same-month
            # classification. See create_temporal_dataset for the caveat
            # attached to horizon 0.
            forecast_horizon=int(cfg.get("forecast_horizon", 1)),
            # "feature_mean" (legacy, circular) | "spi" | "spei"
            target_type=str(cfg.get("target_type", "feature_mean")),
            target_window=int(cfg.get("target_window", 3)),
            base_path=base_path,
            cv_fold=cv_fold,
            n_outer_folds=n_outer_folds,
            purge_heldout_lags=bool(cfg.get("purge_heldout_lags", False)),
        )
        
        # Dynamically determine in_channels from actual data
        actual_in_channels = dataset.features[0].shape[1]
        print(f"\n[DYNAMIC] Detected in_channels = {actual_in_channels} from data")
        if cfg.get("in_channels") != actual_in_channels:
            print(f"  Updating in_channels from config value {cfg.get('in_channels')} → {actual_in_channels}")
            cfg["in_channels"] = actual_in_channels
        # Propagate the detected value into every pre-built grid entry: generate_param_grid
        # snapshotted cfg *before* this dynamic detection, so each cfg_run still holds the
        # stale value that would build a BatchNorm/GCN of the wrong width.
        for cfg_run in grid:
            cfg_run["in_channels"] = actual_in_channels
        
        print("\n" + "=" * 50)
        print(" " * 18 + "DATASET INFO")
        print("=" * 50)
        print(dataset)

        # Overall class balance before any splitting
        _nc = cfg.get("num_classes", 5)
        overall_num_classes = int(_nc[0] if isinstance(_nc, list) else _nc)
        _print_class_balance_for_dataset("Full dataset (before any split)", dataset, num_classes=overall_num_classes)

        # --- Outer hold-out split (done once, before grid search) ---
        # final_test_dataset is never used during training or hyperparameter selection.
        test_holdout_ratio = float(cfg.get("test_holdout_ratio", 0.2))
        outer_train_ratio = 1.0 - test_holdout_ratio
        print(f"\nOuter hold-out split: {outer_train_ratio:.0%} train+val / {test_holdout_ratio:.0%} final test...")
        if bool(cfg.get("stratified_split", False)):
            from masters_research.pipeline.dataset import interleaved_temporal_split
            print("  Outer split strategy: interleaved (systematic) sampling — each season is proportionally")
            print("  represented in both the train+val pool and the final test set.")
            trainval_dataset, final_test_dataset = interleaved_temporal_split(
                dataset, train_ratio=outer_train_ratio,
                cv_fold=cv_fold, n_outer_folds=n_outer_folds,
            )
        else:
            trainval_dataset, final_test_dataset = temporal_signal_split(dataset, train_ratio=outer_train_ratio)
        _print_class_balance_for_dataset("Train+Val pool (grid search)", trainval_dataset, num_classes=overall_num_classes)
        _print_class_balance_for_dataset("Final held-out test set", final_test_dataset, num_classes=overall_num_classes)
        if cfg.get("temporal_quality_audit", True):
            compare_splits(
                trainval_dataset,
                final_test_dataset,
                name_a="TrainVal pool",
                name_b="Final test",
                num_classes=overall_num_classes,
                plots_dir=plots_dir,
                plot_prefix="outer_split",
            )
        final_test_loader = TemporalDataLoader(final_test_dataset)

        # Run grid of experiments
        # baselines_only skips the GNN grid but keeps the identical dataset and
        # splits, so baseline results can be refreshed without retraining GNNs.
        if bool(cfg.get("baselines_only", False)):
            print("\nbaselines_only=true: skipping the GNN grid; training baselines only.")
            grid = []

        for run_idx, cfg_run in enumerate(grid, start=1):
            run_prefix = cfg.get("run_name", "run")
            if cv_fold is not None:
                run_prefix = f"{run_prefix}_f{cv_fold}"
            # Canonical architecture name determines the folder layout for this run.
            model_type_name = _model_type_from_cfg(cfg_run)
            # Include model_type in run_id for clarity
            model_type_label = cfg_run.get("model_type", "")
            if model_type_label:
                run_id = f"{run_prefix}_{model_type_label}_{run_idx}_{int(time.time())}"
            else:
                run_id = f"{run_prefix}{run_idx}_{int(time.time())}"

            # Per-run directories: models/<model_type>/<run_id>/ and plots/<model_type>/<run_id>/
            run_models_dir = save_dir / model_type_name / run_id
            run_plots_dir = plots_dir / model_type_name / run_id
            run_models_dir.mkdir(parents=True, exist_ok=True)
            run_plots_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Starting grid run {run_idx}/{len(grid)} ({run_id}) ===")
            print(f"    Model type: {model_type_name}")
            print(f"    Models dir: {run_models_dir}")
            print(f"    Plots dir:  {run_plots_dir}")

            # Print resolved hyperparameters for this run
            print("Run configuration:")
            for key in sorted(cfg_run.keys()):
                print(f"  {key}: {cfg_run[key]}")

            # set per-run seed
            seed = int(cfg_run.get("seed", cfg.get("seed", 42)))
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Resolve num_classes for this run
            _nc_run = cfg_run.get("num_classes", cfg.get("num_classes", 5))
            num_classes_run = int(_nc_run[0] if isinstance(_nc_run, list) else _nc_run)

            # Resolve number of CV folds (1 = single inner split, >1 = walk-forward CV)
            n_cv_folds = int(cfg_run.get("n_cv_folds", cfg.get("n_cv_folds", 1)))

            # Inner split of train+val pool — skipped when using walk-forward CV
            train_dataset = None
            val_dataset   = None
            if n_cv_folds <= 1:
                print("\nSplitting train+val pool into train / validation sets for this run...")
                train_ratio_val = cfg_run.get("train_ratio", cfg.get("train_ratio", 0.8))
                if bool(cfg_run.get("stratified_split", cfg.get("stratified_split", False))):
                    from masters_research.pipeline.dataset import stratified_temporal_signal_split
                    train_dataset, val_dataset = stratified_temporal_signal_split(
                        trainval_dataset,
                        train_ratio=float(train_ratio_val),
                        num_classes=num_classes_run,
                    )
                else:
                    train_dataset, val_dataset = temporal_signal_split(
                        trainval_dataset, train_ratio=train_ratio_val
                    )
                print("Dataset splitting complete.")
                _print_class_balance_for_dataset(f"Train set ({run_id})", train_dataset, num_classes=num_classes_run)
                _print_class_balance_for_dataset(f"Val set ({run_id})", val_dataset, num_classes=num_classes_run)
                if cfg_run.get("temporal_quality_audit", cfg.get("temporal_quality_audit", True)):
                    compare_splits(
                        train_dataset,
                        val_dataset,
                        name_a=f"Train ({run_id})",
                        name_b=f"Val ({run_id})",
                        num_classes=num_classes_run,
                        plots_dir=run_plots_dir,
                        plot_prefix=f"inner_split_{run_id}",
                    )
            else:
                print(f"\nWalk-forward CV ({n_cv_folds} folds): skipping single inner split.")

            # Device selection
            cfg_device = cfg_run.get("device", cfg.get("device", None))
            if cfg_device == "cpu":
                device = torch.device("cpu")
            elif cfg_device == "cuda":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\nUsing device: {device}")

            # ---------------------------------------------------------------
            # Walk-forward temporal CV  (n_cv_folds > 1)  OR single run
            # ---------------------------------------------------------------
            if n_cv_folds > 1:
                from masters_research.pipeline.dataset import walk_forward_cv_split
                print(f"\nGenerating {n_cv_folds} expanding-window CV folds from the train+val pool...")
                cv_folds = list(walk_forward_cv_split(trainval_dataset, n_folds=n_cv_folds))
                if not cv_folds:
                    print("  Warning: dataset too small for walk-forward CV; falling back to single inner split.")
                    n_cv_folds = 1

            if n_cv_folds > 1:
                fold_results      = []
                fold_val_datasets = []
                for fold_train, fold_val, fold_num in cv_folds:
                    fold_id = f"{run_id}_fold{fold_num}"
                    print(f"\n--- CV Fold {fold_num}/{len(cv_folds)} "
                          f"| train: {fold_train.snapshot_count} snapshots "
                          f"| val: {fold_val.snapshot_count} snapshots ---")
                    _print_class_balance_for_dataset(f"Fold {fold_num} train", fold_train, num_classes=num_classes_run)
                    _print_class_balance_for_dataset(f"Fold {fold_num} val",   fold_val,   num_classes=num_classes_run)
                    result = _train_one_fold(
                        fold_id, fold_train, fold_val,
                        cfg_run, cfg, device, run_models_dir, num_classes_run,
                        trainval_dataset=trainval_dataset,
                        run_plots_dir=run_plots_dir,
                    )
                    fold_results.append(result)
                    fold_val_datasets.append(fold_val)

                cv_val_f1_scores  = [r["val_f1"]  for r in fold_results]
                cv_val_acc_scores = [r["val_acc"] for r in fold_results]
                print(f"\nWalk-forward CV summary ({run_id}):")
                for k, (fv, fa) in enumerate(zip(cv_val_f1_scores, cv_val_acc_scores)):
                    print(f"  Fold {k + 1}: Val Acc={fa:.4f}  Val F1={fv:.4f}")
                print(f"  Mean Val F1: {np.mean(cv_val_f1_scores):.4f}  "
                      f"Std: {np.std(cv_val_f1_scores):.4f}")

                best_fold_idx    = int(np.argmax(cv_val_f1_scores))
                best_fold_result = fold_results[best_fold_idx]
                val_dataset      = fold_val_datasets[best_fold_idx]
                print(f"  Best fold: Fold {best_fold_idx + 1} "
                      f"(Val F1={best_fold_result['val_f1']:.4f})")

                model           = best_fold_result["model"]
                final_acc       = float(np.mean(cv_val_acc_scores))
                final_f1        = float(np.mean(cv_val_f1_scores))
                best_model_path = best_fold_result["best_model_path"]
                model_save_path = best_fold_result["model_save_path"]
                best_f1         = best_fold_result["best_f1"]
                duration        = sum(r["duration"] for r in fold_results)
                final_preds     = best_fold_result["val_preds"]
                final_labels    = best_fold_result["val_labels"]
                final_probs     = best_fold_result["val_probs"]

                print("\n" + "=" * 50)
                print(" " * 15 + "BEST FOLD MODEL")
                print("=" * 50)
                print(model)
                print("=" * 50 + "\n")

            else:
                # ---- Single inner split (original behaviour) ----
                result = _train_one_fold(
                    run_id, train_dataset, val_dataset,
                    cfg_run, cfg, device, run_models_dir, num_classes_run,
                    trainval_dataset=trainval_dataset,
                    run_plots_dir=run_plots_dir,
                )
                model           = result["model"]
                final_acc       = result["val_acc"]
                final_f1        = result["val_f1"]
                best_model_path = result["best_model_path"]
                model_save_path = result["model_save_path"]
                best_f1         = result["best_f1"]
                duration        = result["duration"]
                final_preds     = result["val_preds"]
                final_labels    = result["val_labels"]
                final_probs     = result["val_probs"]

                print("\n" + "=" * 50)
                print(" " * 15 + "MODEL ARCHITECTURE")
                print("=" * 50)
                print(model)
                print("=" * 50 + "\n")

            # Held-out final test evaluation (informational only; no decisions made here)
            print("Evaluating on held-out final test set...")
            ht_preds, ht_labels, ht_probs = test_model(model, final_test_loader)
            ht_acc = float(accuracy_score(ht_labels, ht_preds))
            ht_f1 = float(f1_score(ht_labels, ht_preds, average="weighted"))
            print(f"Final test set  | Acc: {ht_acc:.4f} | F1: {ht_f1:.4f}")

            print(f"Model state saved to {model_save_path}.")

            # Append results to CSV
            row = {
                "run_id": run_id,
                "model_type": model_type_name,
                "params": json.dumps(cfg_run),
                "val_accuracy": final_acc,
                "val_f1": final_f1,
                "final_test_accuracy": ht_acc,
                "final_test_f1": ht_f1,
                "model_path": str(model_save_path),
                "best_model_path": str(best_model_path) if best_model_path else "",
                "run_models_dir": str(run_models_dir),
                "run_plots_dir": str(run_plots_dir),
                "best_f1": best_f1 if best_f1 >= 0 else final_f1,
                "duration_seconds": round(duration, 3),
            }
            import csv
            # Schema check: if an existing grid_results.csv has a different header
            # (from before the model_type / run_models_dir / run_plots_dir columns
            # were added), rotate it aside and start fresh so column order stays
            # consistent for every row in the active file.
            if results_path.exists():
                try:
                    with results_path.open("r", newline="") as _f:
                        existing_header = next(csv.reader(_f), [])
                except Exception:
                    existing_header = []
                if existing_header != header:
                    backup = results_path.with_name(
                        f"{results_path.stem}_legacy_{int(time.time())}.csv"
                    )
                    results_path.rename(backup)
                    print(f"grid_results.csv header changed; archived old file to {backup}")

            write_header = not results_path.exists()
            with results_path.open("a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"Run {run_id} results saved to {results_path}")

            # Save detailed metrics and plots for this run
            _compute_and_save_detailed_metrics(
                run_id=run_id,
                labels=final_labels,
                preds=final_preds,
                probs=final_probs,
                num_classes=int(cfg_run.get("num_classes", cfg.get("num_classes", 5))),
                run_dir=run_models_dir,
                run_plots_dir=run_plots_dir,
            )

            # === Explainability: GNNExplainer (local subgraph) ===
            if bool(cfg_run.get("enable_explainability", cfg.get("enable_explainability", True))) and Explainer is not None and GNNExplainer is not None and ModelConfig is not None:
                print("Generating GNNExplainer local explanations...")
                # Take first snapshot from val_dataset (held-out test set is not used for explainability)
                try:
                    test_snapshot = val_dataset[0]
                    num_nodes = test_snapshot.x.shape[0]
                    # Feature names for readability in plots (must match dataset feature order)
                    # Real feature names attached by create_temporal_dataset — covers
                    # base + lag/roll/diff + Fourier features, so labels stay accurate
                    # even when add_lag_features is toggled.
                    feature_names = list(getattr(dataset, "feature_names", []) or [])
                    # Determine edge weight tensor attribute name across library versions
                    edge_weight_src = getattr(test_snapshot, 'edge_weight', None)
                    if edge_weight_src is None:
                        edge_weight_src = getattr(test_snapshot, 'edge_attr', None)
                    # Choose up to N nodes
                    explain_nodes = int(cfg_run.get("gnnexplainer_nodes", cfg.get("gnnexplainer_nodes", 5)))
                    node_indices = list(range(min(explain_nodes, num_nodes)))
                    explainer = Explainer(
                        model=_ExplainWrapper(model),
                        algorithm=GNNExplainer(epochs=int(cfg_run.get("gnnexplainer_epochs", cfg.get("gnnexplainer_epochs", 100)))),
                        explanation_type='model',
                        node_mask_type='attributes',
                        edge_mask_type='object',
                        # Use explicit multiclass classification mode to match torch_geometric 2.6.x
                        model_config=ModelConfig(mode='multiclass_classification', task_level='node', return_type='log_probs')
                    )
                    # Collect raw arrays across all explained nodes so
                    # explainability_report.py can produce model-level aggregates
                    # spanning every run of a given architecture.
                    _feat_masks_this_run: list[np.ndarray] = []
                    _edge_masks_this_run: list[np.ndarray] = []
                    for idx in node_indices:
                        explanation = explainer(
                            x=test_snapshot.x,
                            edge_index=test_snapshot.edge_index,
                            edge_weight=edge_weight_src,
                            index=idx
                        )
                        # Prefer node feature mask; fall back if needed
                        feat_mask = None
                        if hasattr(explanation, 'node_feat_mask') and explanation.node_feat_mask is not None:
                            feat_mask = explanation.node_feat_mask
                        elif hasattr(explanation, 'feature_mask') and explanation.feature_mask is not None:
                            feat_mask = explanation.feature_mask
                        elif hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
                            # Some versions expose a generic node_mask; use it as last resort
                            feat_mask = explanation.node_mask

                        edge_mask = getattr(explanation, 'edge_mask', None)

                        # Feature importance bar (only if mask available)
                        if feat_mask is not None:
                            mask_np = feat_mask.detach().cpu().numpy()
                            # If mask is multi-dim, reduce to per-feature (e.g., mean across nodes)
                            if mask_np.ndim > 1:
                                mask_np = mask_np.mean(axis=0)
                            _feat_masks_this_run.append(mask_np)
                            plt.figure(figsize=(10,4))
                            # Use feature names if available and lengths match
                            x_labels = feature_names if len(feature_names) == len(mask_np) else list(range(len(mask_np)))
                            sns.barplot(x=x_labels, y=mask_np)
                            plt.xlabel('Feature')
                            plt.ylabel('Importance')
                            plt.title(f'GNNExplainer Feature Importance (node {idx})')
                            if x_labels is feature_names:
                                plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            out_path = run_plots_dir / f"{run_id}_gnnexplainer_node_{idx}_feature_importance.png"
                            plt.savefig(out_path)
                            plt.close()

                        # Edge importance histogram
                        if edge_mask is not None:
                            edge_np = edge_mask.detach().cpu().numpy()
                            _edge_masks_this_run.append(edge_np)
                            plt.figure(figsize=(8,4))
                            sns.histplot(edge_np, bins=20)
                            plt.xlabel('Edge mask value')
                            plt.ylabel('Count')
                            plt.title(f'GNNExplainer Edge Mask Distribution (node {idx})')
                            plt.tight_layout()
                            out_path = run_plots_dir / f"{run_id}_gnnexplainer_node_{idx}_edge_mask_hist.png"
                            plt.savefig(out_path)
                            plt.close()

                    # Persist stacked raw arrays for cross-run aggregation.
                    # feature_masks: (num_explained_nodes, num_features)
                    # edge_masks:    (num_explained_nodes, num_edges) — may be ragged
                    #                if edge count varies per snapshot; save as
                    #                object array to preserve shape.
                    if _feat_masks_this_run:
                        try:
                            feat_stack = np.stack(_feat_masks_this_run, axis=0)
                        except ValueError:
                            # Ragged shapes fall back to object array so we don't crash.
                            feat_stack = np.array(_feat_masks_this_run, dtype=object)
                        edge_stack = (
                            np.array(_edge_masks_this_run, dtype=object)
                            if _edge_masks_this_run else np.array([], dtype=object)
                        )
                        np.savez(
                            run_models_dir / "explainability_gnnexplainer.npz",
                            feature_masks=feat_stack,
                            edge_masks=edge_stack,
                            feature_names=np.array(feature_names, dtype=object),
                        )
                except Exception as e:
                    print(f"GNNExplainer generation failed: {e}")

            # === Explainability: Integrated Gradients ===
            if bool(cfg_run.get("enable_explainability", cfg.get("enable_explainability", True))):
                print("Computing Integrated Gradients on node features...")
                try:
                    test_snapshot = val_dataset[0]
                    # Feature names for readability in plots (must match dataset feature order)
                    # Real feature names attached by create_temporal_dataset — covers
                    # base + lag/roll/diff + Fourier features, so labels stay accurate
                    # even when add_lag_features is toggled.
                    feature_names = list(getattr(dataset, "feature_names", []) or [])
                    # Determine edge weight tensor attribute name across library versions
                    edge_weight_src = getattr(test_snapshot, 'edge_weight', None)
                    if edge_weight_src is None:
                        edge_weight_src = getattr(test_snapshot, 'edge_attr', None)
                    # Choose first K nodes for speed
                    ig_nodes = int(cfg_run.get("ig_nodes", cfg.get("ig_nodes", 5)))
                    node_indices = list(range(min(ig_nodes, test_snapshot.x.shape[0])))
                    ig_steps = int(cfg_run.get("ig_steps", cfg.get("ig_steps", 50)))
                    num_classes_cfg = int(cfg_run.get("num_classes", cfg.get("num_classes", 5)))

                    # Compute IG for every class to ensure outputs exist for all classes
                    per_class_attr = {c: [] for c in range(num_classes_cfg)}
                    # Each interpolation step is a separate forward pass on the real
                    # [num_nodes, F] snapshot (see pipeline/integrated_gradients.py);
                    # captum's stacked-batch evaluation is not valid for these models.
                    ig_deltas: list[float] = []
                    for idx in node_indices:
                        for cls in range(num_classes_cfg):
                            attributions, ig_delta = integrated_gradients(
                                model,
                                test_snapshot.x,
                                test_snapshot.edge_index,
                                edge_weight_src,
                                node_idx=idx,
                                target_class=cls,
                                n_steps=ig_steps,
                            )
                            ig_deltas.append(abs(ig_delta))
                            attr_vec = attributions[idx].detach().cpu().numpy()
                            per_class_attr[cls].append(attr_vec)
                    if ig_deltas:
                        print(f"  IG completeness residual: mean |delta| = {np.mean(ig_deltas):.4f}, "
                              f"max = {np.max(ig_deltas):.4f}")

                    # Aggregate per-class attributions
                    all_class_arrays: list[np.ndarray] = []
                    for cls, vecs in per_class_attr.items():
                        arr = np.stack(vecs, axis=0)
                        all_class_arrays.append(arr)
                        mean_attr = arr.mean(axis=0)
                        # Save bar
                        plt.figure(figsize=(10,4))
                        x_labels = feature_names if len(feature_names) == mean_attr.shape[0] else list(range(mean_attr.shape[0]))
                        sns.barplot(x=x_labels, y=mean_attr)
                        plt.xlabel('Feature')
                        plt.ylabel('IG Attribution (mean)')
                        plt.title(f'Integrated Gradients (class {cls})')
                        if x_labels is feature_names:
                            plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        out_path = run_plots_dir / f"{run_id}_ig_class_{cls}_mean_attr.png"
                        plt.savefig(out_path)
                        plt.close()

                        # Save whisker plot across nodes
                        plt.figure(figsize=(6,4))
                        sns.boxplot(data=arr)
                        plt.xlabel('Feature')
                        plt.ylabel('IG Attribution')
                        plt.title(f'IG Distribution Across Nodes (class {cls})')
                        plt.tight_layout()
                        out_path = run_plots_dir / f"{run_id}_ig_class_{cls}_box.png"
                        plt.savefig(out_path)
                        plt.close()

                    # Persist the full (class, node, feature) tensor so the
                    # model-level aggregation module can compute averages that
                    # span both class and node axes.
                    if all_class_arrays:
                        ig_tensor = np.stack(all_class_arrays, axis=0)  # (C, N, F)
                        np.savez(
                            run_models_dir / "explainability_ig.npz",
                            attributions=ig_tensor,
                            feature_names=np.array(feature_names, dtype=object),
                        )
                except Exception as e:
                    print(f"Integrated Gradients failed: {e}")

            # Persist raw predictions so ROC curves can be aggregated across every
            # run of a given architecture (mean TPR at each FPR, with ±1 std band).
            #
            # NOTE: this file previously stored the VALIDATION predictions
            # (`final_labels`/`final_probs` are the selected fold's val output)
            # despite its name and its consumers treating it as the held-out test
            # set. It now stores the genuine held-out test predictions from
            # `test_model(model, final_test_loader)` above. The validation arrays
            # are kept alongside under explicit names so nothing is lost.
            try:
                np.savez(
                    run_models_dir / "test_probs.npz",
                    labels=np.asarray(ht_labels),
                    probs=np.asarray(ht_probs),
                    val_labels=np.asarray(final_labels),
                    val_probs=np.asarray(final_probs),
                )
            except Exception as e:
                print(f"Saving test_probs.npz failed: {e}")

        # --- Non-graph baselines, trained on the SAME data as the GNNs ---
        # Run inside the main pipeline (rather than as a separate script) so
        # the baselines inherit the identical dataset, splits, preprocessing
        # and t+1 label. That is what makes the cross-family comparison and
        # the paired McNemar tests valid without any re-alignment step.
        if bool(cfg.get("run_baselines", True)):
            try:
                from masters_research.pipeline.baselines_runner import run_baselines
                _seeds = cfg.get("seed", [42])
                _seeds = _seeds if isinstance(_seeds, list) else [_seeds]
                # Reproduce the inner split the grid runs used. It depends only
                # on trainval_dataset and train_ratio/stratified_split (not on
                # seed or model width), so it is identical for every grid run.
                _inner_ratio = float(cfg.get("train_ratio", 0.8))
                if bool(cfg.get("stratified_split", False)):
                    from masters_research.pipeline.dataset import stratified_temporal_signal_split
                    _bl_train, _bl_val = stratified_temporal_signal_split(
                        trainval_dataset, train_ratio=_inner_ratio,
                        num_classes=overall_num_classes,
                    )
                else:
                    _bl_train, _bl_val = temporal_signal_split(
                        trainval_dataset, train_ratio=_inner_ratio
                    )
                # Baselines build their run ids from cfg["run_name"], so the
                # fold tag has to be injected here too — otherwise their runs
                # are indistinguishable across folds and the pooling step in
                # outer_cv.py cannot pick them up.
                _bl_cfg = cfg
                if cv_fold is not None:
                    _bl_cfg = dict(cfg)
                    _bl_cfg["run_name"] = f"{cfg.get('run_name', 'run')}_f{cv_fold}"
                run_baselines(
                    cfg=_bl_cfg,
                    train_dataset=_bl_train,
                    val_dataset=_bl_val,
                    final_test_dataset=final_test_dataset,
                    save_dir=save_dir,
                    plots_dir=plots_dir,
                    results_path=results_path,
                    header=header,
                    seeds=[int(s) for s in _seeds],
                    num_classes=overall_num_classes,
                    detailed_metrics_fn=_compute_and_save_detailed_metrics,
                )
            except Exception as e:
                print(f"Baseline training skipped: {e}")

        # Cross-run aggregate reporting. Under outer CV each fold sees only its
        # own slice, so these are deferred and run once by outer_cv.py over the
        # pooled predictions instead of being rebuilt (and overwritten) per fold.
        if not run_reports:
            print("[CV fold %s] complete - aggregate reports deferred to the pooling step." % cv_fold)
            return

        # After all runs, generate summary plots
        _generate_summary_plots(results_path, plots_dir)

        try:
            training_curves.aggregate_curves(save_dir, plots_dir, run_prefix=cfg.get("run_name"))
        except Exception as e:
            print(f"Learning-curve aggregation skipped: {e}")

        # Emit the per-model best-run classification report and cross-model
        # comparison plots so results are ready to inspect without a follow-up command.
        try:
            from masters_research.pipeline.model_type_report import main as run_model_type_report
            run_model_type_report(base_path)
        except Exception as e:
            print(f"Per-model classification report generation skipped: {e}")

        # Per-class breakdown visuals (row-normalised CM, seed-variance boxplots,
        # cross-model per-class F1). Independent of model_type_report; if only one
        # architecture has runs so far, the per-model plots still get produced.
        try:
            from masters_research.pipeline.per_class_report import main as run_per_class_report
            run_per_class_report(base_path)
        except Exception as e:
            print(f"Per-class breakdown visuals skipped: {e}")

        # Model-level explainability aggregation — reads the raw feature-mask /
        # edge-mask / IG-attribution / test-probs npz files saved by each run
        # and produces one canonical set of GNNExplainer + IG + ROC plots per
        # architecture at models/<ModelType>/explainability/. Only produces
        # outputs for architectures whose runs actually saved the arrays;
        # legacy runs (no npz) are silently skipped.
        try:
            from masters_research.pipeline.explainability_report import main as run_explainability_report
            run_explainability_report(base_path)
        except Exception as e:
            print(f"Model-level explainability aggregation skipped: {e}")

        # Error analysis of the held-out predictions (single split; the rotating
        # cross-validation runs it once over the pooled folds in outer_cv.py).
        if cv_fold is None:
            try:
                from masters_research.pipeline.error_analysis import run_error_analysis
                run_error_analysis(base_path, cfg, Path(save_dir), cfg.get("run_name", "run"), 1)
            except Exception as e:
                print(f"Error analysis skipped: {e}")

    except Exception:
        import traceback

        print("An error occurred during the training process:")
        traceback.print_exc()
