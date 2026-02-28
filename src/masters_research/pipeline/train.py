import json
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
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
try:
    from captum.attr import IntegratedGradients
except Exception:
    IntegratedGradients = None

from masters_research.data import TemporalDataLoader
from masters_research.pipeline.dataset import create_temporal_dataset
from masters_research.models import TemporalSpatialNodeClassifier
from masters_research.pipeline.temporal_quality import run_temporal_quality_audit, compare_splits
import seaborn as sns


class FocalLoss(nn.Module):
    """Multi-class Focal Loss: down-weights easy examples so the model focuses on hard ones.
    Useful when temporal distribution shift causes the test classes to differ from training.
    gamma=0 reduces to standard cross-entropy.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


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


def train_epoch(model, optimizer, criterion, train_loader, grad_clip=None):
    model.train()
    total_loss = 0
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
        loss = criterion(out, targets_long)
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += loss.item()
    return total_loss / loader_len if loader_len > 0 else total_loss


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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(snapshot.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
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
    counts = np.bincount(labels, minlength=num_classes)
    total = labels.size

    print("\n" + "=" * 50)
    print(f"CLASS BALANCE - {name}")
    print("=" * 50)
    print(f"Total labeled samples: {total}")
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

def _compute_and_save_detailed_metrics(run_id: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, num_classes: int, save_dir: Path, plots_dir: Path):
    classes = list(range(num_classes))
    # Core metrics
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision, recall, f1_per_class, support = precision_recall_fscore_support(labels, preds, labels=classes, average=None, zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro')

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=classes)
    cm_path_img = plots_dir / f"{run_id}_confusion_matrix.png"
    cm_path_csv = save_dir / f"{run_id}_confusion_matrix.csv"
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
    per_class_csv = save_dir / f"{run_id}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # Per-class bar chart
    per_class_plot = plots_dir / f"{run_id}_per_class_f1.png"
    _save_per_class_bars(f1_per_class, classes, per_class_plot)

    # Per-run box-and-whisker plot of F1 across classes
    per_run_box_plot = plots_dir / f"{run_id}_per_class_f1_box.png"
    _save_per_run_boxplot(f1_per_class, per_run_box_plot)

    # ROC / AUC (One-vs-Rest) if possible
    roc_auc_macro = None
    try:
        y_true_bin = label_binarize(labels, classes=classes)
        roc_auc_macro = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
        roc_combined = plots_dir / f"{run_id}_roc_curves.png"
        _save_roc_curves(labels, probs, classes, roc_combined, plots_dir)
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
    summary_csv = save_dir / f"{run_id}_summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"Saved detailed metrics for {run_id} to {save_dir} and plots to {plots_dir}")


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
        # Collect per-class metrics from all runs
        models_dir = plots_dir.parent / 'models' if plots_dir.parent.name == 'artifacts' else plots_dir.parent
        per_class_files = list(models_dir.glob('*_per_class_metrics.csv'))
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


def _train_one_fold(
    fold_label: str,
    train_dataset,
    val_dataset,
    cfg_run: dict,
    cfg: dict,
    device,
    save_dir: Path,
    num_classes_run: int,
) -> dict:
    """
    Create a fresh model, run the training loop with early stopping, reload the best
    checkpoint, evaluate on val_dataset, save model state, and return a results dict.

    Returned dict keys:
        model, best_model_path, model_save_path, best_f1,
        val_acc, val_f1, val_preds, val_labels, val_probs, duration.
    """
    train_loader = TemporalDataLoader(train_dataset)
    val_loader   = TemporalDataLoader(val_dataset)

    # --- Model ---
    use_attention = bool(cfg_run.get("use_attention", cfg.get("use_attention", False)))
    use_temporal_attention = bool(cfg_run.get("use_temporal_attention", cfg.get("use_temporal_attention", False)))
    _gru_layers   = int(cfg_run.get("gru_layers",    cfg.get("gru_layers",    1)))
    _input_norm   = bool(cfg_run.get("input_norm",   cfg.get("input_norm",    True)))
    _in_channels  = int(cfg_run.get("in_channels", cfg.get("in_channels", 10)))
    _hidden_channels = int(cfg_run.get("hidden_channels", cfg.get("hidden_channels", 16)))
    _dropout = float(cfg_run.get("dropout", cfg.get("dropout", 0.3)))
    
    if use_temporal_attention:
        # New temporal attention model with history buffer
        from masters_research.models.temporal_attention_gnn import TemporalAttentionGNN
        _history_len = int(cfg_run.get("temporal_history_len", cfg.get("temporal_history_len", 10)))
        _temporal_heads = int(cfg_run.get("temporal_heads", cfg.get("temporal_heads", 4)))
        print(f"  [{fold_label}] Using TemporalAttentionGNN (history_len={_history_len}, heads={_temporal_heads}, dropout={_dropout})")
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

    # --- Class weights (inverse-frequency from training snapshots) ---
    _train_lbls = np.concatenate([
        train_dataset[i].y.cpu().numpy().astype(int)
        for i in range(getattr(train_dataset, "snapshot_count", None) or len(train_dataset))
    ])
    _counts = np.bincount(_train_lbls, minlength=num_classes_run).astype(float)
    _w = 1.0 / np.maximum(_counts, 1)
    _w = (_w / _w.sum()) * num_classes_run
    class_weights = torch.tensor(_w, dtype=torch.float).to(device)
    print(f"  [{fold_label}] Class weights: {[f'{v:.3f}' for v in _w]}")

    # --- Loss function ---
    use_focal   = bool(cfg_run.get("focal_loss",  cfg.get("focal_loss",  True)))
    focal_gamma = float(cfg_run.get("focal_gamma", cfg.get("focal_gamma", 2.0)))
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        print(f"  [{fold_label}] Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  [{fold_label}] Using weighted CrossEntropyLoss")

    # --- Scheduler ---
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

    # --- Training loop ---
    print(f"  [{fold_label}] Starting training...")
    start_time      = time.time()
    best_f1         = -1.0
    best_model_path = None
    eval_interval   = int(cfg_run.get("eval_interval",         cfg.get("eval_interval",         1)))
    min_delta       = float(cfg_run.get("early_stopping_min_delta", cfg.get("early_stopping_min_delta", 0.0)))
    patience        = int(cfg_run.get("early_stopping_patience",    cfg.get("early_stopping_patience",    0)))
    patience_counter = 0
    grad_clip       = cfg_run.get("grad_clip", cfg.get("grad_clip", None))
    last_val_f1     = -1.0
    num_epochs      = int(cfg_run.get("epochs", cfg.get("epochs", 200)))

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, optimizer, criterion, train_loader, grad_clip=grad_clip)
        do_eval = (epoch % eval_interval == 0) or (epoch == num_epochs)
        if do_eval:
            preds, labels, probs = test_model(model, val_loader)
            acc = accuracy_score(labels, preds)
            f1  = f1_score(labels, preds, average="weighted")
            print(f"  [{fold_label}] Epoch {epoch:03d}/{num_epochs} | "
                  f"Loss: {loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
            last_val_f1 = f1
            if f1 > best_f1 + min_delta:
                best_f1          = f1
                patience_counter = 0
                if cfg_run.get("save_best", cfg.get("save_best", True)):
                    best_model_path = save_dir / f"best_{fold_label}.pth"
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

    duration = time.time() - start_time
    print(f"  [{fold_label}] Training finished in {duration:.1f}s  (best Val F1: {best_f1:.4f}).")

    # Reload best checkpoint so the returned model has the best weights
    if best_model_path and Path(best_model_path).exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # Final val evaluation
    val_preds, val_labels, val_probs = test_model(model, val_loader)
    val_acc = float(accuracy_score(val_labels, val_preds))
    val_f1  = float(f1_score(val_labels, val_preds, average="weighted"))

    # Persist full model state dict
    model_save_path = save_dir / f"temporal_spatial_classifier_state_{fold_label}.pth"
    torch.save(model.state_dict(), model_save_path)

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
    }


def main(base_path: Path):
    """
    Main function to run the training and evaluation pipeline.
    Configuration is loaded from configs/default.json in the project root.
    """
    base_path = Path(base_path).resolve()
    config_path = base_path / "configs" / "default.json"
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
            "params",
            "val_accuracy",
            "val_f1",
            "final_test_accuracy",
            "final_test_f1",
            "model_path",
            "best_model_path",
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
            nodes_per_timestamp.plot(kind="bar")
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
        )
        
        # Dynamically determine in_channels from actual data
        actual_in_channels = dataset.features[0].shape[1]
        print(f"\n[DYNAMIC] Detected in_channels = {actual_in_channels} from data")
        if cfg.get("in_channels") != actual_in_channels:
            print(f"  Updating in_channels from config value {cfg.get('in_channels')} → {actual_in_channels}")
            cfg["in_channels"] = actual_in_channels
        
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
                dataset, train_ratio=outer_train_ratio
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
        for run_idx, cfg_run in enumerate(grid, start=1):
            run_prefix = cfg.get("run_name", "run")
            run_id = f"{run_prefix}{run_idx}_{int(time.time())}"
            print(f"\n=== Starting grid run {run_idx}/{len(grid)} ({run_id}) ===")

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
                        plots_dir=plots_dir,
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
                        cfg_run, cfg, device, save_dir, num_classes_run,
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
                    cfg_run, cfg, device, save_dir, num_classes_run,
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
                "params": json.dumps(cfg_run),
                "val_accuracy": final_acc,
                "val_f1": final_f1,
                "final_test_accuracy": ht_acc,
                "final_test_f1": ht_f1,
                "model_path": str(model_save_path),
                "best_model_path": str(best_model_path) if best_model_path else "",
                "best_f1": best_f1 if best_f1 >= 0 else final_f1,
                "duration_seconds": round(duration, 3),
            }
            write_header = not results_path.exists()
            with results_path.open("a", newline="") as csvfile:
                import csv

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
                save_dir=save_dir,
                plots_dir=plots_dir,
            )

            # === Explainability: GNNExplainer (local subgraph) ===
            if bool(cfg_run.get("enable_explainability", cfg.get("enable_explainability", True))) and Explainer is not None and GNNExplainer is not None and ModelConfig is not None:
                print("Generating GNNExplainer local explanations...")
                # Take first snapshot from val_dataset (held-out test set is not used for explainability)
                try:
                    test_snapshot = val_dataset[0]
                    num_nodes = test_snapshot.x.shape[0]
                    # Feature names for readability in plots (must match dataset feature order)
                    feature_names = [
                        "AverageMaximumTemperature_x",
                        "AverageMinimunTemperature_x",
                        "AverageMaximumRelativeHumidity_x",
                        "AverageMinimumRelativeHumidity_x",
                        "TotalRainfall_x",
                        "AverageTotalRadiation_x",
                        "AverageWindSpeed_x",
                        "AverageTotalRelativeEvaporation_x",
                        "month_sin",
                        "month_cos",
                    ]
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
                            out_path = plots_dir / f"{run_id}_gnnexplainer_node_{idx}_feature_importance.png"
                            plt.savefig(out_path)
                            plt.close()

                        # Edge importance histogram
                        if edge_mask is not None:
                            edge_np = edge_mask.detach().cpu().numpy()
                            plt.figure(figsize=(8,4))
                            sns.histplot(edge_np, bins=20)
                            plt.xlabel('Edge mask value')
                            plt.ylabel('Count')
                            plt.title(f'GNNExplainer Edge Mask Distribution (node {idx})')
                            plt.tight_layout()
                            out_path = plots_dir / f"{run_id}_gnnexplainer_node_{idx}_edge_mask_hist.png"
                            plt.savefig(out_path)
                            plt.close()
                except Exception as e:
                    print(f"GNNExplainer generation failed: {e}")

            # === Explainability: Captum Integrated Gradients ===
            if bool(cfg_run.get("enable_explainability", cfg.get("enable_explainability", True))) and IntegratedGradients is not None:
                print("Computing Integrated Gradients on node features...")
                try:
                    test_snapshot = val_dataset[0]
                    # Feature names for readability in plots (must match dataset feature order)
                    feature_names = [
                        "AverageMaximumTemperature_x",
                        "AverageMinimunTemperature_x",
                        "AverageMaximumRelativeHumidity_x",
                        "AverageMinimumRelativeHumidity_x",
                        "TotalRainfall_x",
                        "AverageTotalRadiation_x",
                        "AverageWindSpeed_x",
                        "AverageTotalRelativeEvaporation_x",
                        "month_sin",
                        "month_cos",
                    ]
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
                    for idx in node_indices:
                        def forward_node(x):
                            result = model(x, test_snapshot.edge_index, edge_weight_src)
                            logits = result[0] if isinstance(result, tuple) else result
                            return logits[idx].unsqueeze(0)

                        ig = IntegratedGradients(forward_node)
                        for cls in range(num_classes_cfg):
                            attributions = ig.attribute(
                                inputs=test_snapshot.x.clone().detach().requires_grad_(True),
                                target=cls,
                                n_steps=ig_steps
                            )
                            attr_vec = attributions[idx].detach().cpu().numpy()
                            per_class_attr[cls].append(attr_vec)

                    # Aggregate per-class attributions
                    for cls, vecs in per_class_attr.items():
                        arr = np.stack(vecs, axis=0)
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
                        out_path = plots_dir / f"{run_id}_ig_class_{cls}_mean_attr.png"
                        plt.savefig(out_path)
                        plt.close()

                        # Save whisker plot across nodes
                        plt.figure(figsize=(6,4))
                        sns.boxplot(data=arr)
                        plt.xlabel('Feature')
                        plt.ylabel('IG Attribution')
                        plt.title(f'IG Distribution Across Nodes (class {cls})')
                        plt.tight_layout()
                        out_path = plots_dir / f"{run_id}_ig_class_{cls}_box.png"
                        plt.savefig(out_path)
                        plt.close()
                except Exception as e:
                    print(f"Integrated Gradients failed: {e}")

        # After all runs, generate summary plots
        _generate_summary_plots(results_path, plots_dir)

    except Exception:
        import traceback

        print("An error occurred during the training process:")
        traceback.print_exc()
