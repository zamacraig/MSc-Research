import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread


def _parse_classification_report_text(path: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parse a sklearn classification_report text file into per-class DataFrame and summary dict.

    Returns (class_df, summary) where class_df has index as class labels (str) and columns
    ['precision','recall','f1-score','support']; summary contains keys 'accuracy',
    'macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1'.
    """
    text = path.read_text().strip().splitlines()
    class_rows = []
    summary = {
        'accuracy': np.nan,
        'macro_precision': np.nan,
        'macro_recall': np.nan,
        'macro_f1': np.nan,
        'weighted_precision': np.nan,
        'weighted_recall': np.nan,
        'weighted_f1': np.nan,
    }

    # Regex patterns
    row_pat = re.compile(r"^\s*(\S+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$")
    acc_pat = re.compile(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$")
    avg_pat = re.compile(r"^\s*(macro avg|weighted avg)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)\s*$")

    for line in text:
        m = row_pat.match(line)
        if m:
            lbl, prec, rec, f1, sup = m.groups()
            class_rows.append({
                'label': lbl,
                'precision': float(prec),
                'recall': float(rec),
                'f1-score': float(f1),
                'support': int(sup),
            })
            continue
        m_acc = acc_pat.match(line)
        if m_acc:
            acc, sup = m_acc.groups()
            summary['accuracy'] = float(acc)
            continue
        m_avg = avg_pat.match(line)
        if m_avg:
            which, prec, rec, f1, sup = m_avg.groups()
            if which == 'macro avg':
                summary['macro_precision'] = float(prec)
                summary['macro_recall'] = float(rec)
                summary['macro_f1'] = float(f1)
            else:
                summary['weighted_precision'] = float(prec)
                summary['weighted_recall'] = float(rec)
                summary['weighted_f1'] = float(f1)

    class_df = pd.DataFrame(class_rows)
    if not class_df.empty:
        class_df = class_df.set_index('label')
    return class_df, summary


def _find_latest_best_gnn_report(models_dir: Path) -> Optional[Path]:
    candidates = sorted(models_dir.glob("*_classification_report_best.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_confusion_pngs(models_dir: Path) -> Dict[str, Optional[Path]]:
    paths = {
        'GNN': None,
        'RandomForest': models_dir / 'RandomForest_confusion_matrix.png',
        'Keras': models_dir / 'Keras_confusion_matrix.png',
    }
    # GNN best
    gnn_pngs = sorted(models_dir.glob("*_confusion_matrix_best.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if gnn_pngs:
        paths['GNN'] = gnn_pngs[0]
    for k in ['RandomForest', 'Keras']:
        if not paths[k].exists():
            paths[k] = None
    return paths


def _grouped_bars_per_class(class_metrics: Dict[str, pd.DataFrame], metric: str, out_path: Path) -> None:
    # Union of class labels across models
    labels = sorted(set().union(*(df.index.tolist() for df in class_metrics.values() if df is not None and not df.empty)))
    if not labels:
        return
    plot_df = []
    for model_name, df in class_metrics.items():
        if df is None or df.empty:
            continue
        for lbl in labels:
            val = df.loc[lbl, metric] if lbl in df.index else np.nan
            plot_df.append({'class': lbl, 'model': model_name, 'value': val})
    plot_df = pd.DataFrame(plot_df)

    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    sns.barplot(data=plot_df, x='class', y='value', hue='model')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Class')
    plt.title(f'Per-class {metric} by Model')
    plt.legend(title='Model')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _radar_chart_f1(class_metrics: Dict[str, pd.DataFrame], out_path: Path) -> None:
    labels = sorted(set().union(*(df.index.tolist() for df in class_metrics.values() if df is not None and not df.empty)))
    if not labels:
        return
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for model_name, df in class_metrics.items():
        if df is None or df.empty:
            continue
        vals = [df.loc[lbl, 'f1-score'] if lbl in df.index else np.nan for lbl in labels]
        vals = np.array(vals)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, label=model_name)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title('Radar Chart: Per-class F1 by Model')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _averages_comparison(summary_metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    rows = []
    for model_name, s in summary_metrics.items():
        if not s:
            continue
        rows.append({'model': model_name, 'macro': s.get('macro_f1', np.nan), 'micro': s.get('accuracy', np.nan), 'weighted': s.get('weighted_f1', np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df_melt = df.melt(id_vars='model', value_vars=['macro', 'micro', 'weighted'], var_name='average', value_name='value')

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melt, x='average', y='value', hue='model')
    plt.ylabel('Score')
    plt.xlabel('Average Type')
    plt.ylim(0, 1)
    plt.title('Macro/Micro/Weighted Averages by Model')
    plt.legend(title='Model')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _confusion_side_by_side(conf_pngs: Dict[str, Optional[Path]], out_path: Path) -> None:
    keys = ['GNN', 'RandomForest', 'Keras']
    imgs = [(k, imread(str(conf_pngs[k]))) for k in keys if conf_pngs.get(k) is not None]
    if not imgs:
        return
    cols = len(imgs)
    plt.figure(figsize=(6 * cols, 5))
    for i, (name, img) in enumerate(imgs, start=1):
        ax = plt.subplot(1, cols, i)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{name} Confusion Matrix')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _stacked_recall_per_class(class_metrics: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """Stack recall values per class across models (GNN, RF, Keras)."""
    labels = sorted(set().union(*(df.index.tolist() for df in class_metrics.values() if df is not None and not df.empty)))
    if not labels:
        return

    # Build recall per model per class
    rec_df = pd.DataFrame({'class': labels})
    for model_name, df in class_metrics.items():
        if df is None or df.empty:
            rec_df[model_name] = np.nan
        else:
            rec_df[model_name] = [df.loc[lbl, 'recall'] if lbl in df.index else np.nan for lbl in labels]

    # Plot stacked bars
    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    ax = plt.gca()
    x = np.arange(len(labels))
    width = 0.8
    bottom = np.zeros(len(labels))
    colors = {
        'GNN': '#1f77b4',        # blue
        'RandomForest': '#2ca02c',# green
        'Keras': '#ff7f0e',       # orange
    }
    legend_handles = []
    for model_name in ['GNN', 'RandomForest', 'Keras']:
        if model_name in rec_df.columns:
            vals = rec_df[model_name].fillna(0.0).to_numpy()
            bar = ax.bar(x, vals, width, bottom=bottom, label=model_name, color=colors.get(model_name))
            legend_handles.append(bar)
            bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Recall (stacked)')
    ax.set_xlabel('Class')
    ax.set_title('Stacked Recall per Class Across Models')
    ax.legend(title='Model')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main(base_path: Path | str) -> Path:
    base_path = Path(base_path).resolve()
    models_dir = base_path / 'src' / 'masters_research' / 'artifacts' / 'models'
    plots_dir = base_path / 'src' / 'masters_research' / 'artifacts' / 'plots' / 'comparison'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Locate classification reports
    rf_txt = models_dir / 'RandomForest_classification_report.txt'
    keras_txt = models_dir / 'Keras_classification_report.txt'
    gnn_txt = _find_latest_best_gnn_report(models_dir)

    class_metrics: Dict[str, pd.DataFrame] = {'GNN': None, 'RandomForest': None, 'Keras': None}
    summary_metrics: Dict[str, Dict[str, float]] = {'GNN': {}, 'RandomForest': {}, 'Keras': {}}

    if rf_txt.exists():
        class_metrics['RandomForest'], summary_metrics['RandomForest'] = _parse_classification_report_text(rf_txt)
    if keras_txt.exists():
        class_metrics['Keras'], summary_metrics['Keras'] = _parse_classification_report_text(keras_txt)
    if gnn_txt and gnn_txt.exists():
        class_metrics['GNN'], summary_metrics['GNN'] = _parse_classification_report_text(gnn_txt)

    # Plots: grouped bars for precision/recall/f1
    _grouped_bars_per_class(class_metrics, 'precision', plots_dir / 'per_class_precision_by_model.png')
    _grouped_bars_per_class(class_metrics, 'recall', plots_dir / 'per_class_recall_by_model.png')
    _grouped_bars_per_class(class_metrics, 'f1-score', plots_dir / 'per_class_f1_by_model.png')

    # Radar chart (F1)
    _radar_chart_f1(class_metrics, plots_dir / 'radar_f1_by_model.png')

    # Macro/micro/weighted averages comparison
    _averages_comparison(summary_metrics, plots_dir / 'averages_comparison.png')

    # Stacked recall per class
    _stacked_recall_per_class(class_metrics, plots_dir / 'stacked_recall_per_class.png')

    # Confusion matrices side-by-side
    conf_pngs = _find_confusion_pngs(models_dir)
    _confusion_side_by_side(conf_pngs, plots_dir / 'confusion_matrices_side_by_side.png')

    print(f"Saved comparison plots to: {plots_dir}")
    return plots_dir


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[3]
    main(project_root)
