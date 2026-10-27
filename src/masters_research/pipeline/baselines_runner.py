"""Non-graph baselines trained inside the main GNN run.

Historically the Random Forest and Keras MLP baselines lived in
``baselines/baseline_models.py``, which built its own dataset straight from
the raw station CSVs. That gave the two model families *different targets*:
the baselines forecast terciles of a mean over RAW features (including the
constant Lat/Lon/Alt columns), while the GNNs forecast terciles of a mean over
SEASONALLY-NORMALISED weather + lag features. Both were one-month-ahead, but
they were not forecasting the same quantity, so any head-to-head comparison
was confounded.

This module removes that confound. It consumes the *same*
``DynamicGraphTemporalSignalBatch`` the GNNs consume, flattens each snapshot
into tabular ``(station-month, feature)`` rows, and trains on the *same*
train/val/test partitions. Every baseline therefore sees:

  * the same 21-dimensional feature vector,
  * the same seasonally-normalised, trainval-fitted preprocessing,
  * the same calendar-matched t+1 label,
  * the same evaluation samples.

That last property is what makes the paired McNemar tests in
``mcnemar_report.py`` valid without any re-alignment step.

The only thing the baselines do *not* get is the graph: each row is
classified independently, with no access to neighbouring stations'
contemporaneous features. That is the intended contrast — the gap between
the two families isolates the value of the spatial inductive bias.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from masters_research.pipeline import training_curves

BASELINE_MODEL_TYPES = ["RandomForest", "KerasMLP"]


# ---------------------------------------------------------------------------
# Snapshot flattening
# ---------------------------------------------------------------------------

def flatten_snapshots(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every snapshot's node-feature block into tabular rows.

    Returns (X, y) where each row is one (station, month) observation and the
    label is that station's t+1 drought category. Node identity and graph
    structure are deliberately discarded — that is what makes these baselines
    non-graph models.
    """
    xs, ys = [], []
    n = getattr(dataset, "snapshot_count", None) or len(dataset)
    for i in range(n):
        snap = dataset[i]
        xs.append(snap.x.detach().cpu().numpy())
        ys.append(snap.y.detach().cpu().numpy().astype(int))
    X, y = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    # Nodes carrying the -1 sentinel have no drought index for the target
    # month. The GNNs keep them in the graph and mask the loss; a tabular
    # model has no such need, so they are dropped outright. Both families are
    # therefore fit and scored on exactly the same labelled observations.
    keep = y >= 0
    return X[keep], y[keep]


# ---------------------------------------------------------------------------
# Individual baseline fits
# ---------------------------------------------------------------------------

def _fit_random_forest(X_tr, y_tr, seed, cfg, X_val=None, y_val=None, history=None):
    rf = RandomForestClassifier(
        n_estimators=int(cfg.get("rf_n_estimators", 200)),
        max_depth=cfg.get("rf_max_depth", None),
        min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 2)),
        max_features=cfg.get("rf_max_features", "sqrt"),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    if history is not None and X_val is not None:
        history.extend(training_curves.random_forest_history(rf, X_tr, y_tr, X_val, y_val))
    return rf, lambda m, X: (m.predict(X).astype(int), m.predict_proba(X))


def _fit_keras_mlp(X_tr, y_tr, seed, cfg, num_classes, X_val=None, y_val=None, history=None):
    """Lazy-import Keras so a missing TensorFlow install only disables this
    one baseline rather than breaking the whole training run."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical

    tf.keras.utils.set_random_seed(seed)
    model = Sequential([
        Input(shape=(X_tr.shape[1],)),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=float(cfg.get("mlp_learning_rate", 0.003))),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(
        X_tr, to_categorical(y_tr, num_classes=num_classes),
        epochs=int(cfg.get("mlp_epochs", 100)),
        batch_size=int(cfg.get("mlp_batch_size", 64)),
        # No validation_split: Keras would silently hold back the last 10% of the
        # training rows, so the MLP would train on less data than the RF and GNNs.
        verbose=0,
        # Learning-curve logging only; validation never feeds back into training.
        callbacks=([training_curves.keras_history_callback(X_tr, y_tr, X_val, y_val, history)]
                   if history is not None and X_val is not None else None),
    )

    def predict(m, X):
        probs = m.predict(X, verbose=0)
        return np.argmax(probs, axis=1).astype(int), probs

    return model, predict


# ---------------------------------------------------------------------------
# Entry point called from pipeline/train.py
# ---------------------------------------------------------------------------

def run_baselines(
    cfg: dict,
    train_dataset,
    val_dataset,
    final_test_dataset,
    save_dir: Path,
    plots_dir: Path,
    results_path: Path,
    header: list[str],
    seeds: list[int],
    num_classes: int,
    detailed_metrics_fn,
) -> None:
    """Train each non-graph baseline once per seed on the GNN's own splits.

    Writes the same artefacts a GNN run writes — a grid_results.csv row, the
    per-run detailed metrics, and a test_probs.npz — so every downstream
    report (model_type_report, per_class_report, model_comparison,
    mcnemar_report) picks the baselines up with no special-casing.
    """
    X_tr, y_tr = flatten_snapshots(train_dataset)
    X_val, y_val = flatten_snapshots(val_dataset)
    X_test, y_test = flatten_snapshots(final_test_dataset)

    print("\n" + "=" * 60)
    print(" " * 16 + "NON-GRAPH BASELINES")
    print("=" * 60)
    print(f"Tabular shapes  train={X_tr.shape}  val={X_val.shape}  test={X_test.shape}")
    print("Same features, same splits, same t+1 label as the GNN runs above.")

    record_curves = bool(cfg.get("record_training_curves", True))
    builders = {
        "RandomForest": lambda s, h: _fit_random_forest(X_tr, y_tr, s, cfg, X_val, y_val, h),
        "KerasMLP": lambda s, h: _fit_keras_mlp(X_tr, y_tr, s, cfg, num_classes, X_val, y_val, h),
    }

    for model_type in BASELINE_MODEL_TYPES:
        for seed in seeds:
            run_id = f"{cfg.get('run_name', 'run')}_{model_type.lower()}_{seed}_{int(time.time())}"
            run_models_dir = save_dir / model_type / run_id
            run_plots_dir = plots_dir / model_type / run_id
            run_models_dir.mkdir(parents=True, exist_ok=True)
            run_plots_dir.mkdir(parents=True, exist_ok=True)

            start = time.time()
            history = [] if record_curves else None
            try:
                model, predict = builders[model_type](seed, history)
            except Exception as e:
                print(f"  [{model_type} seed={seed}] SKIPPED: {e}")
                continue

            val_preds, val_probs = predict(model, X_val)
            test_preds, test_probs = predict(model, X_test)
            duration = time.time() - start

            val_acc = float(accuracy_score(y_val, val_preds))
            val_f1 = float(f1_score(y_val, val_preds, average="weighted"))
            test_acc = float(accuracy_score(y_test, test_preds))
            test_f1 = float(f1_score(y_test, test_preds, average="weighted"))
            print(f"  [{model_type:13s} seed={seed:5d}] "
                  f"val_f1={val_f1:.4f}  test_f1={test_f1:.4f}  ({duration:.1f}s)")

            if history:
                training_curves.save_run_history(
                    history, run_models_dir, run_plots_dir, title=f"{model_type} - {run_id}",
                    x_label="Trees" if model_type == "RandomForest" else "Epoch",
                )

            # Same per-run artefacts a GNN run produces.
            detailed_metrics_fn(
                run_id=run_id, labels=y_val, preds=val_preds, probs=val_probs,
                num_classes=num_classes, run_dir=run_models_dir,
                run_plots_dir=run_plots_dir,
            )
            # Held-out TEST predictions, matching the GNN runs. This previously
            # stored the validation arrays despite the filename, which made every
            # downstream consumer (ROC aggregation, McNemar) compare models on
            # validation data. Validation is retained under explicit names.
            try:
                np.savez(run_models_dir / "test_probs.npz",
                         labels=np.asarray(y_test), probs=np.asarray(test_probs),
                         val_labels=np.asarray(y_val), val_probs=np.asarray(val_probs))
            except Exception as e:
                print(f"    saving test_probs.npz failed: {e}")

            row = {
                "run_id": run_id,
                "model_type": model_type,
                "params": json.dumps({"seed": seed, "model_type": model_type,
                                      "family": "baseline"}),
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "final_test_accuracy": test_acc,
                "final_test_f1": test_f1,
                "model_path": "",
                "best_model_path": "",
                "run_models_dir": str(run_models_dir),
                "run_plots_dir": str(run_plots_dir),
                "best_f1": val_f1,
                "duration_seconds": round(duration, 3),
            }
            write_header = not results_path.exists()
            with results_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    print("=" * 60)
