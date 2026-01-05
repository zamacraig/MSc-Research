import json
from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data_loader import load_and_preprocess_data
from train import create_temporal_dataset
from torch_geometric_temporal.signal import temporal_signal_split
from TemporalDataLoaderClass import TemporalDataLoader
from model import TemporalSpatialNodeClassifier


def main():
    base = Path(__file__).parent
    results_path = base / 'grid_results.csv'
    if not results_path.exists():
        print(f'No grid results found at {results_path}. Run grid search first.')
        return

    df = pd.read_csv(results_path)
    if df.empty:
        print('grid_results.csv is empty')
        return

    # Sort by F1 score (descending) and accuracy as tiebreaker
    df_sorted = df.sort_values(by=['f1', 'accuracy'], ascending=[False, False]).reset_index(drop=True)

    # Save a simple summary (top 5)
    summary_path = base / 'grid_summary.txt'
    with summary_path.open('w') as fh:
        fh.write('Top 5 grid runs (sorted by f1 then accuracy)\n')
        fh.write('-' * 60 + '\n')
        for i, row in df_sorted.head(5).iterrows():
            fh.write(f"Rank {i+1}: run_id={row['run_id']}, f1={row['f1']:.4f}, acc={row['accuracy']:.4f}\n")
            try:
                params = json.loads(row['params'])
            except Exception:
                params = row['params']
            fh.write(f"  params: {json.dumps(params)}\n\n")
    print(f'Wrote grid summary to {summary_path}')

    # Pick best run (first row)
    best = df_sorted.iloc[0]
    try:
        best_params = json.loads(best['params'])
    except Exception:
        best_params = {}
    best_model_path = Path(best['model_path'])
    if not best_model_path.exists():
        # try relative to pipeline dir
        candidate = base / best_model_path.name
        if candidate.exists():
            best_model_path = candidate
        else:
            print(f"Model file {best['model_path']} not found.")
            return

    print(f"Best run: {best['run_id']} with f1={best['f1']}, model={best_model_path}")

    # Recreate dataset and split according to best params
    # Load data (same path used in training)
    attribute_df, node_df = load_and_preprocess_data(base_path="../..")
    final_df = pd.read_csv('../../../Dataset/TransientData/corrected1_df.csv')
    dataset = create_temporal_dataset(final_df)

    train_ratio = float(best_params.get('train_ratio', 0.8))
    _, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
    test_loader = TemporalDataLoader(test_dataset)

    # Instantiate model and load state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = int(best_params.get('in_channels', 8))
    hidden_ch = int(best_params.get('hidden_channels', 16))
    num_cls = int(best_params.get('num_classes', 5))

    model = TemporalSpatialNodeClassifier(in_channels=in_ch, hidden_channels=hidden_ch, num_classes=num_cls).to(device)
    state = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for snapshot in test_loader:
            out = model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            preds = out.argmax(dim=1).cpu().numpy()
            labels = snapshot.y.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    text_report = classification_report(all_labels, all_preds, digits=4)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    report_json_path = base / f"best_run_{best['run_id']}_report.json"
    report_txt_path = base / f"best_run_{best['run_id']}_report.txt"

    with report_json_path.open('w') as fh:
        json.dump({'run_id': best['run_id'], 'params': best_params, 'accuracy': acc, 'f1': f1, 'classification_report': report}, fh, indent=2)

    with report_txt_path.open('w') as fh:
        fh.write(f"Best run: {best['run_id']}\n")
        fh.write(f"Accuracy: {acc:.4f}\nF1 (weighted): {f1:.4f}\n\n")
        fh.write(text_report)

    print(f"Saved classification report to {report_json_path} and {report_txt_path}")


if __name__ == '__main__':
    main()
