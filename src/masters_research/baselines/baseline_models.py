import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


def _load_original_dataset(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load station metadata and monthly attribute CSVs from Dataset/OriginalDataset.

    Returns (attribute_df, node_df).
    """
    data_dir = base_path / "Dataset" / "OriginalDataset"
    # Station metadata
    node_df = pd.read_csv(data_dir / "StationsData.csv", sep=";", skiprows=[0], header=None,
                          names=["ComputerNo", "StationName", "Latitude", "Longitude", "Altitude"]) 

    # Weather station monthly data (8 files as in the notebook)
    station_files = [
        "St_BrandvleiB.csv",
        "St_Kakamas.csv",
        "St_Luckhof.csv",
        "St_Olifantshoek.csv",
        "St_Prieska.csv",
        "St_Vaalharts.csv",
        "St_VanWyksvleiB.csv",
        "St_BerkleyWest.csv",
    ]
    frames = []
    for fname in station_files:
        df = pd.read_csv(data_dir / fname, sep=";", skiprows=[0], header=None,
                         names=[
                             "Compno", "Year", "Month",
                             "AverageMaximumTemperature", "AverageMinimunTemperature",
                             "AverageMaximumRelativeHumidity", "AverageMinimumRelativeHumidity",
                             "TotalRainfall", "AverageTotalRadiation", "AverageWindSpeed",
                             "AverageTotalRelativeEvaporation",
                         ])
        frames.append(df)
    attribute_df = pd.concat(frames, ignore_index=True)
    return attribute_df, node_df


def _clean_and_prepare(attribute_df: pd.DataFrame, node_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Replace decimal commas and tildes
    attribute_df = attribute_df.replace(',', '.', regex=True)
    attribute_df = attribute_df.replace('~', '0', regex=True)
    node_df = node_df.replace(',', '.', regex=True)
    node_df = node_df.replace('~', '0', regex=True)

    # Convert numeric
    attribute_df = attribute_df.apply(pd.to_numeric)
    node_df[["ComputerNo", "Latitude", "Longitude", "Altitude"]] = node_df[["ComputerNo", "Latitude", "Longitude", "Altitude"]].apply(pd.to_numeric)

    # Rename for merge
    node_df = node_df.rename(columns={"ComputerNo": "Compno"})
    return attribute_df, node_df


def _build_graph_df(attribute_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    graph_df = pd.merge(attribute_df, node_df, on='Compno', how='inner')
    graph_df = graph_df.drop(['StationName'], axis=1)
    # Timestamp as Year-Month, converted to datetime, sorted
    graph_df.insert(1, 'TimeStamp', graph_df.apply(lambda row: str(row['Year']).replace('.0', '') + "-" + str(row['Month']).replace('.0', ''), axis=1))
    graph_df['TimeStamp'] = pd.to_datetime(graph_df['TimeStamp'])
    graph_df = graph_df.sort_values(by='TimeStamp').reset_index(drop=True)
    # Drop helper columns
    graph_df = graph_df[['TimeStamp', 'Compno', 'Latitude', 'Longitude', 'Altitude',
                         'AverageMaximumTemperature', 'AverageMinimunTemperature',
                         'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity',
                         'TotalRainfall', 'AverageTotalRadiation', 'AverageWindSpeed',
                         'AverageTotalRelativeEvaporation']]
    return graph_df


def _create_targets(graph_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize selected columns
    columns_to_normalize = [
        'Latitude', 'Longitude', 'Altitude', 'AverageMaximumTemperature', 'AverageMinimunTemperature',
        'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity', 'TotalRainfall',
        'AverageTotalRadiation', 'AverageWindSpeed', 'AverageTotalRelativeEvaporation'
    ]
    scaler = MinMaxScaler()
    graph_df[columns_to_normalize] = scaler.fit_transform(graph_df[columns_to_normalize])
    # Target as mean of normalized features
    graph_df['Target'] = graph_df.drop(['Compno', 'TimeStamp'], axis=1).apply(lambda x: x.mean(), axis=1)
    # Quantile categories
    graph_df['Quantile_Category'] = pd.qcut(graph_df['Target'], q=3, labels=['Low', 'Medium', 'High'])
    # Encode
    le = LabelEncoder()
    graph_df['Category_encoded'] = le.fit_transform(graph_df['Quantile_Category'])
    return graph_df


def _prepare_final_df(graph_df: pd.DataFrame) -> pd.DataFrame:
    final_df = graph_df[['TimeStamp', 'Compno', 'Latitude', 'Longitude', 'Altitude',
                         'AverageMaximumTemperature', 'AverageMinimunTemperature',
                         'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity',
                         'TotalRainfall', 'AverageTotalRadiation', 'AverageWindSpeed',
                         'AverageTotalRelativeEvaporation', 'Target']].copy()
    final_df = final_df.sort_values(by=['Compno', 'TimeStamp'])
    return final_df


def _add_time_shift_features(final_df: pd.DataFrame) -> pd.DataFrame:
    time_df = final_df.copy()
    time_df['TimeStamp'] = pd.to_datetime(time_df['TimeStamp'])
    # Align by compno & timestamp for lags/leads
    time_corrected_df = time_df.sort_values(by=['Compno', 'TimeStamp']).reset_index(drop=True)
    # Shifted target (next month)
    time_corrected_df['Shifted_Target'] = time_corrected_df.apply(
        lambda row: time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] + pd.DateOffset(months=1)), 'Target'
        ].values[0] if not time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] + pd.DateOffset(months=1))
        ].empty else None,
        axis=1
    )
    # Lag1
    time_corrected_df['Lag1_Target'] = time_corrected_df.apply(
        lambda row: time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] - pd.DateOffset(months=1)), 'Target'
        ].values[0] if not time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] - pd.DateOffset(months=1))
        ].empty else None,
        axis=1
    )
    # Lag2
    time_corrected_df['Lag2_Target'] = time_corrected_df.apply(
        lambda row: time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] - pd.DateOffset(months=2)), 'Target'
        ].values[0] if not time_corrected_df.loc[
            (time_corrected_df['Compno'] == row['Compno']) &
            (time_corrected_df['TimeStamp'] == row['TimeStamp'] - pd.DateOffset(months=2))
        ].empty else None,
        axis=1
    )
    analysis_df = time_corrected_df[['TimeStamp', 'Compno', 'Latitude', 'Longitude', 'Altitude',
                                     'AverageMaximumTemperature', 'AverageMinimunTemperature',
                                     'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity',
                                     'TotalRainfall', 'AverageTotalRadiation', 'AverageWindSpeed',
                                     'AverageTotalRelativeEvaporation', 'Lag1_Target', 'Lag2_Target', 'Shifted_Target']]
    # Use shifted targets as label
    analysis_df = analysis_df.dropna(subset=['Lag1_Target', 'Lag2_Target', 'Shifted_Target']).rename(columns={'Shifted_Target': 'Target'})
    return analysis_df


def _save_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    # Determine classes from data
    classes = sorted(pd.Series(y_true).unique().tolist())
    target_names = [str(c) for c in classes]
    report_str = classification_report(y_true, y_pred, labels=classes, target_names=target_names, digits=4, zero_division=0)
    out_txt.write_text(report_str)


def _save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(pd.Series(y_true).unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - RandomForest Baseline")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main(base_path: Path | str | None = None):
    """Run baseline models from the notebook as a Python script.

    - Loads OriginalDataset CSVs
    - Prepares features and target
    - GridSearchCV for RandomForest
    - Evaluates and saves original text classification report and confusion matrix
    - Saves the best RF model via joblib
    - Optionally runs a Keras grid search if TensorFlow is available
    """
    base_path = Path(base_path or Path(__file__).resolve().parents[3]).resolve()

    # Load and preprocess

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
    # --- Data loading and cleaning ---
    attribute_df, node_df = _load_original_dataset(base_path)
    attribute_df, node_df = _clean_and_prepare(attribute_df, node_df)
    graph_df = _build_graph_df(attribute_df, node_df)
    # Create 'Target' as mean of features (no normalization or binning here)
    feature_cols_for_target = [
        'Latitude', 'Longitude', 'Altitude',
        'AverageMaximumTemperature', 'AverageMinimunTemperature',
        'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity',
        'TotalRainfall', 'AverageTotalRadiation', 'AverageWindSpeed',
        'AverageTotalRelativeEvaporation'
    ]
    graph_df['Target'] = graph_df[feature_cols_for_target].mean(axis=1)
    final_df = _prepare_final_df(graph_df)
    analysis_df = _add_time_shift_features(final_df)
    analysis_df = analysis_df.dropna()

    # --- Custom target quantile binning transformer ---
    class TargetQuantileBinner(BaseEstimator, TransformerMixin):
        def __init__(self, n_bins=3, labels=None):
            self.n_bins = n_bins
            self.labels = list(range(self.n_bins))
        def fit(self, y, *_):
            import pandas as pd
            self.bin_edges_ = pd.qcut(y, q=self.n_bins, retbins=True, labels=False, duplicates='drop')[1]
            return self
        def transform(self, y):
            import pandas as pd
            binned = pd.cut(y, bins=self.bin_edges_, labels=self.labels, include_lowest=True)
            # Assign out-of-bounds to the lowest bin (can change to self.labels[-1] for upper outliers)
            if self.labels[0] not in binned.cat.categories:
                binned = binned.cat.add_categories([self.labels[0]])
            binned = binned.fillna(self.labels[0])
            return binned.astype(int).to_numpy()

    # --- Prepare features and target ---
    feature_cols = [
        'Latitude', 'Longitude', 'Altitude',
        'AverageMaximumTemperature', 'AverageMinimunTemperature',
        'AverageMaximumRelativeHumidity', 'AverageMinimumRelativeHumidity',
        'TotalRainfall', 'AverageTotalRadiation', 'AverageWindSpeed',
        'AverageTotalRelativeEvaporation', 'Lag1_Target', 'Lag2_Target'
    ]
    X = analysis_df[feature_cols]
    y_raw = analysis_df['Target']

    # --- Split before any scaling or binning ---
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    # --- Fit quantile binning only on train ---
    binner = TargetQuantileBinner(n_bins=3, labels=['Low', 'Medium', 'High'])
    y_train = binner.fit_transform(y_train_raw)
    y_test = binner.transform(y_test_raw)

    # --- Pipeline for features ---
    scaler = MinMaxScaler()
    feature_pipeline = Pipeline([
        ('scaler', scaler)
    ])

    # --- RandomForest GridSearch ---
    param_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline([
        ('preprocess', feature_pipeline),
        ('clf', rf)
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid={
        'clf__n_estimators': param_grid['n_estimators'],
        'clf__max_depth': param_grid['max_depth'],
        'clf__min_samples_split': param_grid['min_samples_split'],
        'clf__min_samples_leaf': param_grid['min_samples_leaf'],
        'clf__max_features': param_grid['max_features'],
    }, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"CV Best Score: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")
    print("\nClassification Report with Best Model:")
    print(classification_report(y_test, y_pred_best))

    # Outputs
    out_dir_models = base_path / "src" / "masters_research" / "artifacts" / "models"
    out_dir_models.mkdir(parents=True, exist_ok=True)
    # Save model
    rf_model_path = out_dir_models / "RandomForestClassifier.pkl"
    joblib.dump(best_model, rf_model_path)
    # Save classification report (original text layout)
    cr_txt = out_dir_models / "RandomForest_classification_report.txt"
    _save_classification_report_text(y_test, y_pred_best, cr_txt)
    # Save confusion matrix plot
    cm_png = out_dir_models / "RandomForest_confusion_matrix.png"
    _save_confusion_matrix_plot(y_test, y_pred_best, cm_png)
    print(f"Saved RF model to: {rf_model_path}")
    print(f"Saved RF classification report to: {cr_txt}")
    print(f"Saved RF confusion matrix plot to: {cm_png}")

    # Optional: Keras baseline if TensorFlow available
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        # Try SciKeras; fallback to direct Keras if incompatible
        try:
            from scikeras.wrappers import KerasClassifier

            def create_model(learning_rate=0.001, activation='relu'):
                model = Sequential([
                    Dense(128, activation=activation, input_shape=(X_train.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(len(y.unique()), activation='softmax')
                ])
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model

            keras_clf = KerasClassifier(model=create_model, verbose=0)
            param_grid_keras = {
                'batch_size': [16, 32, 64],
                'epochs': [120, 135, 150],
                'model__learning_rate': [0.0025, 0.003, 0.0035],
                'model__activation': ['relu', 'tanh', 'swish']
            }
            grid_keras = GridSearchCV(estimator=keras_clf, param_grid=param_grid_keras, cv=3, verbose=1, n_jobs=-1)
            grid_keras_result = grid_keras.fit(X_train, y_train)
            print(f"Keras Best Parameters: {grid_keras_result.best_params_}")
            print(f"Keras CV Best Score: {grid_keras_result.best_score_:.4f}")

            best_keras = grid_keras_result.best_estimator_
            y_pred_keras = best_keras.predict(X_test)
            keras_acc = accuracy_score(y_test, y_pred_keras)
            print(f"Test Accuracy (Keras Best): {keras_acc:.4f}")
            print("\nKeras Classification Report with Best Model:")
            print(classification_report(y_test, y_pred_keras))

            # Save Keras model (h5) and estimator in artifacts/models
            keras_model_path = out_dir_models / "KerasClassifier.h5"
            keras_model_path.parent.mkdir(parents=True, exist_ok=True)
            best_keras.model_.save(str(keras_model_path))
            keras_estimator_path = out_dir_models / "KerasClassifier.pkl"
            joblib.dump(best_keras, keras_estimator_path)
            keras_cr_txt = out_dir_models / "Keras_classification_report.txt"
            _save_classification_report_text(y_test, y_pred_keras, keras_cr_txt)
            keras_cm_png = out_dir_models / "Keras_confusion_matrix.png"
            _save_confusion_matrix_plot(y_test, y_pred_keras, keras_cm_png)
            print(f"Saved Keras model to: {keras_model_path}")
            print(f"Saved Keras estimator to: {keras_estimator_path}")
            print(f"Saved Keras classification report to: {keras_cr_txt}")
            print(f"Saved Keras confusion matrix plot to: {keras_cm_png}")
        except Exception as sci_err:
            print(f"SciKeras unavailable or incompatible; training direct Keras model: {sci_err}")

            def create_model(learning_rate=0.003, activation='relu'):
                model = Sequential([
                    Dense(128, activation=activation, input_shape=(X_train.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation=activation),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(len(y.unique()), activation='softmax')
                ])
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model

            keras_model = create_model()
            # Use reasonable defaults matching the grid
            keras_model.fit(X_train, y_train, batch_size=32, epochs=135, verbose=0)
            y_pred_probs = keras_model.predict(X_test, verbose=0)
            y_pred_keras = np.argmax(y_pred_probs, axis=1)
            keras_acc = accuracy_score(y_test, y_pred_keras)
            print(f"Test Accuracy (Keras direct): {keras_acc:.4f}")
            print("\nKeras Classification Report (direct training):")
            print(classification_report(y_test, y_pred_keras))

            # Save direct Keras model
            keras_model_path = out_dir_models / "KerasClassifier.h5"
            keras_model.save(str(keras_model_path))
            keras_cr_txt = out_dir_models / "Keras_classification_report.txt"
            _save_classification_report_text(y_test, y_pred_keras, keras_cr_txt)
            keras_cm_png = out_dir_models / "Keras_confusion_matrix.png"
            _save_confusion_matrix_plot(y_test, y_pred_keras, keras_cm_png)
            print(f"Saved Keras model to: {keras_model_path}")
            print(f"Saved Keras classification report to: {keras_cr_txt}")
            print(f"Saved Keras confusion matrix plot to: {keras_cm_png}")
    except Exception as e:
        print(f"Skipping Keras baseline (TensorFlow not available or failed): {e}")


if __name__ == "__main__":
    # Run from project root for consistent relative paths
    project_root = Path(__file__).resolve().parents[3]
    os.chdir(project_root)
    print(f"Project root: {project_root}")
    main(project_root)
