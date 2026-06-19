"""
Yoga-82 Pose Classifier Training
Trains LightGBM classifiers for 3 hierarchy levels (6, 20, 82 classes).
Uses features_train.csv / features_test.csv from build_features.py.
"""
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Try LightGBM first, fall back to sklearn GradientBoosting
try:
    import lightgbm as lgb
    HAS_LGBM = True
    print("Using LightGBM")
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    print("LightGBM not found -- falling back to sklearn RandomForest + GradientBoosting")

POSE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(POSE_DIR, "static")
FEATURES_TRAIN = os.path.join(POSE_DIR, "features_train.csv")
FEATURES_TEST = os.path.join(POSE_DIR, "features_test.csv")

# Feature columns start after: image_path, label_6, label_20, label_82
LABEL_COLS = ["label_6", "label_20", "label_82"]
META_COLS = ["image_path", "label_6", "label_20", "label_82"]


def load_features():
    """Load train and test feature CSVs."""
    train_df = pd.read_csv(FEATURES_TRAIN)
    test_df = pd.read_csv(FEATURES_TEST)

    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Replace any NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return train_df, test_df, X_train, X_test, feature_cols


def train_lgbm(X_train, y_train, X_test, y_test, num_classes, label_name):
    """Train LightGBM classifier with 5-fold CV for hyperparameter selection."""
    print(f"\n{'='*60}")
    print(f"  Training LightGBM for {label_name} ({num_classes} classes)")
    print(f"{'='*60}")

    # Encode labels to 0-based
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Hyperparameter candidates
    param_sets = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
         "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
         "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.08,
         "num_leaves": 50, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    best_score = -1
    best_params = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, params in enumerate(param_sets):
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
            clf = lgb.LGBMClassifier(
                objective="multiclass" if num_classes > 2 else "binary",
                num_class=num_classes if num_classes > 2 else None,
                is_unbalance=True,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                **params,
            )
            clf.fit(X_train[tr_idx], y_train_enc[tr_idx])
            val_pred = clf.predict(X_train[val_idx])
            f1 = f1_score(y_train_enc[val_idx], val_pred, average="macro", zero_division=0)
            fold_scores.append(f1)

        mean_f1 = np.mean(fold_scores)
        print(f"  Params {i+1}: mean macro-F1 = {mean_f1:.4f} (folds: {[f'{s:.4f}' for s in fold_scores]})")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    print(f"  > Best CV macro-F1: {best_score:.4f}")

    # Train final model on full train set
    final_clf = lgb.LGBMClassifier(
        objective="multiclass" if num_classes > 2 else "binary",
        num_class=num_classes if num_classes > 2 else None,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        **best_params,
    )
    final_clf.fit(X_train, y_train_enc)

    return final_clf, le, best_params


def train_sklearn_fallback(X_train, y_train, X_test, y_test, num_classes, label_name):
    """Fallback: train sklearn RandomForest."""
    print(f"\n{'='*60}")
    print(f"  Training RandomForest for {label_name} ({num_classes} classes)")
    print(f"{'='*60}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train_enc)

    return clf, le, {"n_estimators": 500, "class_weight": "balanced"}


def evaluate_model(clf, le, X_test, y_test, label_name):
    """Evaluate model on test set."""
    y_test_enc = le.transform(y_test)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)

    # Inference latency
    t0 = time.perf_counter()
    for _ in range(100):
        clf.predict(X_test[:1])
    latency_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"\n  {label_name} -- Test Results:")
    print(f"    Accuracy:        {acc:.4f}")
    print(f"    Macro F1:        {f1_macro:.4f}")
    print(f"    Weighted F1:     {f1_weighted:.4f}")
    print(f"    Inference:       {latency_ms:.3f} ms/sample")

    # Per-class report (top confused pairs)
    cm = confusion_matrix(y_test_enc, y_pred)

    # Find top-5 confused pairs
    confused = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused.append((i, j, cm[i][j]))
    confused.sort(key=lambda x: -x[2])

    print(f"    Top-5 confused pairs:")
    for ci, cj, cnt in confused[:5]:
        print(f"      class {le.inverse_transform([ci])[0]} -> {le.inverse_transform([cj])[0]}: {cnt} errors")

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "latency_ms": float(latency_ms),
        "num_test_samples": len(y_test),
        "confusion_matrix_shape": list(cm.shape),
    }


def main():
    print("=" * 60)
    print("Yoga-82 Pose Classifier Training")
    print("=" * 60)

    assert os.path.exists(FEATURES_TRAIN), f"Missing: {FEATURES_TRAIN}\nRun build_features.py first!"
    assert os.path.exists(FEATURES_TEST), f"Missing: {FEATURES_TEST}\nRun build_features.py first!"

    train_df, test_df, X_train, X_test, feature_cols = load_features()
    print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Test samples:  {X_test.shape[0]}")

    os.makedirs(STATIC_DIR, exist_ok=True)

    results = {}
    models = {}
    encoders = {}

    for label_col in LABEL_COLS:
        y_train = train_df[label_col].values
        y_test = test_df[label_col].values
        num_classes = len(np.unique(y_train))

        if HAS_LGBM:
            clf, le, params = train_lgbm(
                X_train, y_train, X_test, y_test, num_classes, label_col
            )
        else:
            clf, le, params = train_sklearn_fallback(
                X_train, y_train, X_test, y_test, num_classes, label_col
            )

        metrics = evaluate_model(clf, le, X_test, y_test, label_col)
        metrics["best_params"] = params

        # Save model
        model_path = os.path.join(STATIC_DIR, f"classifier_{label_col.split('_')[1]}.joblib")
        joblib.dump({"model": clf, "label_encoder": le}, model_path)
        print(f"  Saved: {model_path}")

        results[label_col] = metrics
        models[label_col] = clf
        encoders[label_col] = le

    # Save feature column list for the pipeline
    pipeline_data = {
        "feature_columns": feature_cols,
        "joint_indices": [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
        "coord_joint_indices": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    }
    pipeline_path = os.path.join(STATIC_DIR, "feature_pipeline.joblib")
    joblib.dump(pipeline_data, pipeline_path)
    print(f"\nSaved feature pipeline: {pipeline_path}")

    # Save results summary
    results_path = os.path.join(POSE_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    # Print final summary table
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Level':<12} {'Accuracy':>10} {'Macro F1':>10} {'W. F1':>10} {'Latency':>10}")
    print("-" * 55)
    for label_col in LABEL_COLS:
        r = results[label_col]
        print(f"{label_col:<12} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
              f"{r['weighted_f1']:>10.4f} {r['latency_ms']:>8.3f}ms")


if __name__ == "__main__":
    main()
