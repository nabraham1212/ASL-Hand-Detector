import os
import sys
import json
from datetime import datetime

import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Config ----
CSV_PATH = os.path.join("data", "asl_landmarks.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "asl_model.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "asl_model_info.json")
CM_IMAGE_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

EXPECTED_FEATURES = 63
MIN_SAMPLES_WARN = 20
N_ESTIMATORS = 200
RANDOM_STATE = 42
TEST_SIZE = 0.2


def die(msg):
    """Print a clear error and stop. Used only when the dataset is unusable."""
    print(f"\nERROR: {msg}")
    sys.exit(1)


def load_and_validate():
    """Load the CSV and run all sanity checks. Returns (X, y, feature_cols, df)."""
    # 1) File exists?
    if not os.path.exists(CSV_PATH):
        die(f"Dataset not found at {CSV_PATH}. "
            f"Run collect_asl_data.py (Phase 3) first to create it.")

    df = pd.read_csv(CSV_PATH)

    # 2) Empty?
    if df.shape[0] == 0:
        die(f"{CSV_PATH} has a header but no data rows. Collect samples first.")

    # 3) label column present?
    if "label" not in df.columns:
        die("No 'label' column found. The CSV header should start with 'label'.")

    feature_cols = [c for c in df.columns if c != "label"]

    # 4) exactly 63 feature columns?
    if len(feature_cols) != EXPECTED_FEATURES:
        die(f"Expected {EXPECTED_FEATURES} feature columns, found {len(feature_cols)}. "
            f"The CSV may be from a different format. Delete it and re-collect, "
            f"or check collect_asl_data.py.")

    # 5) coerce features to numeric and handle missing/bad values
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = n_before - len(df)
    if dropped > 0:
        print(f"Note: dropped {dropped} row(s) with missing/invalid feature values.")
    if df.shape[0] == 0:
        die("All rows had invalid feature values. Dataset unusable.")

    # 6) label counts
    counts = df["label"].value_counts().sort_index()

    # 7) at least 2 labels?
    if counts.shape[0] < 2:
        die(f"Only 1 label ({list(counts.index)}) in the dataset. "
            f"A classifier needs at least 2 different letters. Collect more.")

    # 8) warn on thin labels
    for label, n in counts.items():
        if n < MIN_SAMPLES_WARN:
            print(f"Warning: Label {label} only has {n} samples. "
                  f"Model accuracy may be poor.")

    X = df[feature_cols]
    y = df["label"]
    return X, y, feature_cols, df, counts


def make_split(X, y, counts):
    """Stratified split when possible; fall back gracefully when a class is too small."""
    min_count = int(counts.min())

    # Stratify needs at least 2 samples per class (one for train, one for test).
    if min_count >= 2:
        try:
            return train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            ), "stratified"
        except ValueError:
            pass  # fall through to non-stratified

    print("Warning: at least one label has too few samples for a stratified split. "
          "Falling back to a random (non-stratified) split. Collect more samples "
          "(aim for 30+ per letter) for reliable evaluation.")
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    ), "random"


def print_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    header = "      " + "".join(f"{lab:>5}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = f"{lab:>4}  " + "".join(f"{cm[i][j]:>5}" for j in range(len(labels)))
        print(row)


def try_save_confusion_image(y_test, y_pred, labels):
    """Optional PNG of the confusion matrix. Skipped if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="magma", colorbar=False)
        ax.set_title("ASL Model Confusion Matrix")
        fig.tight_layout()
        fig.savefig(CM_IMAGE_PATH, dpi=120)
        plt.close(fig)
        print(f"Confusion matrix image saved to: {CM_IMAGE_PATH}")
    except ImportError:
        print("(matplotlib not installed - skipping confusion matrix image. Optional.)")
    except Exception as e:
        print(f"(Could not save confusion matrix image: {e})")


def main():
    print("=== SignSpell - Phase 4: Train ASL Model ===\n")

    X, y, feature_cols, df, counts = load_and_validate()

    print(f"Dataset loaded: {CSV_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Labels: {', '.join(counts.index.tolist())}")
    print("Samples per label:")
    for label, n in counts.items():
        print(f"   {label}: {n}")
    print(f"Features per sample: {len(feature_cols)}")

    (X_train, X_test, y_train, y_test), split_kind = make_split(X, y, counts)
    print(f"\nSplit ({split_kind}): {len(X_train)} train / {len(X_test)} test")

    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS})...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = list(model.classes_)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print_confusion_matrix(y_test, y_pred, labels)

    # ---- Save model + metadata ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    info = {
        "model_type": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "dataset_path": CSV_PATH,
        "labels": labels,                          # order matches model.classes_
        "num_samples": int(len(df)),
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,           # exact order Phase 5 must rebuild
        "samples_per_label": {k: int(v) for k, v in counts.items()},
        "test_accuracy": round(float(acc), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split_kind": split_kind,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {INFO_PATH}")
    try_save_confusion_image(y_test, y_pred, labels)
    print("\nDone. Ready for Phase 5 (live prediction).")


if __name__ == "__main__":
    main()