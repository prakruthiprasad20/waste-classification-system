"""
Train ANN and Random Forest waste classification models.

This project uses a clearly labelled synthetic dataset because a small, stable
public CSV dataset could not be bundled with the project. The synthetic data
is for demonstration and should not be presented as real-world data.
"""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"
MODEL_PATH = PROJECT_DIR / "waste_model.pkl"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "confusion_matrix.png"
ANN_LOSS_PATH = PROJECT_DIR / "ann_training_loss.png"

FEATURE_COLUMNS = [
    "biodegradable",
    "recyclable",
    "weight_grams",
    "material_composition",
]
TARGET_COLUMN = "waste_category"
RANDOM_SEED = 42


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a labelled, reproducible synthetic dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    row_count = 120
    labels = []

    specifications = {
        "Organic": ("Food Waste", 0.95, 0.08, 350, 180),
        "Plastic": ("Plastic", 0.08, 0.92, 180, 120),
        "Paper": ("Paper", 0.82, 0.88, 220, 130),
        "Metal": ("Metal", 0.02, 0.96, 420, 260),
        "Glass": ("Glass", 0.01, 0.94, 650, 280),
    }

    for category, (material, biodegradable_probability,
                   recyclable_probability, mean_weight, weight_scale) in specifications.items():
        for _ in range(row_count):
            weight = max(20, round(rng.normal(mean_weight, weight_scale), 1))
            labels.append({
                "biodegradable": int(rng.random() < biodegradable_probability),
                "recyclable": int(rng.random() < recyclable_probability),
                "weight_grams": weight,
                "material_composition": material,
                "waste_category": category,
            })

    frame = pd.DataFrame(labels)
    frame = frame.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    frame.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic training dataset created at: {DATASET_PATH}")
    return frame


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the dataset before training."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    cleaned = dataset.copy()
    cleaned = cleaned.drop_duplicates()

    for column in ["biodegradable", "recyclable"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["weight_grams"] = pd.to_numeric(
        cleaned["weight_grams"], errors="coerce"
    )
    cleaned["material_composition"] = (
        cleaned["material_composition"].astype("string").str.strip().str.title()
    )
    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype("string").str.strip().str.title()
    )

    cleaned = cleaned.dropna(subset=required)
    cleaned = cleaned[
        cleaned["biodegradable"].isin([0, 1])
        & cleaned["recyclable"].isin([0, 1])
        & (cleaned["weight_grams"] >= 0)
    ].copy()

    cleaned["biodegradable"] = cleaned["biodegradable"].astype(int)
    cleaned["recyclable"] = cleaned["recyclable"].astype(int)
    cleaned["weight_grams"] = cleaned["weight_grams"].astype(float)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str)

    if len(cleaned) < 20:
        raise ValueError("Not enough valid rows remain for training.")
    return cleaned


def train_model():
    """Load, preprocess, train ANN and Random Forest, evaluate, and save."""
    if not DATASET_PATH.exists():
        dataset = generate_synthetic_dataset()
    else:
        dataset = pd.read_csv(DATASET_PATH)

    cleaned = clean_dataset(dataset)
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", "passthrough", ["biodegradable", "recyclable"]),
            ("weight", StandardScaler(), ["weight_grams"]),
            (
                "material",
                OneHotEncoder(handle_unknown="ignore"),
                ["material_composition"],
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", random_forest),
        ]
    )

    # MLPClassifier is a feed-forward Artificial Neural Network (ANN).
    # StandardScaler and one-hot encoding make the input suitable for neurons.
    ann = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
    )
    ann_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ann),
        ]
    )

    print("\nTraining the Artificial Neural Network (ANN)...")
    ann_pipeline.fit(X_train, y_train)
    ann_predictions = ann_pipeline.predict(X_test)

    print("Training the Random Forest comparison model...")
    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)

    ann_accuracy = accuracy_score(y_test, ann_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, ann_predictions, labels=labels)

    print("\n===== TRAINING RESULTS =====")
    print(f"Cleaned dataset size: {len(cleaned)}")
    print(f"Training dataset size: {len(X_train)}")
    print(f"Testing dataset size: {len(X_test)}")
    print(f"ANN accuracy: {ann_accuracy:.2%}")
    print(f"Random Forest accuracy: {rf_accuracy:.2%}")
    print(f"ANN epochs completed: {ann.n_iter_}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())
    print("\nClassification Report:")
    print(classification_report(y_test, ann_predictions, zero_division=0))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Waste Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix image saved at: {CONFUSION_MATRIX_PATH}")

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    loss_ax.plot(ann.loss_curve_, color="darkorange")
    loss_ax.set_title("ANN Training Loss")
    loss_ax.set_xlabel("Training iteration")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)
    loss_fig.tight_layout()
    loss_fig.savefig(ANN_LOSS_PATH, dpi=150)
    plt.close(loss_fig)
    print(f"ANN loss curve saved at: {ANN_LOSS_PATH}")

    saved_object = {
        # The ANN is the model used by predict.py and the terminal app.
        "model": ann_pipeline,
        "ann_model": ann_pipeline,
        "random_forest_model": rf_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "allowed_materials": sorted(X["material_composition"].unique().tolist()),
        "accuracy": ann_accuracy,
        "ann_accuracy": ann_accuracy,
        "random_forest_accuracy": rf_accuracy,
        "training_size": len(X_train),
        "testing_size": len(X_test),
        "labels": labels,
        "model_type": "Artificial Neural Network (MLPClassifier)",
    }
    joblib.dump(saved_object, MODEL_PATH)
    print(f"Trained model saved at: {MODEL_PATH}")
    return saved_object


if __name__ == "__main__":
    train_model()
