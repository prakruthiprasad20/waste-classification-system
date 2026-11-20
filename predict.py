"""
Load the trained waste classification model and generate predictions.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "waste_model.pkl"
HISTORY_PATH = PROJECT_DIR / "prediction_history.csv"

DISPOSAL_GUIDE = {
    "Organic": "Green/Compost Bin",
    "Plastic": "Dry Waste/Plastic Recycling Bin",
    "Paper": "Dry Waste/Paper Recycling Bin",
    "Metal": "Dry Waste/Metal Recycling Bin",
    "Glass": "Glass Recycling Bin",
}


def load_model():
    """Load the saved model object and report a clear error if it is absent."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "waste_model.pkl was not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


def predict_waste(waste_details: Dict) -> Tuple[str, float]:
    """Generate a category and confidence using the trained model."""
    saved_object = load_model()
    model = saved_object["model"]
    feature_columns = saved_object["feature_columns"]

    input_frame = pd.DataFrame(
        [{
            "biodegradable": int(waste_details["biodegradable"]),
            "recyclable": int(waste_details["recyclable"]),
            "weight_grams": float(waste_details["weight_grams"]),
            "material_composition": str(waste_details["material_composition"]),
        }],
        columns=feature_columns,
    )

    prediction = str(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    confidence = float(max(probabilities))
    return prediction, confidence


def get_disposal_recommendation(category: str) -> str:
    """Return a disposal recommendation for a predicted category."""
    return DISPOSAL_GUIDE.get(category, "General Waste Bin")


def get_prediction_explanation(waste_details: Dict, category: str) -> str:
    """Provide a beginner-friendly explanation of the input pattern."""
    material = str(waste_details["material_composition"])
    if material == category or (category == "Organic" and material == "Food Waste"):
        return f"The material composition ({material}) strongly matches {category}."
    return (
        f"The ANN combined the material ({material}), biodegradability, "
        f"recyclability, and weight patterns to select {category}."
    )


def get_environmental_tip(category: str) -> str:
    """Return a simple environmental action associated with the category."""
    tips = {
        "Organic": "Compost it where possible to reduce landfill methane.",
        "Plastic": "Clean and send it to a plastic recycling collection point.",
        "Paper": "Keep it dry and recycle it with other paper products.",
        "Metal": "Recycle it; metals can often be recovered repeatedly.",
        "Glass": "Reuse or place it in a glass recycling collection point.",
    }
    return tips.get(category, "Follow your local waste-management guidance.")


def record_prediction(waste_details: Dict, category: str, confidence: float):
    """Append a prediction to a CSV history file for later review."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **waste_details,
        "predicted_category": category,
        "confidence": confidence,
    }
    history = pd.DataFrame([row])
    history.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False,
    )
