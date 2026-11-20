"""
Interactive terminal application for waste classification.
"""

from predict import (
    MODEL_PATH,
    get_environmental_tip,
    get_prediction_explanation,
    get_disposal_recommendation,
    load_model,
    predict_waste,
    record_prediction,
)


def prompt_yes_no(question: str) -> int:
    """Prompt until the user enters yes or no."""
    while True:
        value = input(question).strip().lower()
        if value in {"y", "yes"}:
            return 1
        if value in {"n", "no"}:
            return 0
        print("Invalid input. Please enter Yes or No.")


def prompt_weight() -> float:
    """Prompt until the user enters a positive numeric weight."""
    while True:
        value = input("Weight (grams): ").strip()
        try:
            weight = float(value)
            if weight <= 0:
                raise ValueError
            return weight
        except ValueError:
            print("Invalid input. Please enter a weight greater than 0.")


def prompt_material(allowed_materials) -> str:
    """Prompt until the user enters an available material name."""
    while True:
        value = input("Material composition: ").strip()
        matches = {
            material.lower(): material
            for material in allowed_materials
        }
        if value.lower() in matches:
            return matches[value.lower()]
        print("Invalid material. Choose one of: " + ", ".join(allowed_materials))


def main():
    print("=" * 50)
    print("   WASTE TYPE CLASSIFICATION SYSTEM")
    print("=" * 50)

    if not MODEL_PATH.exists():
        print("\nNo trained model found.")
        print("Please run: python train_model.py")
        return

    try:
        saved_object = load_model()
    except Exception as error:
        print(f"\nCould not load the model: {error}")
        return

    print(
        f"\nActive model: {saved_object.get('model_type', 'Trained classifier')}"
    )
    print(f"ANN accuracy: {saved_object['accuracy']:.2%}")
    if "random_forest_accuracy" in saved_object:
        print(
            "Random Forest comparison accuracy: "
            f"{saved_object['random_forest_accuracy']:.2%}"
        )
    print(
        f"Training: {saved_object['training_size']} rows | "
        f"Testing: {saved_object['testing_size']} rows"
    )

    while True:
        print("\nEnter waste details:")
        biodegradable = prompt_yes_no("Biodegradable (Yes/No): ")
        recyclable = prompt_yes_no("Recyclable (Yes/No): ")
        weight = prompt_weight()
        material = prompt_material(saved_object["allowed_materials"])

        print("\nAnalyzing waste...")
        details = {
            "biodegradable": biodegradable,
            "recyclable": recyclable,
            "weight_grams": weight,
            "material_composition": material,
        }
        category, confidence = predict_waste(details)
        record_prediction(details, category, confidence)

        print(f"\nPredicted Waste Category: {category.upper()}")
        print(f"Prediction Confidence: {confidence:.2%}")
        print(f"Why: {get_prediction_explanation(details, category)}")
        print(
            "Recommended Disposal: "
            + get_disposal_recommendation(category)
        )
        print(f"Environmental tip: {get_environmental_tip(category)}")

        again = input("\nClassify another item? (Yes/No): ").strip().lower()
        if again not in {"y", "yes"}:
            print("\nThank you for using the Waste Classification System.")
            break


if __name__ == "__main__":
    main()
