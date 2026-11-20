"""Batch prediction utility for CSV files containing waste details."""

from pathlib import Path

import pandas as pd

from predict import predict_waste


REQUIRED_COLUMNS = {
    "biodegradable",
    "recyclable",
    "weight_grams",
    "material_composition",
}


def classify_csv(input_path: str, output_path: str = "classified_waste.csv"):
    """Classify every row in a CSV and save predictions plus confidence."""
    input_file = Path(input_path)
    frame = pd.read_csv(input_file)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    results = [
        predict_waste(row.to_dict())
        for _, row in frame.iterrows()
    ]
    frame["predicted_category"] = [result[0] for result in results]
    frame["confidence"] = [result[1] for result in results]
    frame.to_csv(output_path, index=False)
    return Path(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify waste from a CSV.")
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="classified_waste.csv")
    arguments = parser.parse_args()
    print(f"Saved results to {classify_csv(arguments.input_csv, arguments.output)}")
