"""
Optional image-based waste classification using a CNN.

The included download helper can prepare a real labelled image dataset in:
image_dataset/<category>/*.jpg

Example:
image_dataset/Plastic/bottle.jpg
image_dataset/Paper/newspaper.jpg

The CNN is intentionally separate from the numerical ANN because images and
structured waste details require different input representations.
"""

from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DATASET_DIR = PROJECT_DIR / "image_dataset"
IMAGE_MODEL_PATH = PROJECT_DIR / "waste_image_cnn.keras"
IMAGE_CLASSES_PATH = PROJECT_DIR / "image_classes.npy"
IMAGE_SIZE = (128, 128)


CNN_GUIDANCE = {
    "Cardboard": {
        "disposal": "Dry Waste / Paper Recycling",
        "tip": "Flatten the box and keep it dry before recycling.",
        "information": "Cardboard is made from recyclable paper fibers.",
        "preparation": "Remove plastic tape and food-soiled sections.",
    },
    "Glass": {
        "disposal": "Glass Recycling Bin",
        "tip": "Do not mix broken glass with ordinary recyclables.",
        "information": "Glass can be recycled repeatedly without losing quality.",
        "preparation": "Rinse the item and check local rules for lids and caps.",
    },
    "Metal": {
        "disposal": "Metal Recycling / Scrap Collection",
        "tip": "Rinse cans or containers to remove food residue.",
        "information": "Metal recycling saves substantial energy and raw materials.",
        "preparation": "Keep sharp metal items safely wrapped before disposal.",
    },
    "Paper": {
        "disposal": "Dry Waste / Paper Recycling",
        "tip": "Keep paper clean and free from oil or liquids.",
        "information": "Clean paper can usually be collected with recycling paper.",
        "preparation": "Remove plastic sleeves and shred confidential documents if needed.",
    },
    "Plastic": {
        "disposal": "Plastic Recycling / Dry Waste",
        "tip": "Check the plastic recycling number and local acceptance rules.",
        "information": "Reusable or recyclable plastic should be separated from trash.",
        "preparation": "Empty and rinse containers before recycling.",
    },
    "Trash": {
        "disposal": "General Waste Bin",
        "tip": "Keep hazardous or contaminated items out of regular recyclables.",
        "information": "This class represents items that are not easily recyclable.",
        "preparation": "If possible, reduce or replace the item instead of disposing of it.",
    },
}


def get_cnn_guidance(category: str) -> dict:
    """Return disposal and environmental information for a CNN category."""
    return CNN_GUIDANCE.get(
        category,
        {
            "disposal": "Follow Local Waste Guidance",
            "tip": "Separate the item according to local recycling rules.",
            "information": "Waste acceptance rules vary by location.",
            "preparation": "Clean and dry recyclable items before disposal.",
        },
    )


def get_confidence_message(confidence: float) -> str:
    """Describe whether the CNN prediction is high or uncertain."""
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence - verify the item manually"


def train_image_cnn(epochs: int = 15):
    """Train and save a CNN from a folder-per-class image dataset."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for CNN training. Install requirements-image.txt."
        ) from error

    if not IMAGE_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Run 'python download_image_dataset.py' first."
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        IMAGE_DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=32,
        validation_split=0.2,
        subset="both",
        seed=42,
    )
    train_dataset, validation_dataset = dataset
    class_names = np.array(train_dataset.class_names)
    if len(class_names) < 2:
        raise ValueError("The image dataset needs at least two category folders.")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )
    model.save(IMAGE_MODEL_PATH)
    np.save(IMAGE_CLASSES_PATH, class_names)
    print(f"CNN saved to {IMAGE_MODEL_PATH}")
    return model


def predict_image(image_path: str):
    """Predict an image category and return category plus confidence."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for image prediction. Install requirements-image.txt."
        ) from error

    if not IMAGE_MODEL_PATH.exists() or not IMAGE_CLASSES_PATH.exists():
        raise FileNotFoundError(
            "Train the CNN first with: python image_cnn.py --train"
        )

    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)[None, ...]
    probabilities = tf.keras.models.load_model(IMAGE_MODEL_PATH).predict(
        image_array, verbose=0
    )[0]
    classes = np.load(IMAGE_CLASSES_PATH, allow_pickle=True)
    index = int(np.argmax(probabilities))
    return str(classes[index]), float(probabilities[index])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or use the waste CNN.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--image", help="Image path for prediction.")
    args = parser.parse_args()
    if args.train:
        train_image_cnn()
    elif args.image:
        category, confidence = predict_image(args.image)
        guidance = get_cnn_guidance(category)
        print(f"\nPredicted Waste Category: {category.upper()}")
        print(f"Prediction Confidence: {confidence:.2%}")
        print(f"Confidence Level: {get_confidence_message(confidence)}")
        print(f"Recommended Disposal: {guidance['disposal']}")
        print(f"Recycling Tip: {guidance['tip']}")
        print(f"Preparation: {guidance['preparation']}")
        print(f"Extra Information: {guidance['information']}")
    else:
        parser.error("Use --train or --image <path>.")
