"""Optional visual dashboard for the ANN, batch predictions, and CNN."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from batch_predict import classify_csv
from image_cnn import (
    IMAGE_CLASSES_PATH,
    IMAGE_MODEL_PATH,
    get_cnn_guidance,
    get_confidence_message,
    predict_image,
)
from predict import (
    get_disposal_recommendation,
    get_environmental_tip,
    get_prediction_explanation,
    load_model,
    predict_waste,
    record_prediction,
)



st.set_page_config(page_title="Waste Classification System", page_icon="♻️", layout="wide")
st.title("♻️ Waste Classification System")
st.caption("ANN on structured waste details, batch analysis, and CNN image classification")
saved = load_model()

tab_manual, tab_batch, tab_image, tab_about = st.tabs(
    ["Manual ANN Prediction", "Batch CSV", "CNN Image Classification", "Model Insights"]
)

with tab_manual:
    st.subheader("Classify structured waste details")
    left, right = st.columns(2)
    with left:
        biodegradable = st.selectbox("Biodegradable?", ["Yes", "No"])
        recyclable = st.selectbox("Recyclable?", ["Yes", "No"])
    with right:
        weight = st.number_input("Weight (grams)", min_value=1.0, value=250.0)
        material = st.selectbox("Material composition", saved["allowed_materials"])
    if st.button("Predict with ANN", type="primary"):
        details = {
            "biodegradable": biodegradable == "Yes",
            "recyclable": recyclable == "Yes",
            "weight_grams": weight,
            "material_composition": material,
        }
        category, confidence = predict_waste(details)
        record_prediction(details, category, confidence)
        st.success(f"{category.upper()} ({confidence:.2%} confidence)")
        st.caption(get_prediction_explanation(details, category))
        st.info(f"Recommended disposal: {get_disposal_recommendation(category)}")
        st.info(f"Environmental tip: {get_environmental_tip(category)}")

with tab_batch:
    st.subheader("Classify many records at once")
    upload = st.file_uploader("Upload a CSV", type="csv")
    if upload and st.button("Classify CSV"):
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / "input.csv"
            output_path = Path(folder) / "classified.csv"
            input_path.write_bytes(upload.getvalue())
            classify_csv(str(input_path), str(output_path))
            result = pd.read_csv(output_path)
        st.dataframe(result, use_container_width=True)
        st.download_button("Download results", result.to_csv(index=False), "classified_waste.csv")

with tab_image:
    st.subheader("CNN image classification")
    cnn_ready = IMAGE_MODEL_PATH.exists() and IMAGE_CLASSES_PATH.exists()
    if cnn_ready:
        image_classes = ", ".join(
            str(label)
            for label in np.load(IMAGE_CLASSES_PATH, allow_pickle=True)
        )
        st.success(f"CNN is ready. Image classes: {image_classes}")
    else:
        st.warning(
            "CNN model not found. Run `python image_cnn.py --train` "
            "after preparing the image dataset."
        )
    image = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
    if image:
        st.image(image, width=300)
    if image and cnn_ready and st.button("Predict image"):
        with tempfile.NamedTemporaryFile(suffix=Path(image.name).suffix, delete=False) as file:
            file.write(image.getvalue())
            image_path = file.name
        try:
            category, confidence = predict_image(image_path)
            guidance = get_cnn_guidance(category)
            st.success(f"{category.upper()} - {confidence:.2%} confidence")
            st.caption(get_confidence_message(confidence))

            info_left, info_right = st.columns(2)
            with info_left:
                st.info(f"Recommended disposal: {guidance['disposal']}")
                st.info(f"Preparation: {guidance['preparation']}")
            with info_right:
                st.success(f"Recycling tip: {guidance['tip']}")
                st.write(f"About this category: {guidance['information']}")
        except (FileNotFoundError, RuntimeError) as error:
            st.error(str(error))

with tab_about:
    st.subheader("Evaluation")
    st.metric("ANN accuracy", f"{saved['ann_accuracy']:.2%}")
    st.metric("Random Forest accuracy", f"{saved['random_forest_accuracy']:.2%}")
    st.image("ann_training_loss.png", caption="ANN training loss")
    st.image("confusion_matrix.png", caption="ANN confusion matrix")
