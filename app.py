
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ODIR-5K Eye Disease Classifier",
    page_icon="👁️",
    layout="centered"
)

IMAGE_SIZE = (224, 224)

# Training dataset statistics
AGE_MEAN = 57.879857079053146
AGE_STD = 11.71689123095599

# Must match the model's training output order
CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M", "O"]

CLASS_DESCRIPTIONS = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension-related Findings",
    "M": "Myopia",
    "O": "Other Retinal Abnormalities",
}


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "odir5k_model.keras",
        compile=False
    )


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # EfficientNetB0 in this trained model uses
    # its internal rescaling layer.
    # Keep pixel values in the 0-255 range.
    image_array = np.asarray(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# ==========================================
# 4. CLINICAL FEATURE PREPROCESSING
# ==========================================
def preprocess_clinical(age, gender):
    # Standardize age using training statistics
    standardized_age = (age - AGE_MEAN) / AGE_STD

    # Original training order:
    # [standardized_age, Female indicator, Male indicator]
    female_indicator = 1.0 if gender == "Female" else 0.0
    male_indicator = 1.0 if gender == "Male" else 0.0

    clinical_array = np.array(
        [[
            standardized_age,
            female_indicator,
            male_indicator
        ]],
        dtype=np.float32
    )

    return clinical_array


# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================
st.title("👁️ ODIR-5K Eye Disease Classifier")

st.subheader("Multimodal Deep Learning Research Project")

st.markdown(
    """
    Explore the predictions of an EfficientNetB0 model
    using a retinal fundus image, age, and gender.
    """
)

st.warning(
    """
    **Research prototype — not for clinical use.**

    This model is not a medical device. Its predictions are
    experimental and may be incorrect. Do not use them for
    diagnosis, screening, treatment, or medical decisions.
    Consult a qualified eye-care professional for medical advice.
    """
)

st.divider()

# ==========================================
# 6. USER INPUTS
# ==========================================
st.subheader("Patient Information")

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=58,
        step=1
    )

with col2:
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )


# ==========================================
# 7. DISPLAY UPLOADED IMAGE
# ==========================================
image_array = None

if uploaded_file is not None:
    try:
        display_image, image_array = preprocess_image(uploaded_file)

        st.image(
            display_image,
            caption="Uploaded retinal fundus image",
            width=350
        )

    except Exception as exc:
        st.error(f"Unable to process the uploaded image: {exc}")
        st.stop()


# ==========================================
# 8. PREDICTION
# ==========================================
if st.button("Run Research Prediction", type="primary"):

    if uploaded_file is None:
        st.error("Please upload a retinal fundus image first.")
        st.stop()

    try:
        with st.spinner("Loading model and generating predictions..."):

            model = load_model()

            clinical_array = preprocess_clinical(age, gender)

            # Model expects two inputs:
            # 1. Image: (1, 224, 224, 3)
            # 2. Clinical features: (1, 3)
            probabilities = model.predict(
                [image_array, clinical_array],
                verbose=0
            )[0]

        # Select highest-scoring class
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100

        # ======================================
        # 9. DISPLAY RESULTS
        # ======================================
        st.divider()
        st.subheader("Model Prediction Results")

        st.metric(
            label="Highest-scoring category",
            value=CLASS_DESCRIPTIONS[predicted_class]
        )

        st.metric(
            label="Model confidence score",
            value=f"{confidence:.2f}%"
        )

        st.caption(
            "The score is the model's output for the selected class. "
            "It is not a clinically validated probability of disease."
        )

        # ======================================
        # 10. ALL CLASS SCORES
        # ======================================
        st.subheader("Scores Across All Eight Classes")

        results = []

        for index, class_code in enumerate(CLASS_NAMES):
            results.append({
                "Class": class_code,
                "Category": CLASS_DESCRIPTIONS[class_code],
                "Model Score (%)": round(
                    float(probabilities[index]) * 100, 2
                )
            })

        results.sort(
            key=lambda row: row["Model Score (%)"],
            reverse=True
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # Visual score comparison
        st.subheader("Class Score Comparison")

        chart_data = {
            row["Class"]: row["Model Score (%)"]
            for row in results
        }

        st.bar_chart(chart_data)

        st.info(
            """
            These results are experimental outputs from a model
            trained on the ODIR-5K dataset. The model has uneven
            performance across classes and has not been validated
            for clinical use. A low score does not rule out disease.
            """
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check the model file, dependency "
            "versions, and model input compatibility."
        )
        st.exception(exc)


# ==========================================
# 11. FOOTER
# ==========================================
st.divider()

st.caption(
    "ODIR-5K Multimodal Classification | "
    "EfficientNetB0 + Clinical Features | "
    "Educational Research Prototype"
)
