import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Classifier",
    page_icon="🍅",
    layout="centered"
)


# ============================================================
# LOAD YOUR 3-CLASS MODEL
#
# Your model file:
# tomato_leaf_classifier_v2.h5
# ============================================================

@st.cache_resource
def load_leaf_model():
    return tf.keras.models.load_model(
        "tomato_leaf_classifier_v2.h5"
    )


leaf_model = load_leaf_model()


# ============================================================
# CLASS NAMES
#
# Your Google Colab showed:
#
# healthy = 0
# other = 1
# yellow_curl_virus = 2
#
# Therefore, this order MUST NOT be changed.
# ============================================================

CLASS_NAMES = [
    "Healthy Tomato Leaf",
    "Other / Invalid Image",
    "Tomato Yellow Leaf Curl Virus"
]


# ============================================================
# APP TITLE
# ============================================================

st.title("🍅 Tomato Leaf Classifier")

st.write(
    "Upload a tomato leaf image to determine whether "
    "it is a Healthy Tomato Leaf or affected by "
    "Tomato Yellow Leaf Curl Virus."
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "avif"]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # DISPLAY UPLOADED IMAGE
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ----------------------------------------------------
        # RESIZE IMAGE
        #
        # Your model was trained using:
        # img_size = (224, 224)
        # ----------------------------------------------------

        img = image.resize(
            (224, 224)
        )


        # ----------------------------------------------------
        # CONVERT IMAGE TO NUMPY ARRAY
        # ----------------------------------------------------

        img_array = np.array(
            img,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # NORMALIZE IMAGE
        #
        # Your training used:
        #
        # ImageDataGenerator(
        #     rescale=1./255
        # )
        #
        # Therefore we also divide by 255 here.
        # ----------------------------------------------------

        img_array = img_array / 255.0


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        predictions = leaf_model.predict(
            img_array,
            verbose=0
        )


        # ----------------------------------------------------
        # FIND CLASS WITH HIGHEST PROBABILITY
        # ----------------------------------------------------

        predicted_class_index = np.argmax(
            predictions[0]
        )


        # ----------------------------------------------------
        # CALCULATE CONFIDENCE
        # ----------------------------------------------------

        confidence = (
            predictions[0][predicted_class_index]
            * 100
        )


        # ----------------------------------------------------
        # GET PREDICTED CLASS NAME
        # ----------------------------------------------------

        predicted_class = CLASS_NAMES[
            predicted_class_index
        ]


        # ====================================================
        # HANDLE "OTHER" CLASS
        #
        # Your class order says:
        #
        # 0 = Healthy
        # 1 = Other
        # 2 = Yellow Curl Virus
        #
        # Therefore "other" is index 1.
        # ====================================================

        if predicted_class_index == 1:

            st.error(
                "⚠️ Invalid Image\n\n"
                "This image does not appear to be either:\n\n"
                "• Healthy Tomato Leaf\n\n"
                "• Tomato Yellow Leaf Curl Virus\n\n"
                "Please upload a clear tomato leaf image."
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )


        # ====================================================
        # HANDLE VALID TOMATO LEAF
        # ====================================================

        else:

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )


        # ====================================================
        # DISPLAY ALL CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )


        st.write(
            f"Healthy Tomato Leaf: "
            f"{predictions[0][0] * 100:.2f}%"
        )


        st.write(
            f"Other / Invalid Image: "
            f"{predictions[0][1] * 100:.2f}%"
        )


        st.write(
            f"Tomato Yellow Leaf Curl Virus: "
            f"{predictions[0][2] * 100:.2f}%"
        )


    except Exception as e:

        st.error(
            "An error occurred while processing "
            "the uploaded image."
        )

        st.exception(e)