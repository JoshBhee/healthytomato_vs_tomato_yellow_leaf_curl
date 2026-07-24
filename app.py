import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

@st.cache_resource
def load_leaf_model():
    return tf.keras.models.load_model('tomato_leaf_classifier_v2.h5')

@st.cache_resource
def load_sanity_model():
    return MobileNetV2(weights='imagenet')

leaf_model = load_leaf_model()
sanity_model = load_sanity_model()

# Keywords that suggest the image is plant/leaf-related, based on ImageNet labels
PLANT_KEYWORDS = ['leaf', 'plant', 'flower', 'vegetable', 'fruit', 'tree', 'fungus', 'mushroom', 'cabbage', 'broccoli']

st.title("Tomato Leaf Classifier")
st.write("Upload a tomato leaf image to check for Healthy or Yellow Leaf Curl Virus.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # --- Sanity check: is this even plant-related? ---
        sanity_img = image.resize((224, 224))
        sanity_array = np.expand_dims(np.array(sanity_img), axis=0)
        sanity_array = preprocess_input(sanity_array)
        sanity_preds = decode_predictions(sanity_model.predict(sanity_array, verbose=0), top=5)[0]

        is_plant_related = any(
            any(keyword in label.lower() for keyword in PLANT_KEYWORDS)
            for (_, label, _) in sanity_preds
        )

        if not is_plant_related:
            st.error(
                "⚠️ Invalid Image\n\n"
                "This image does not appear to be either:\n\n"
                "• Healthy Tomato Leaf\n\n"
                "• Tomato Yellow Leaf Curl Virus\n\n"
                "Please upload a clear tomato leaf image."
            )
        else:
            # --- Passed sanity check, now run the real classifier ---
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = float(leaf_model.predict(img_array, verbose=0)[0][0])

            if prediction > 0.5:
                confidence = prediction * 100
                label = "Tomato Yellow Leaf Curl Virus"
            else:
                confidence = (1 - prediction) * 100
                label = "Healthy Tomato Leaf"

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Something went wrong processing this image: {e}")
