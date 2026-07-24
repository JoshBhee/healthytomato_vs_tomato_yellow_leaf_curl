import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato_leaf_classifier.h5')

model = load_model()

st.title("Tomato Leaf Classifier")
st.write("Upload a tomato leaf image to check for Healthy or Yellow Leaf Curl Virus.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = float(model.predict(img_array, verbose=0)[0][0])

        # class_indices from training: {'healthy': 0, 'yellow_curl_virus': 1}
        if prediction > 0.5:
            confidence = prediction * 100
            label = "Tomato Yellow Leaf Curl Virus"
        else:
            confidence = (1 - prediction) * 100
            label = "Healthy Tomato Leaf"

        if confidence < 80:
            st.error(
                "⚠️ Invalid Image\n\n"
                "This image does not appear to be either:\n\n"
                "• Healthy Tomato Leaf\n\n"
                "• Tomato Yellow Leaf Curl Virus\n\n"
                "Please upload a clear tomato leaf image."
            )
        else:
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Something went wrong processing this image: {e}")