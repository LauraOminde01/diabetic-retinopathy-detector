import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/diabetic_retinopathy_model.keras')
    return model

model = load_model()

st.title("Diabetic Retinopathy Detector")
st.write(
    "Upload a retina scan image, and this tool will estimate whether it shows "
    "signs of diabetic retinopathy. Built using transfer learning on MobileNetV2, "
    "trained and evaluated on a balanced dataset of retina images."
)

uploaded_file = st.file_uploader("Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_prob = model.predict(img_array)[0][0]

    # Recall from training: class_names were ['DR', 'No_DR']
    # 0 = DR, 1 = No_DR, so prediction_prob close to 1 means No_DR, close to 0 means DR
    if prediction_prob > 0.5:
        confidence = prediction_prob
        st.success(f"Result: No signs of Diabetic Retinopathy detected")
        st.write(f"Confidence: {confidence:.1%}")
    else:
        confidence = 1 - prediction_prob
        st.error(f"Result: Signs of Diabetic Retinopathy detected")
        st.write(f"Confidence: {confidence:.1%}")

    st.warning(
        "This tool achieves 89% accuracy on test data and is intended as a "
        "portfolio demonstration, not a medical diagnostic tool. Always consult "
        "a qualified eye care professional for an actual diagnosis."
    )

st.markdown("---")
st.caption("Model: MobileNetV2 transfer learning | Test accuracy: 89.18% | DR recall: 89%")