import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ===========================
# 1. Load Model & Class Names
# ===========================
MODEL_PATH = "mobinet_model.h5" 
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

IMG_SIZE = (224, 224)  # MobileNetV2 input size

# ===========================
# 2. Streamlit UI
# ===========================
st.set_page_config(page_title="Fish Classifier", layout="centered")

st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===========================
    # 3. Preprocess Image
    # ===========================
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # ===========================
    # 4. Predict
    # ===========================
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # ===========================
    # 5. Display Results
    # ===========================
    st.markdown(f"### 🏆 Prediction: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show top 3 predictions
    top_indices = predictions[0].argsort()[-3:][::-1]
    st.subheader("Top 3 Predictions")
    for i in top_indices:
        st.write(f"{class_names[i]}: {predictions[0][i]*100:.2f}%")
