import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import load_model
from src.gradcam import make_gradcam_heatmap
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # must match class_indices.json order

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan and the model will predict the tumor type.")



model = load_model()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    base = model.get_layer("efficientnetb0")
    conv_model = tf.keras.Model(base.inputs, base.get_layer(last_conv_layer_name).output)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs = conv_model(inputs)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in model.layers[2:]:   # skip input_layer and efficientnetb0
            x = layer(x)
        predictions = x

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess — DO NOT divide by 255, the model has Rescaling built in
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        pred_class = CLASS_NAMES[pred_idx]
        confidence = preds[0][pred_idx] * 100

    st.subheader(f"Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("### All class probabilities")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{name}: {preds[0][i]*100:.2f}%")
        st.progress(float(preds[0][i]))

    st.write("### Grad-CAM: where the model focused")
    heatmap = make_gradcam_heatmap(img_array, model, "top_conv")
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    original = np.uint8(img_array[0])
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    st.image(overlay, caption="Grad-CAM heatmap", use_column_width=True)

    st.caption("⚠️ This is a demo project for educational purposes only — not a medical diagnostic tool.")