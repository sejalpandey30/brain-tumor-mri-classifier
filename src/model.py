"""
Model loading for the Brain Tumor MRI Classifier.
"""

import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("results/brain_tumor_model.keras")