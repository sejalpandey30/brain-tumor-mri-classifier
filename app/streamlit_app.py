import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import load_model
from src.gradcam import make_gradcam_heatmap
from src.ui_components import UIComponents
from src.theme import apply_theme, get_theme_config
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import json
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "history" not in st.session_state:
    st.session_state.history = []
if "show_model_info" not in st.session_state:
    st.session_state.show_model_info = False

# Apply theme
apply_theme(st.session_state.theme)
ui = UIComponents(st.session_state.theme)

# Load model
@st.cache_resource
def load_ml_model():
    return load_model()

model = load_ml_model()

# Constants
IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DESCRIPTIONS = {
    'glioma': 'Brain tumor originating in glial cells. Requires aggressive treatment.',
    'meningioma': 'Tumor arising from the meninges surrounding the brain.',
    'pituitary': 'Tumor found in the pituitary gland, which regulates hormones.',
    'notumor': 'No tumor detected. Normal brain scan.'
}
CLASS_ICONS = {
    'glioma': '🔴',
    'meningioma': '🟠',
    'notumor': '🟢',
    'pituitary': '🟡'
}

# Sidebar
with st.sidebar:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("☀️ Light", key="light_btn", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    with col2:
        if st.button("🌙 Dark", key="dark_btn", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Navigation")
    page = st.radio(
        "Select Page:",
        ["🔍 Analyzer", "📊 Dashboard", "📁 Batch Analysis", "ℹ️ Model Info", "⚙️ Settings"]
    )
    st.markdown("---")

# Main header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1>🧠 Brain Tumor MRI Classifier</h1>
    <p>Professional Medical Imaging Analysis System</p>
</div>
""", unsafe_allow_html=True)

if page == "🔍 Analyzer":
    st.markdown("### Single Image Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload MRI Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI scan in JPG or PNG format"
        )
    
    with col2:
        st.markdown("**Image Format**")
        st.info("JPG/PNG, 224×224 recommended")
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📸 Original MRI Scan**")
            st.image(img, use_column_width=True)
        
        # Preprocess
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)
        
        with col2:
            st.markdown("**🔧 Image Info**")
            st.write(f"Filename: {uploaded_file.name}")
            st.write(f"Size: {img.size}")
            st.write(f"Mode: {img.mode}")
            st.write(f"Upload Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prediction
        st.markdown("---")
        st.markdown("### 🔬 Analysis Results")
        
        if st.button("🚀 Analyze MRI Scan", key="analyze_btn", use_container_width=True):
            with st.spinner("🔄 Analyzing scan..."):
                preds = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(preds[0])
                pred_class = CLASS_NAMES[pred_idx]
                confidence = preds[0][pred_idx] * 100
                
                # Store in history
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'filename': uploaded_file.name,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'probabilities': {NAME: float(preds[0][i] * 100) for i, NAME in enumerate(CLASS_NAMES)}
                })
                
                st.success("✅ Analysis Complete!")
        
        # Display results if prediction was made
        if st.session_state.history:
            latest = st.session_state.history[-1]
            pred_class = latest['prediction']
            confidence = latest['confidence']
            
            # Main prediction card
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Class",
                    f"{CLASS_ICONS[pred_class]} {pred_class.upper()}",
                    delta="High Confidence" if confidence > 80 else "Medium Confidence"
                )
            
            with col2:
                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%",
                    delta="Good" if confidence > 75 else "Fair"
                )
            
            with col3:
                st.markdown("**Classification**")
                st.success(CLASS_DESCRIPTIONS[pred_class])
            
            # Probability distribution
            st.markdown("---")
            st.markdown("### 📊 Confidence Distribution")
            
            col_labels, col_bars = st.columns([1, 3])
            
            with col_bars:
                for i, name in enumerate(CLASS_NAMES):
                    prob = latest['probabilities'][name]
                    st.write(f"**{CLASS_ICONS[name]} {name.capitalize()}**")
                    st.progress(prob / 100.0)
                    st.write(f"{prob:.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)
            
            # Grad-CAM visualization
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Explainability Analysis")
            st.info(
                "🔍 Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes "
                "which regions of the MRI scan the model focused on when making the prediction. "
                "Red areas indicate high attention, blue areas indicate low attention."
            )
            
            with st.spinner("🔄 Generating Grad-CAM heatmap..."):
                heatmap = make_gradcam_heatmap(img_array, model, "top_conv")
                heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                original = np.uint8(img_array[0])
                overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img_resized, caption="Original MRI", use_column_width=True)
                with col2:
                    st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), caption="Heatmap", use_column_width=True)
                with col3:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_column_width=True)
            
            # Disclaimer
            st.markdown("---")
            st.warning(
                "⚠️ **Medical Disclaimer**: This is an educational AI system and NOT a substitute for professional medical diagnosis. "
                "Always consult qualified radiologists and medical professionals for clinical decisions."
            )

elif page == "📊 Dashboard":
    st.markdown("### System Dashboard & Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(st.session_state.history), delta="Session Count")
    
    with col2:
        if st.session_state.history:
            avg_confidence = np.mean([h['confidence'] for h in st.session_state.history])
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}%")
        else:
            st.metric("Avg. Confidence", "N/A")
    
    with col3:
        st.metric("Model Accuracy", "86%", delta="Test Set")
    
    with col4:
        st.metric("Classes", len(CLASS_NAMES), delta="Tumor Types")
    
    st.markdown("---")
    
    # Model Statistics
    st.markdown("### 📈 Model Performance Metrics")
    
    metrics_data = {
        'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'Precision': [0.94, 0.81, 0.87, 0.85],
        'Recall': [0.69, 0.79, 0.99, 0.98],
        'F1-Score': [0.79, 0.80, 0.93, 0.91]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Precision Scores**")
        for i, cls in enumerate(metrics_data['Class']):
            st.write(f"{CLASS_ICONS[CLASS_NAMES[i]]} {cls}: {metrics_data['Precision'][i]:.2f}")
    
    with col2:
        st.markdown("**Recall Scores**")
        for i, cls in enumerate(metrics_data['Class']):
            st.write(f"{CLASS_ICONS[CLASS_NAMES[i]]} {cls}: {metrics_data['Recall'][i]:.2f}")
    
    # Session History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📋 Analysis History")
        
        history_data = []
        for h in st.session_state.history:
            history_data.append({
                'Time': h['timestamp'],
                'File': h['filename'],
                'Prediction': h['prediction'],
                'Confidence': f"{h['confidence']:.2f}%"
            })
        
        st.dataframe(history_data, use_container_width=True)
    else:
        st.info("No analysis history yet. Upload and analyze an MRI scan to see results here.")

elif page == "📁 Batch Analysis":
    st.markdown("### Batch MRI Analysis")
    st.info("Upload multiple MRI scans for batch processing and comparative analysis.")
    
    uploaded_files = st.file_uploader(
        "📤 Upload Multiple MRI Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Analyze All Scans", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name} ({idx + 1}/{len(uploaded_files)})")
                
                img = Image.open(file).convert("RGB")
                img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img_resized).astype("float32")
                img_array = np.expand_dims(img_array, axis=0)
                
                preds = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(preds[0])
                pred_class = CLASS_NAMES[pred_idx]
                confidence = preds[0][pred_idx] * 100
                
                results.append({
                    'Filename': file.name,
                    'Prediction': pred_class,
                    'Confidence': f"{confidence:.2f}%",
                    'Timestamp': datetime.now().strftime('%H:%M:%S')
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            st.success("✅ Batch analysis complete!")
            
            st.markdown("---")
            st.markdown("### 📊 Batch Results")
            st.dataframe(results, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Scans", len(results))
            with col2:
                prediction_counts = {}
                for r in results:
                    pred = r['Prediction']
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                most_common = max(prediction_counts, key=prediction_counts.get)
                st.metric("Most Common", f"{CLASS_ICONS[most_common]} {most_common}")
            with col3:
                avg_conf = np.mean([float(r['Confidence'].rstrip('%')) for r in results])
                st.metric("Avg Confidence", f"{avg_conf:.2f}%")

elif page == "ℹ️ Model Info":
    st.markdown("### 🧬 Model Architecture & Information")
    
    st.markdown("#### Neural Network Architecture")
    st.info(
        "**EfficientNetB0 Transfer Learning Model**\n\n"
        "Base Model: EfficientNetB0 (pretrained on ImageNet)\n"
        "Input Size: 224 × 224 × 3\n\n"
        "**Custom Head:**\n"
        "- GlobalAveragePooling2D\n"
        "- Dense(128, relu) + Dropout(0.3)\n"
        "- Dense(4, softmax)\n\n"
        "**Total Parameters:** ~4.2M\n"
        "**Trainable Parameters:** ~132K (head only in phase 1, ~800K in phase 2)"
    )
    
    st.markdown("#### Training Strategy")
    st.markdown(
        """\n        **Phase 1 - Head Training:**
        - Freeze base model completely
        - Train only the custom head layers
        - Learning Rate: 1e-3
        - Epochs: 20
        
        **Phase 2 - Fine-tuning:**
        - Unfreeze last 20 layers of EfficientNetB0
        - Fine-tune entire model at lower learning rate
        - Learning Rate: 1e-5
        - Epochs: 10
        """
    )
    
    st.markdown("#### Dataset Information")
    st.markdown(
        """\n        **Source:** Brain Tumor MRI Dataset (Kaggle)\n
        **Total Images:** ~7,000\n
        **Classes:**
        - Glioma (1,426 images)
        - Meningioma (1,425 images)
        - Pituitary Tumor (1,457 images)
        - No Tumor (1,595 images)
        
        **Test Set Size:** 1,600 images (400 per class)
        """
    )
    
    st.markdown("#### Grad-CAM Explainability")
    st.markdown(
        """\n        **What is Grad-CAM?**
        
        Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations 
        for predictions from CNN-based models.
        
        **How it works:**
        1. Computes gradients of the predicted class w.r.t. activation maps
        2. Weights activations by gradient importance
        3. Creates a heatmap showing model attention regions
        4. Overlays heatmap on original image
        
        **Interpretation:**
        - 🔴 Red: High model attention
        - 🟠 Orange/Yellow: Medium attention
        - 🔵 Blue: Low attention
        """
    )
    
    st.markdown("#### Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Accuracy", "86%", delta="Test Set")
        st.metric("Macro F1-Score", "0.86")
    
    with col2:
        st.metric("Model Size", "~17 MB", delta=".keras format")
        st.metric("Inference Time", "~200ms", delta="Per image")
    
    st.markdown("#### Limitations & Considerations")
    st.warning(
        """⚠️ **Important Limitations:**
        
        - Not clinically validated or approved for medical use
        - Trained on limited dataset; may not generalize to all MRI protocols
        - Glioma recall is lowest (69%); some cases may be missed
        - No cross-institution validation
        - Grad-CAM shows WHERE model looked, not WHY
        - Not a replacement for professional radiologist review
        """
    )

elif page == "⚙️ Settings":
    st.markdown("### Settings & Preferences")
    
    st.markdown("#### Display Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Theme:** " + ("☀️ Light" if st.session_state.theme == "light" else "🌙 Dark"))
        show_predictions = st.checkbox("Show raw prediction scores", value=True)
        show_gradcam = st.checkbox("Show Grad-CAM by default", value=True)
    
    with col2:
        auto_history = st.checkbox("Auto-save analysis history", value=True)
        confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 50)
    
    st.markdown("---")
    st.markdown("#### Model Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Image Size:** 224 × 224")
        st.markdown("**Model Type:** EfficientNetB0")
    
    with col2:
        st.markdown("**Framework:** TensorFlow/Keras")
        st.markdown("**Format:** .keras")
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        """
        **Brain Tumor MRI Classifier**
        
        A professional medical imaging analysis system powered by deep learning.
        
        **Tech Stack:**
        - TensorFlow/Keras - Deep learning
        - Streamlit - Web interface
        - EfficientNetB0 - Model architecture
        - Grad-CAM - Explainability
        - OpenCV - Image processing
        
        **License:** MIT
        
        **Repository:** [GitHub](https://github.com/sejalpandey30/brain-tumor-mri-classifier)
        """
    )
    
    if st.button("🧹 Clear Session History"):
        st.session_state.history = []
        st.success("✅ Session history cleared!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85rem; color: #888;">
        <p>🧠 Brain Tumor MRI Classifier | Professional Medical AI | v2.0</p>
        <p>⚠️ For educational purposes only. Not intended for clinical diagnosis.</p>
        <p>© 2024 | Powered by TensorFlow, Streamlit & EfficientNetB0</p>
    </div>
    """,
    unsafe_allow_html=True
)
