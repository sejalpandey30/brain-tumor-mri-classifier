import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import load_model
from src.gradcam import make_gradcam_heatmap
from src.ui_styling import apply_theme_css, get_theme_colors
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%234A90E2'/><path d='M50 20 Q60 35 65 50 Q60 60 50 65 Q40 60 35 50 Q40 35 50 20' fill='white'/></svg>",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "history" not in st.session_state:
    st.session_state.history = []
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

# Load model
@st.cache_resource
def load_ml_model():
    return load_model()

model = load_ml_model()

# Constants
IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DESCRIPTIONS = {
    'glioma': 'Malignant tumor originating in glial cells. Requires aggressive multimodal treatment including surgery, radiation, and chemotherapy.',
    'meningioma': 'Tumor arising from meninges (membranes surrounding brain/spinal cord). Usually benign but requires careful monitoring and treatment.',
    'pituitary': 'Adenoma in the pituitary gland. Affects hormone production and regulation. Endocrine management is essential.',
    'notumor': 'No abnormal findings detected. Normal brain tissue without pathological lesions or masses.'
}

# Apply theme CSS
apply_theme_css(st.session_state.theme)

# ============================================================================
# PROFESSIONAL HEADER
# ============================================================================
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])

with header_col2:
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
            <svg width="44" height="44" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" stroke="#4A90E2" stroke-width="2.5"/>
                <path d="M20 8 L28 16 L26 24 L14 24 L12 16 Z" fill="#4A90E2" opacity="0.85"/>
                <circle cx="20" cy="18" r="3" fill="#FF6B6B"/>
            </svg>
        </div>
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.5px;">
            Brain Tumor MRI Classifier
        </h1>
        <p style="margin: 0.7rem 0 0 0; font-size: 1.05rem; color: #666; font-weight: 500;">
            Advanced Medical Imaging Analysis System
        </p>
    </div>
    """, unsafe_allow_html=True)

# Theme toggle button
col1, col2, col3 = st.columns([10, 1, 1])
with col3:
    if st.session_state.theme == "light":
        if st.button("🌙", key="dark_toggle", help="Switch to Dark Mode", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    else:
        if st.button("☀️", key="light_toggle", help="Switch to Light Mode", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()

st.markdown("""
<div style="height: 2px; background: linear-gradient(90deg, #4A90E2 0%, #357ABD 50%, #4A90E2 100%); margin: 1.5rem 0; border-radius: 1px;"></div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT - ANALYSIS PAGE
# ============================================================================

st.markdown("""
<div style="margin-bottom: 2.5rem;">
    <h2 style="font-size: 1.35rem; font-weight: 600; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 12px;">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <span>Upload MRI Scan</span>
    </h2>
    <p style="color: #666; font-size: 0.97rem; margin: 0;">Select a high-resolution brain MRI image for comprehensive analysis. Supported formats: JPG, PNG</p>
</div>
""", unsafe_allow_html=True)

upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Select Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with info_col:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E8F0F8 0%, #F5F7FA 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #4A90E2; height: 100%;">
        <p style="font-size: 0.88rem; margin: 0; color: #333; font-weight: 600;">RECOMMENDED SPECS</p>
        <p style="font-size: 0.85rem; margin: 0.4rem 0 0 0; color: #666; line-height: 1.5;">224×224 pixels or higher for optimal accuracy</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    st.markdown("""<div style="height: 1px; background: linear-gradient(90deg, transparent 0%, #E0E0E0 50%, transparent 100%); margin: 2.5rem 0;"></div>""", unsafe_allow_html=True)
    
    # Image preview section
    st.markdown("""
    <h2 style="font-size: 1.35rem; font-weight: 600; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 12px;">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <circle cx="8.5" cy="8.5" r="1.5"></circle>
            <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
        <span>Image Preview & Details</span>
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True, caption="Uploaded MRI Scan", output_format="auto")
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #E8F0F8 0%, #F5F7FA 100%); padding: 1.8rem; border-radius: 10px; height: 100%;">
            <p style="font-weight: 650; margin: 0 0 1.2rem 0; color: #1F2937; font-size: 0.98rem;">FILE INFORMATION</p>
            <div style="font-size: 0.93rem; line-height: 1.8;">
                <p style="margin: 0.6rem 0; color: #555;"><span style="font-weight: 600; color: #333;">Filename:</span><br>{uploaded_file.name}</p>
                <p style="margin: 0.6rem 0; color: #555;"><span style="font-weight: 600; color: #333;">Dimensions:</span><br>{img.width} × {img.height} pixels</p>
                <p style="margin: 0.6rem 0; color: #555;"><span style="font-weight: 600; color: #333;">Format:</span><br>{img.format}</p>
                <p style="margin: 0.6rem 0; color: #555;"><span style="font-weight: 600; color: #333;">File Size:</span><br>{uploaded_file.size / 1024:.1f} KB</p>
                <p style="margin: 0.6rem 0; color: #555;"><span style="font-weight: 600; color: #333;">Uploaded:</span><br>{datetime.now().strftime('%H:%M:%S')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Preprocess image
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    
    # Analysis button
    st.markdown("""<div style="height: 2rem;"></div>""", unsafe_allow_html=True)
    
    col_btn, col_space = st.columns([1.2, 8.8])
    with col_btn:
        analyze_clicked = st.button(
            "Analyze Scan",
            use_container_width=True,
            key="analyze_btn"
        )
    
    if analyze_clicked:
        with st.spinner("Processing MRI scan with deep learning model..."):
            preds = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(preds[0])
            pred_class = CLASS_NAMES[pred_idx]
            confidence = preds[0][pred_idx] * 100
            
            # Store prediction
            st.session_state.current_prediction = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': uploaded_file.name,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': {NAME: float(preds[0][i] * 100) for i, NAME in enumerate(CLASS_NAMES)},
                'image': img,
                'img_array': img_array
            }
            
            # Add to history
            st.session_state.history.append(st.session_state.current_prediction.copy())
    
    # Display results if available
    if st.session_state.current_prediction:
        st.markdown("""<div style="height: 2rem;"></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, #4A90E2 0%, transparent 100%); margin: 1.5rem 0;"></div>""", unsafe_allow_html=True)
        
        pred = st.session_state.current_prediction
        pred_class = pred['prediction']
        confidence = pred['confidence']
        
        # Results header
        st.markdown("""
        <h2 style="font-size: 1.35rem; font-weight: 600; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 12px;">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            <span>Analysis Results</span>
        </h2>
        """, unsafe_allow_html=True)
        
        # Main prediction card with dynamic coloring
        confidence_level = "High" if confidence > 75 else "Medium" if confidence > 60 else "Low"
        color = "#10B981" if confidence > 75 else "#F59E0B" if confidence > 60 else "#EF4444"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}15 0%, {color}08 100%); 
                    border: 2px solid {color}; border-radius: 14px; padding: 2.2rem; margin-bottom: 2.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2.5rem;">
                <div style="text-align: center;">
                    <p style="font-size: 0.82rem; color: #666; margin: 0 0 0.7rem 0; font-weight: 650; letter-spacing: 0.5px;">DIAGNOSIS</p>
                    <p style="font-size: 2rem; font-weight: 750; color: {color}; margin: 0; text-transform: uppercase;">{pred_class}</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 0.82rem; color: #666; margin: 0 0 0.7rem 0; font-weight: 650; letter-spacing: 0.5px;">CONFIDENCE SCORE</p>
                    <p style="font-size: 2rem; font-weight: 750; color: {color}; margin: 0;">{confidence:.1f}%</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 0.82rem; color: #666; margin: 0 0 0.7rem 0; font-weight: 650; letter-spacing: 0.5px;">CONFIDENCE LEVEL</p>
                    <p style="font-size: 1.4rem; font-weight: 700; color: {color}; margin: 0;">{confidence_level}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Description box
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #E8F0F8 0%, #F5F7FA 100%); padding: 1.6rem; 
                    border-radius: 10px; border-left: 4px solid #4A90E2; margin-bottom: 2.2rem;">
            <p style="font-weight: 680; color: #1F2937; margin: 0 0 0.8rem 0; font-size: 0.96rem; letter-spacing: 0.3px;">DIAGNOSIS DESCRIPTION</p>
            <p style="color: #555; line-height: 1.7; margin: 0; font-size: 0.96rem;">{CLASS_DESCRIPTIONS[pred_class]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence distribution
        st.markdown("""
        <h3 style="font-size: 1.15rem; font-weight: 650; margin-bottom: 1.8rem; display: flex; align-items: center; gap: 10px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="12" y1="2" x2="12" y2="22"></line>
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
            </svg>
            <span>Prediction Confidence Distribution</span>
        </h3>
        """, unsafe_allow_html=True)
        
        # Confidence bars
        for i, name in enumerate(CLASS_NAMES):
            prob = pred['probabilities'][name]
            bar_color = "#4A90E2" if name == pred_class else "#D0D5E0"
            
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.6rem;">
                    <span style="font-weight: 600; color: #333; font-size: 0.97rem; text-transform: capitalize;">{name}</span>
                    <span style="font-weight: 700; color: {bar_color}; font-size: 0.97rem;">{prob:.1f}%</span>
                </div>
                <div style="background: #E0E0E0; height: 10px; border-radius: 5px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);">
                    <div style="background: linear-gradient(90deg, {bar_color} 0%, {bar_color}CC 100%); 
                                height: 100%; width: {prob}%; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Grad-CAM section
        st.markdown("""<div style="height: 1px; background: linear-gradient(90deg, transparent 0%, #E0E0E0 50%, transparent 100%); margin: 2.5rem 0;"></div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="font-size: 1.15rem; font-weight: 650; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M8 12h8M12 8v8"></path>
            </svg>
            <span>Explainability Analysis (Grad-CAM)</span>
        </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="color: #666; font-size: 0.97rem; margin-bottom: 1.8rem; line-height: 1.7;">
            Grad-CAM visualization shows regions where the neural network focused when making the prediction. 
            <span style="color: #333; font-weight: 600;">Red regions</span> indicate high model attention, 
            <span style="color: #333; font-weight: 600;">blue regions</span> indicate low attention.
        </p>
        """, unsafe_allow_html=True)
        
        with st.spinner("Generating attention heatmap..."):
            heatmap = make_gradcam_heatmap(pred['img_array'], model, "top_conv")
            heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            original = np.uint8(pred['img_array'][0])
            overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img_resized, caption="Original Scan", use_column_width=True, output_format="auto")
            
            with col2:
                st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), caption="Attention Heatmap", use_column_width=True, output_format="auto")
            
            with col3:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay Analysis", use_column_width=True, output_format="auto")
        
        st.markdown("""<div style="height: 2rem;"></div>""", unsafe_allow_html=True)
        
        # Analysis history section
        st.markdown("""
        <h3 style="font-size: 1.15rem; font-weight: 650; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 10px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <span>Session Analysis Summary</span>
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Scans Analyzed",
                len(st.session_state.history),
                delta="In Session",
                delta_color="off"
            )
        
        with col2:
            avg_confidence = np.mean([h['confidence'] for h in st.session_state.history])
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1f}%",
                delta="Overall",
                delta_color="off"
            )
        
        with col3:
            from collections import Counter
            predictions = [h['prediction'] for h in st.session_state.history]
            unique_classes = len(set(predictions))
            st.metric(
                "Classes Found",
                unique_classes,
                delta="Unique Types",
                delta_color="off"
            )
        
        with col4:
            most_common = Counter(predictions).most_common(1)[0][0]
            st.metric(
                "Most Common",
                most_common.capitalize(),
                delta="Result",
                delta_color="off"
            )
        
        st.markdown("""<div style="height: 1.5rem;"></div>""", unsafe_allow_html=True)
        
        # Medical disclaimer
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1.5px solid #FCD34D; border-radius: 10px; padding: 1.4rem;">
            <div style="display: flex; gap: 1.2rem;">
                <div style="font-size: 1.8rem; line-height: 1; min-width: 2rem;">⚠️</div>
                <div>
                    <p style="font-weight: 680; color: #92400E; margin: 0 0 0.6rem 0; font-size: 0.97rem; letter-spacing: 0.3px;">MEDICAL DISCLAIMER</p>
                    <p style="color: #B45309; margin: 0; font-size: 0.93rem; line-height: 1.6;">
                        This AI system is for educational and research purposes only. It is <strong>NOT FDA-approved</strong> and should not be used 
                        for clinical diagnosis or treatment decisions. Always consult qualified radiologists and medical professionals for all medical matters.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3.5rem; margin-bottom: 1.2rem; opacity: 0.8;">📋</div>
        <p style="font-size: 1.15rem; color: #333; font-weight: 600; margin: 0 0 0.6rem 0;">No MRI Scan Uploaded</p>
        <p style="color: #999; font-size: 0.98rem; margin: 0;">Upload a brain MRI image above to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PROFESSIONAL FOOTER
# ============================================================================
st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent 0%, #4A90E2 50%, transparent 100%); margin: 3.5rem 0 2.5rem 0;"></div>""", unsafe_allow_html=True)

st.markdown("""
<footer style="padding: 2.5rem 0; text-align: center;">
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2.5rem; margin-bottom: 2rem; padding-bottom: 2rem; border-bottom: 1px solid #E0E0E0;">
        <div>
            <p style="font-weight: 700; color: #1F2937; margin: 0 0 0.6rem 0; font-size: 0.93rem; letter-spacing: 0.3px; text-transform: uppercase;">Technology Stack</p>
            <p style="color: #666; margin: 0; font-size: 0.92rem; line-height: 1.6;">TensorFlow • EfficientNetB0<br/>Streamlit • Grad-CAM</p>
        </div>
        <div>
            <p style="font-weight: 700; color: #1F2937; margin: 0 0 0.6rem 0; font-size: 0.93rem; letter-spacing: 0.3px; text-transform: uppercase;">Performance Metrics</p>
            <p style="color: #666; margin: 0; font-size: 0.92rem; line-height: 1.6;">86% Accuracy • F1: 0.86<br/>4 Tumor Classifications</p>
        </div>
        <div>
            <p style="font-weight: 700; color: #1F2937; margin: 0 0 0.6rem 0; font-size: 0.93rem; letter-spacing: 0.3px; text-transform: uppercase;">Application Status</p>
            <p style="color: #666; margin: 0; font-size: 0.92rem; line-height: 1.6;">Research Grade • Educational<br/>Not for Clinical Use</p>
        </div>
    </div>
    <div>
        <p style="color: #999; font-size: 0.88rem; margin: 0; letter-spacing: 0.2px;">© 2024 Brain Tumor MRI Classifier. Licensed under MIT License.</p>
        <p style="color: #999; font-size: 0.88rem; margin: 0.4rem 0 0 0; letter-spacing: 0.2px;">Research and educational purposes only. Not approved for clinical use.</p>
    </div>
</footer>
""", unsafe_allow_html=True)
