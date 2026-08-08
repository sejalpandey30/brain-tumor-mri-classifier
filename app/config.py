"""Configuration settings for the application."""

# Image settings
IMG_SIZE = 224
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']

# Model settings
MODEL_PATH = "results/brain_tumor_model.keras"
CLASS_INDICES_PATH = "results/class_indices.json"

# Class information
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

CLASS_DESCRIPTIONS = {
    'glioma': 'Brain tumor originating in glial cells. Requires aggressive treatment including surgery, radiation, and chemotherapy.',
    'meningioma': 'Tumor arising from the meninges (membranes surrounding the brain and spinal cord). Usually benign but can cause complications.',
    'pituitary': 'Tumor found in the pituitary gland, which regulates hormone production. Can affect hormone levels and various bodily functions.',
    'notumor': 'No tumor detected. Brain scan appears normal without abnormal growths or masses.'
}

CLASS_ICONS = {
    'glioma': '🔴',
    'meningioma': '🟠',
    'notumor': '🟢',
    'pituitary': '🟡'
}

# Grad-CAM settings
GRAD_CAM_LAYER = "top_conv"
GRAD_CAM_OPACITY = 0.4

# UI settings
THEME_OPTIONS = ['light', 'dark']
DEFAULT_THEME = 'light'

# Model metrics (from training)
MODEL_METRICS = {
    'test_accuracy': 0.86,
    'macro_f1': 0.86,
    'per_class': {
        'glioma': {'precision': 0.94, 'recall': 0.69, 'f1': 0.79},
        'meningioma': {'precision': 0.81, 'recall': 0.79, 'f1': 0.80},
        'notumor': {'precision': 0.87, 'recall': 0.99, 'f1': 0.93},
        'pituitary': {'precision': 0.85, 'recall': 0.98, 'f1': 0.91}
    }
}

# Dataset information
DATASET_INFO = {
    'name': 'Brain Tumor MRI Dataset',
    'source': 'Kaggle',
    'url': 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
    'total_images': 7000,
    'classes': {
        'glioma': 1426,
        'meningioma': 1425,
        'pituitary': 1457,
        'notumor': 1595
    },
    'test_set_size': 1600
}
