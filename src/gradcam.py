"""
Grad-CAM explainability for the Brain Tumor MRI Classifier.
Handles the nested EfficientNetB0 sub-model structure.
"""

import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model):
    """Finds the last conv layer name inside the nested efficientnetb0 sub-model."""
    base = model.get_layer("efficientnetb0")
    for layer in reversed(base.layers):
        if "conv" in layer.name.lower():
            return layer.name
    return None


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
