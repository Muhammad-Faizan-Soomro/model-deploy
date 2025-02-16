import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
from PIL import Image
import json
import os
from flask import render_template

tf.keras.config.enable_unsafe_deserialization()

# Required for loading custom model
@register_keras_serializable()
def custom_preprocess(x):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(x)

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model('mobilenetv2_model.keras')
with open('EfficientNetB0_labels.json', 'r') as f:
    class_names = json.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Process image
        image = Image.open(request.files['image'].stream).convert('RGB')
        image = image.resize((224, 224))  # Match model's expected input
        img_array = np.array(image, dtype=np.float32)  # Maintain 0-255 range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        
        return jsonify({
            'class': class_names[predicted_idx],
            'confidence': float(predictions[0][predicted_idx]),
            'all_predictions': dict(zip(class_names, predictions[0].astype(float)))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500