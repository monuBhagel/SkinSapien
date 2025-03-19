from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "/Users/rajeshkumar/skinsapiens-webapp/backend/models/final_skin_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels (update with actual class names)
CLASS_NAMES = ["Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus","Ringworm", "Eczema", "Psoriasis", "Melanoma"]

# Image preprocessing function
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image file")
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", file.filename)
        file.save(file_path)
        
        img = preprocess_image(file_path)
        if img is None:
            return jsonify({"error": "Invalid or corrupt image file"}), 400
        
        if model is None:
            return jsonify({"error": "Model failed to load"}), 500
        
        preds = model.predict(img)[0]
        top_3_indices = np.argsort(preds)[-3:][::-1]
        
        results = [{"disease": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top_3_indices]
        os.remove(file_path)  # Clean up temp file
        
        return jsonify(results)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
