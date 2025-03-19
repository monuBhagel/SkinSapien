from flask import Flask, render_template, request
import mysql.connector
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# MySQL Database Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Default user in XAMPP
    password="",  # Leave empty if no password is set
    database="skinsapiens"
)
cursor = db.cursor()

# Load trained model
MODEL_PATH = "/Users/rajeshkumar/skinsapiens-webapp/backend/models/final_skin_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus",
    "Ringworm", "Eczema", "Psoriasis", "Melanoma"
]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", results=None, error="No file uploaded")

    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    preds = model.predict(img)[0]
    top_3_indices = np.argsort(preds)[-3:][::-1]

    results = [{"disease": CLASS_NAMES[i], "confidence": f"{preds[i]:.2f}"} for i in top_3_indices]
    
    # Save prediction to MySQL
    query = """
        INSERT INTO diagnoses (filename, disease_1, confidence_1, disease_2, confidence_2, disease_3, confidence_3)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    values = (file.filename, results[0]["disease"], results[0]["confidence"], 
              results[1]["disease"], results[1]["confidence"], 
              results[2]["disease"], results[2]["confidence"])
    cursor.execute(query, values)
    db.commit()

    os.remove(file_path)
    return render_template("index.html", results=results)

@app.route('/history')
def history():
    cursor.execute("SELECT * FROM diagnoses ORDER BY created_at DESC")
    diagnoses = cursor.fetchall()
    return render_template("history.html", diagnoses=diagnoses)

if __name__ == '__main__':
    app.run(debug=True)
