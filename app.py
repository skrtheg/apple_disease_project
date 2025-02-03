from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("apple_disease_model.keras")

# Define disease classes
classes = ["Apple Black Rot", "Apple Cedar Rust", "Apple Scab"]

# Initialize Flask app
app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Get uploaded image
    img = Image.open(file)  # Open as PIL image
    img = img.resize((128, 128))  # Resize
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)  # Get highest probability index
    result = classes[class_index]  # Get disease name

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
