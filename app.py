from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLOv8 model
model_path = "weights/best.pt"  # Adjust the path as needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')  # Upload form page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file:
        # Convert file to an image
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Run YOLO prediction
        results = model(img)
        detections = results[0].boxes.data.cpu().numpy()  # Extract detections

        # Format predictions
        predictions = [
            {
                "class": int(box[5]),  # Detected class
                "confidence": float(box[4]),  # Confidence score
                "bbox": box[:4].tolist()  # Bounding box coordinates
            }
            for box in detections
        ]
        return jsonify(predictions)
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
