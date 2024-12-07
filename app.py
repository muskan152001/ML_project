from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLOv8 model
model_path = "weights/best.pt"  # Adjust the path if needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')  # Render the upload form

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

        # Draw bounding boxes and labels on the image
        for box in detections:
            x1, y1, x2, y2, conf, cls = box[:6]
            label = f"{model.names[int(cls)]} {conf:.2f}"
            # Draw the rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Add the label with a background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(x1), int(y1) - text_height - baseline), (int(x1) + text_width, int(y1)), (255, 0, 0), -1)
            cv2.putText(img, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save the output image with bounding boxes and labels
        output_path = "static/output.jpg"
        cv2.imwrite(output_path, img)

        # Return the output image path
        return jsonify({"image_url": output_path})

    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
