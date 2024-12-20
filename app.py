from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and label encoder
model = load_model('models/CNN_Covid19_Xray_Version.h5')  # Replace with your model path
le = pickle.load(open("models/Label_encoder.pkl", 'rb'))  # Load the label encoder

# Path to store uploaded images 
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to preprocess and predict the label of an image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    predicted_label = le.inverse_transform([predicted_index])[0]
    return predicted_label, confidence_score, predictions[0]

# Helper function to calculate various metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1, specificity, sensitivity

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image and make predictions
        predicted_label, confidence_score, predictions = process_image(file_path)

        # Example ground truth and predictions (replace with real data in production)
        y_true = [0, 1, 0, 1]  # Example ground truth
        y_pred = [0, 1, 1, 1]  # Example predicted labels
        
        # Calculate metrics
        accuracy, precision, recall, f1, specificity, sensitivity = calculate_metrics(y_true, y_pred)

        return render_template(
            'result.html',
            image_path=url_for('uploaded_file', filename=filename),
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=specificity,
            sensitivity=sensitivity,
        )  
@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)