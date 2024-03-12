from flask import Flask, jsonify, request  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np 
import os 
from flask_cors import CORS  
from datetime import datetime 
import logging
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.utils import CustomObjectScope

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model_path = './check_model.h5'

with CustomObjectScope({'F1Score': F1Score}):
    model = load_model(model_path)

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path):
    # Load image, resize to target size, convert to array, expand dimensions, and normalize
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

def check(predictions):
    predicted_index = np.argmax(predictions)
    status = class_labels[predicted_index]
    logging.info(f"Predicted class is : {status}")
    return status

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        uploaded_file = request.files['image']
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"uploaded_image_{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(image_path)

        img_array = preprocess_image(image_path)

        predictions = model.predict(img_array)
  
        predicted_class = class_labels[np.argmax(predictions)]
        status = check(predictions)
        logging.info(f"Predicted class: {predicted_class}")
        logging.info(f"Predictions: {predictions.tolist()}")
        logging.info(f"Status: {status}")

        return jsonify({'message': 'Image uploaded successfully', 'predicted_class': predicted_class,'result': status}), 200

    except Exception as e:
        logging.error(str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)