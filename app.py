from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = './tumor_classifier.h5' 
model = load_model(model_path)

# Define the class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = class_labels[np.argmax(predictions)]

        # Cleanup: Remove the uploaded image
        os.remove(file_path)

        return jsonify({'predicted_class': predicted_class, 'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
