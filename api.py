# Importing necessary libraries
from flask import Flask, jsonify, request  # Flask for web app, jsonify for JSON responses, request for handling HTTP requests
from tensorflow.keras.models import load_model  # Load pre-trained Keras model
from tensorflow.keras.preprocessing import image  # Image preprocessing tools
import numpy as np  # Numerical operations
import os  # Operating system interaction
from flask_cors import CORS  # Enable Cross-Origin Resource Sharing
from datetime import datetime  # Date and time functions

# Creating a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Specify the folder for uploaded images and create it if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model_path = './check_model.h5'
model = load_model(model_path)

# Define class labels for classification
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image preprocessing function
def preprocess_image(image_path):
    # Load image, resize to target size, convert to array, expand dimensions, and normalize
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Separate function to check an image (not currently used in the main code)
def check(input_img):
    print(" your image is : " + input_img)
    print(input_img)

    img = image.load_img("images/" + input_img, target_size=(224, 224))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)

    print(img)
    output = model.predict(img)

    print(output)
    if output[0][0] == 1:
        status = True
    else:
        status = False

    print(status)
    return status
# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if 'image' is present in the request files
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        # Get the uploaded file
        uploaded_file = request.files['image']
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image with a timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"uploaded_image_{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(image_path)

        # Preprocess the uploaded image
        img_array = preprocess_image(image_path)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class label
        predicted_class = class_labels[np.argmax(predictions)]

        # Return JSON response with prediction results
        return jsonify({'message': 'Image uploaded successfully', 'predicted_class': predicted_class, 'predictions': predictions.tolist(),'Status':check(img_array)}), 200

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
