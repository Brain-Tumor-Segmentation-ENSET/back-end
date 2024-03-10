import os
from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"uploaded_image_{timestamp}.jpg"

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    print('Image uploaded successfully')
    return jsonify({'message': 'Image uploaded successfully', 'image_path': image_path}), 200


if __name__ == '__main__':
    app.run(debug=True)
