import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask app setup
app = Flask(__name__)
model = load_model('../src/pneumonia_cnn_model.h5')  # Adjust path if needed
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Image dimensions (match your training)
img_width, img_height = 150, 150

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_tensor)
    result = 'Pneumonia Detected' if prediction[0][0] > 0.5 else 'Normal'

    return render_template('index.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
