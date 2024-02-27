from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
from model import TransferLearningModel
import os

app = Flask(__name__)

# Create an instance of the TransferLearningModel class
model = TransferLearningModel(n_classes=1, input_shape=(640, 640, 3))

# Build the model
model = model.build_model()
model_path = r'C:\Users\VIKAS CHEIIURU\OneDrive\Documents\projects_of_vikas_chelluru\Resume_Classification_using_DeepLearning\model_weights\model.h5'

model.load_weights(model_path)
target_size = (640, 640)
class_names = ["Not_Resume","Resume"]

@app.route('/')
def index():
    return render_template('index.html', result_image=None)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image provided", 400

    image = request.files['image']
    if image.filename == '':
        return "No selected image", 400

    # Read the uploaded image and convert it to a numpy array
    image_data = np.frombuffer(image.read(), np.uint8)
    uploaded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Resize the uploaded image to the target size
    resized_image = cv2.resize(uploaded_image, target_size) 

    image_array = np.array(resized_image) / 255.0  # Normalize pixel values to be in the range [0, 1]

    # Add an extra dimension to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    y_pred = model.predict(image_array)

    # Define the threshold
    threshold = 0.5

    # Convert probabilities to class labels using the threshold
    predicted_class_label = 1 if y_pred[0][0] >= threshold else 0

    # Optionally, you can also get the class name if needed
    predicted_class_name = class_names[predicted_class_label]

    print(f"Predicted Class Label: {predicted_class_label}")
    print(f"Predicted Class Name: {predicted_class_name}")

    
    annotated_image = uploaded_image.copy()
    # Draw the predicted class name on top of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_position =text_position = (10, annotated_image.shape[0] - 10) 
    text_color = (255, 0, 0)
    
    cv2.putText(annotated_image, f'Predicted Class: {predicted_class_name}', text_position, font, font_scale, text_color, font_thickness)

    # Save the annotated image for displaying
    cv2.imwrite('app/static/result.jpg', annotated_image)

    return render_template('index.html', result_image='/static/result.jpg')

if __name__ == '__main__':
    app.run(debug=True)
