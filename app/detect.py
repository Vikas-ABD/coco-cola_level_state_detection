import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="custom_ssd_model\custom_model_lite\detect.tflite")
interpreter.allocate_tensors()

labels= ["closedstable", "openlow", "openmore", "openstable", "unseallow", "unsealmore", "unsealstable"]


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Read the image
image_path = "test.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB, as TFLite model expects RGB format
height=640
width=640

# Resize the image to the expected input shape
input_shape = input_details[0]['shape']
resized_image = cv2.resize(image, (640, 640))

# Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
if input_details[0]['dtype'] == np.float32:
    input_data = (np.float32(resized_image) - 127.5) / 127.5
else:
    input_data = np.uint8(resized_image)

# Add a batch dimension
input_data = np.expand_dims(input_data, axis=0)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensors
boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objectss
# Optionally, you can filter out objects with low confidence
threshold = 0.5
for i in range(len(scores)):
    if scores[i] >= threshold:
        ymin, xmin, ymax, xmax = boxes[i]
        class_id = int(classes[i])
        class_name = labels[class_id]
        confidence = scores[i]
        # Convert box coordinates from normalized (0-1) to actual image dimensions
        ymin = int(ymin * height)
        xmin = int(xmin * width)
        ymax = int(ymax * height)
        xmax = int(xmax * width)

        print(xmax)
        
        # Draw bounding box
        cv2.rectangle(resized_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        # Draw label
        label_text = f'{class_name}: {confidence:.2f}'
        label_ymin = max(ymin, 15)  # Adjusted to avoid out-of-bounds errors
        cv2.rectangle(resized_image, (xmin, label_ymin - 15), (xmin + len(label_text) * 7, label_ymin + 5), (255, 255, 255), cv2.FILLED)
        cv2.putText(resized_image, label_text, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



# Create a new figure
plt.figure()


# Show the final image
plt.imshow(resized_image)


plt.show()
