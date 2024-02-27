from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import 

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through YOLOv8 object detection network and returns and array
        of bounding boxes.
        :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)


# def detect_objects_on_image(buf):
#     """
#     Function receives an image,
#     passes it through YOLOv8 neural network
#     and returns an array of detected objects
#     and their bounding boxes
#     :param buf: Input image file stream
#     :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
#     """
#     model = YOLO("best.pt")
#     results = model.predict(Image.open(buf))
#     result = results[0]
#     output = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = [
#             round(x) for x in box.xyxy[0].tolist()
#         ]
#         class_id = box.cls[0].item()
#         prob = round(box.conf[0].item(), 2)
#         output.append([
#             x1, y1, x2, y2, result.names[class_id], prob
#         ])
#     return output


def detect_objects_on_image(buf):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    input_data = np.array(Image.open(buf).resize((width, height)))
    input_data = np.expand_dims(input_data, axis=0)

    # Normalize the input data (if needed)
    input_mean = 127.5
    input_std = 127.5
    input_data = (input_data - input_mean) / input_std

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output to get bounding boxes and labels
    # (Note: This part will depend on the structure of your model's output tensor)

    # Example post-processing for a model that outputs bounding boxes and classes:
    boxes = output_data[..., :4]  # Assuming boxes are in the first four elements of the output tensor
    scores = output_data[..., 4:]  # Assuming scores are in the remaining elements
    classes = np.argmax(scores, axis=-1)  # Assuming classes are predicted by the highest score
    boxes = convert_boxes(boxes)  # Convert boxes to proper format

    # Assuming convert_boxes function converts boxes to x1, y1, x2, y2 format

    # Return bounding boxes and labels
    return boxes, classes

# Define a function to convert box coordinates if needed
def convert_boxes(boxes):
    # Convert boxes to x1, y1, x2, y2 format (if needed)
    # For example, if the TFLite model outputs normalized box coordinates:
    return boxes * np.array([width, height, width, height])

if __name__ == "__main__":
    app.run(debug=True)
