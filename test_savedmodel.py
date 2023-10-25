import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Set up the paths and configurations for the model and label map
PATH_TO_MODEL_DIR = 'custom_ssd_model'
PATH_TO_LABELS = 'custom_ssd_model\custom_model_lite\bottle_label_map.pbtxt'
MIN_CONF_THRESH = 0.60

# Load the model
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Initialize the list to store detected classes
detected_classes_list = []

# Open a video capture stream (you can change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    input_tensor = tf.convert_to_tensor(image_expanded, dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # Process the detections and update the list
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detected_classes = list(set([category_index[i]['name'] for i in detections['detection_classes']]))
    detected_classes_list.append(detected_classes)

    # Visualization (you can modify this according to your needs)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=MIN_CONF_THRESH,
        agnostic_mode=False
    )

    # Show the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()

    

   


