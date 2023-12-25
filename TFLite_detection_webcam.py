
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,640),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    #required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.3)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x640')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

#MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = r'custom_ssd_model/custom_model_lite/detect.tflite'

# Path to label map file
PATH_TO_LABELS = r"custom_ssd_model\custom_model_lite\labelmap.txt"

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
 
# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
#v = 0

while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # Read the single image
    #image_path = "demo.png"
    #image = cv2.imread(image_path)
    #image = cv2.resize(image, (640, 640))
    
    
    
    
    # Capture frame from the camera
    # camera video
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    image=frame_rgb

    # Split the image into four parts from left to right
    height, width, _ = image.shape
    part_width = width // 4

    detection_results = [-1,-1,-1,-1]

    for j in range(4):

        # Extract the current part
        start_x = j * part_width
        end_x = start_x + part_width
        part = image[:, start_x:end_x, :]

        # Resize the part to the expected input shape
        part_resized = cv2.resize(part, (640, 640))
        frame_resized = part_resized
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects
        print(classes)
        # Find the index of the detection with the maximum score
        #max_score_index = scores.index(max(scores))
        # Convert scores to a NumPy array
        scores_np = np.array(scores)

# Find the index of the maximum score
        max_score_index = np.argmax(scores_np)
        
        # Check if the maximum score is above the threshold
        if scores_np[max_score_index] >= min_conf_threshold:
            
            
# Get bounding box coordinates and draw the box on the original image 'part'
            ymin = int(max(1, (boxes[max_score_index][0] * height)))
            xmin = int(max(1, (boxes[max_score_index][1] * width)))
            ymax = int(min(height, (boxes[max_score_index][2] * height)))
            xmax = int(min(width, (boxes[max_score_index][3] * width)))

# Ensure that the coordinates are valid
            if ymin > 0 and xmin > 0 and ymax > 0 and xmax > 0:
                
            
                cv2.rectangle(part, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[max_score_index])]
                label = '%s: %d%%' % (object_name, int(scores[max_score_index] * 100))
                label_ymin = max(ymin, 15)  # Adjusted to avoid out-of-bounds errors
                cv2.rectangle(part, (xmin, label_ymin - 15),
                  (xmin + len(label) * 7, label_ymin + 5), (255, 255, 255), cv2.FILLED)
                cv2.putText(part, label, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Store the detection class index into the array
                detection_results[j]=(int(classes[max_score_index]))

        # Draw framerate in the corner of the frame
        cv2.putText(part, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                    2, cv2.LINE_AA)

        # Display or save the result if needed
        cv2.imshow(f'Object detector - Part {j + 1}', part)
        print(f'Detection results for Part {j + 1}: {detection_results}')
        cv2.waitKey(100) #delay for nect loop image to display
         
    # After the loop
    print(detection_results)
    print(type(detection_results))

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    #v = v + 1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()