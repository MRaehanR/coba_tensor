import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import time

# Load the model
model = hub.load("./openimages_v4_ssd_mobilenet_v2_1").signatures["default"]

# Initialize color codes for detected classes
colorcodes = {}

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    
    # Draw rectangle for bounding box
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=2)
    
    # Draw filled rectangle for label background
    label_height = int(20)
    cv2.rectangle(image, (left, top - label_height), (right, top), color=color, thickness=-1)
    
    # Add label text
    font_scale = 0.5
    thickness = 1
    cv2.putText(image, namewithscore, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def draw(image, boxes, classnames, scores):
    boxes_idx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.2)
    for i in boxes_idx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        
        # Assign colors for each class
        if classname not in colorcodes:
            colorcodes[classname] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = colorcodes[classname]
        
        # Format label with class name and score
        namewithscore = f"{classname}:{int(scores[i] * 100)}%"
        
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
    
    return image

# Open video stream
video = cv2.VideoCapture(0)
video.set(3, 640)  # Set width
video.set(4, 480)  # Set height

# FPS calculation
prev_time = time.time()

while True:
    ret, img = video.read()
    if not ret:
        break
    
    img = cv2.resize(img, (900, 700))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]
    
    # Perform detection
    detection = model(img_tensor)
    result = {key: value.numpy() for key, value in detection.items()}
    
    # Draw detection boxes
    image_with_boxes = draw(img, result['detection_boxes'], result['detection_class_entities'], result['detection_scores'])
    
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(image_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Detection", image_with_boxes)
    
    # Break on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
