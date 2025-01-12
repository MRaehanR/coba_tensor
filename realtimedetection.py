import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import random

# Load the TensorFlow Lite model (converted for edge devices)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# TensorFlow Lite input-output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

colorcodes = {}

def preprocess_image(image):
    """Preprocess image for TFLite model."""
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image / 255.0, axis=0).astype(np.float32)
    return image

def drawbox(image, ymin, xmin, ymax, xmax, label, color):
    """Draw bounding box and label on the image."""
    h, w, _ = image.shape
    left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(image, (left, top - label_size[1] - 10), (left + label_size[0], top), color, -1)
    cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_boxes(image, boxes, classes, scores, threshold=0.3):
    """Draw all valid bounding boxes."""
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            class_name = classes[i].decode('utf-8') if isinstance(classes[i], bytes) else str(classes[i])
            score = int(scores[i] * 100)
            label = f"{class_name}: {score}%"
            if class_name not in colorcodes:
                colorcodes[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            drawbox(image, ymin, xmin, ymax, xmax, label, colorcodes[class_name])
    return image

# Video input
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_data = preprocess_image(frame)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Detection boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Detection classes
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Detection scores

    # Draw results on the frame
    frame = draw_boxes(frame, boxes, classes, scores)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
