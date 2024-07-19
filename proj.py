import cv2
import numpy as np
import requests
import pyttsx3

# Initialize pyttsx3 for TTS
engine = pyttsx3.init()

# Replace with your ESP32-CAM JPEG stream URL
url = 'http://192.168.97.129/640x480.jpg'

# ESP32-CAM IP address (make sure it is static or use mDNS)
esp32_ip = 'http://192.168.97.129'

# Paths to the YOLO files
weights_path = r'C:\Users\pavan\OneDrive\Desktop\mini project\yolo\yolov3.weights'
config_path = r'C:\Users\pavan\OneDrive\Desktop\mini project\yolo\yolov3.cfg'
names_path = r'C:\Users\pavan\OneDrive\Desktop\mini project\yolo\coco.names'

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load object names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def send_to_esp32(text):
    try:
        response = requests.post(f"{esp32_ip}/speak", data={'text': text})
        if response.status_code == 200:
            print(f"Sent to ESP32: {text}")
        else:
            print(f"Failed to send to ESP32: {response.status_code}")
    except Exception as e:
        print(f"Error sending to ESP32: {e}")

while True:
    try:
        # Access the JPEG stream
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Unable to fetch frame from stream, status code: {response.status_code}")
            continue

        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            print(f"Unexpected content type: {content_type}")
            print(response.text)  # Print out the HTML or other text content
            continue  # Skip this iteration

        frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        if frame is None:
            print("Error: Frame is None")
            continue

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_labels = set()
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                detected_labels.add(label)

        # Send detected object labels to ESP32
        for label in detected_labels:
            send_to_esp32(label)

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Exception occurred: {e}")

cv2.destroyAllWindows()
