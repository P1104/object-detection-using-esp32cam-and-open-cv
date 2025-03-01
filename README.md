# Object Detection using ESP32-CAM and OpenCV with TTS

This project demonstrates how to perform object detection using an ESP32-CAM and OpenCV, and provide audio output for the detected objects using the `pyttsx3` text-to-speech library.

## Features

- Real-time object detection using YOLO (You Only Look Once) model
- Voice feedback for detected objects
- Integration with ESP32-CAM for capturing video stream

## Prerequisites

- Python 3.6 or higher
- OpenCV
- Requests library
- pyttsx3 library
- ESP32-CAM with firmware to provide a JPEG stream
- YOLO model files (weights, configuration, and class names)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/Object-detection-using-OpenCV-and-ESP32-CAM.git
   cd Object-detection-using-OpenCV-and-ESP32-CAM
Create and activate a virtual environment (optional but recommended):

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install required packages:

sh
Copy code
pip install opencv-python numpy requests pyttsx3
Download YOLO model files:

yolov3.weights
yolov3.cfg
coco.names
Place these files in the project directory.

Usage
Configure the ESP32-CAM URL:

Replace url with the URL of your ESP32-CAM JPEG stream in the script.
Run the script:

sh
Copy code
python object_detection.py
View the output:

The script will display the video stream with detected objects.
Detected objects will be announced using TTS.
