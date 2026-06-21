# Conveyor Belt Object Detection

A real-time object detection system that detects and classifies objects using a webcam. Built with Python, OpenCV, and a pre-trained SSD MobileNet model.

## Features
- Real-time object detection using webcam
- Detects 90+ object classes
- Bounding boxes with confidence scores
- Simple GUI built with Tkinter
- Non-maximum suppression to avoid duplicate detections

## Tech Stack
- Python 3.x
- OpenCV
- NumPy
- Pillow (PIL)
- Tkinter

## Setup & Installation

```bash
pip install -r requirements.txt

Download Required Model Files

frozen_inference_graph.pb
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
objects.txt
