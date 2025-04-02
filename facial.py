{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import cv2\
from ultralytics import YOLO\
\
# Load the YOLOv8 model\
model = YOLO('/home/pi/best.pt')  # Path to your downloaded model file\
\
# Initialize the video stream (use a camera or load a video file)\
cap = cv2.VideoCapture(0)  # Use camera (or provide video file path)\
\
while cap.isOpened():\
    ret, frame = cap.read()\
\
    if not ret:\
        break\
\
    # Run the model on the frame (predictions)\
    results = model(frame)\
\
    # Extract prediction details (boxes, labels, and confidence scores)\
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()\
    pred_labels = results[0].boxes.cls.cpu().numpy()\
    pred_conf = results[0].boxes.conf.cpu().numpy()\
\
    # Display the results on the image\
    for i in range(len(pred_boxes)):\
        x1, y1, x2, y2 = pred_boxes[i]\
        label = int(pred_labels[i])  # Class label\
        confidence = float(pred_conf[i])  # Confidence score\
\
        # Draw bounding box\
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)\
        cv2.putText(frame, f'\{model.names[label]\} \{confidence:.2f\}', (int(x1), int(y1)-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)\
\
    # Show the image with bounding boxes\
    cv2.imshow("YOLOv8 Inference", frame)\
\
    # Press 'q' to quit the video stream\
    if cv2.waitKey(1) & 0xFF == ord('q'):\
        break\
\
cap.release()\
cv2.destroyAllWindows()\
}