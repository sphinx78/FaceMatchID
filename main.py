import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("Mask_Detector_Model.keras")

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
