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

 
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(img_w - 1, endX), min(img_h - 1, endY)

            x, y, w, h = startX, startY, endX - startX, endY - startY

            face = frame[startY:endY, startX:endX]
            
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face = cv2.resize(face, (256, 256))
            face = face / 256.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)

            if prediction[0][0] > 0.5:
                label = "Invalid: Movement Detected"
                color = (0, 0, 255) # Red
            else:
                label = "Valid: Perfect for ID"
                color = (0, 255, 0) # Green

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, label, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

    cv2.imshow("ID Photo Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()