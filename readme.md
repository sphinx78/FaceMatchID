# Official Document Photo Assistant

This is an AI-powered webcam assistant designed to help capture perfect profile photos for official documents (like passports, IDs, and visas). It uses a custom deep learning model to detect when you are holding completely still with a neutral facial expression.

## Features
- **Real-time Face Detection**: Uses OpenCV's highly accurate ResNet SSD Face Detector.
- **Stillness Detection**: Evaluates your facial expression and movement in real-time.
- **Visual Feedback**:
  - **Green Box (Valid: Perfect for ID)**: You are holding still and maintaining a neutral expression.
  - **Red Box (Invalid: Movement Detected)**: You are moving, talking, or smiling (which is often rejected for official IDs).

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- TensorFlow / Keras (`tensorflow`)
- NumPy (`numpy`)

## Setup
1. Ensure you have the required dependencies installed. You can install them via pip:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure the model (`Mask_Detector_Model.keras`) and the face detector files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) are in the project root directory.

## Usage
Run the main script to start the webcam assistant:
```bash
python detect.py
```

- A webcam window named "ID Photo Assistant" will open.
- Hold still and keep a neutral face to get a "Valid" green box.
- Press `q` to quit the application.

## How it Works
The assistant re-purposes a convolutional neural network (CNN) model to monitor the user's face. By passing the real-time webcam feed through OpenCV's DNN module and evaluating the extracted facial crops with the Keras model, the application provides instant feedback on posture and expression suitability.
