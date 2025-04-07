from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model safely
model_path = "asl_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None  # Prevent NameError if model is missing

# Load label encoder safely
data_path = "asl_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    encoder = LabelEncoder()
    encoder.fit(df["label"].values)
else:
    encoder = None  # Prevent NameError

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        recognized_gesture = "Detecting..."

        if result.multi_hand_landmarks and model is not None and encoder is not None:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks).reshape(1, -1)

                # Predict gesture only if model exists
                prediction = model.predict(landmarks)
                gesture_index = np.argmax(prediction)
                recognized_gesture = encoder.inverse_transform([gesture_index])[0]

                cv2.putText(frame, f"ASL: {recognized_gesture}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure templates/index.html exists

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
