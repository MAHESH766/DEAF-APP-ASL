import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define ASL gestures to record
GESTURES = ["hello", "yes", "no", "thank you", "please"]
data = []
labels = []

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'c' to capture a gesture, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, "Press 'c' to capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print("Select a gesture:", GESTURES)
                label = input("Enter gesture name: ").strip().lower()
                if label in GESTURES:
                    data.append(landmarks)
                    labels.append(label)
                    print(f"Captured gesture: {label}")
                else:
                    print("Invalid gesture.")

    cv2.imshow("ASL Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data to a CSV file
df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("asl_data.csv", index=False)
print("Data saved to asl_data.csv")
