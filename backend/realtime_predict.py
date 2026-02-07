import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from collections import deque
import math

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=7)

def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def extract_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    features = []

    angle_indices = [
        (0,1,2),(1,2,3),(2,3,4),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20)
    ]

    for i,j,k in angle_indices:
        features.append(calc_angle(pts[i], pts[j], pts[k]))

    palm = pts[0]
    dists = []
    for i in range(1,21):
        dists.append(np.linalg.norm(pts[i]-palm))

    maxd = max(dists)
    dists = [d/maxd for d in dists]

    features.extend(dists)
    return features

print("Real-time ASL recognizer started")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    label = "None"
    conf = 0

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        feats = extract_features(hand.landmark)
        X = pd.DataFrame([feats], columns=model.feature_names_in_)

        probs = model.predict_proba(X)[0]
        idx = np.argmax(probs)

        prediction_buffer.append(model.classes_[idx])

        label = max(set(prediction_buffer), key=prediction_buffer.count)
        conf = probs[idx]*100

        cv2.rectangle(frame, (50,50), (350,150), (0,255,0), 2)

        cv2.putText(frame, f"Gesture: {label}", (70,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Confidence: {conf:.1f}%", (70,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("ASL Landmark Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
