import cv2
import mediapipe as mp
import csv
import os
import math
import numpy as np

DATA_PATH = "data/gestures.csv"
LABEL = input("Enter label (A/B/C/...): ").strip().upper()

os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

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
    for i in range(1,21):
        features.append(np.linalg.norm(pts[i] - palm))

    max_dist = max(features[-20:])
    features[-20:] = [d/max_dist for d in features[-20:]]

    return features

file_exists = os.path.exists(DATA_PATH)

with open(DATA_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        header = [f"f{i}" for i in range(35)] + ["label"]
        writer.writerow(header)

    print("Press SPACE to save sample | ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            features = extract_features(hand.landmark)

        cv2.putText(frame, f"Label: {LABEL}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1)

        if key == 32 and result.multi_hand_landmarks:
            writer.writerow(features + [LABEL])
            print("Saved")

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
