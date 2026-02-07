import base64
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sign Language Recognition API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and MediaPipe


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "gesture_model.pkl")
model = joblib.load(model_path)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


class PredictRequest(BaseModel):
    image: str  # base64 encoded image

class PredictResponse(BaseModel):
    gesture: str
    confidence: float

def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
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

    for i, j, k in angle_indices:
        features.append(calc_angle(pts[i], pts[j], pts[k]))

    palm = pts[0]
    dists = []
    for i in range(1, 21):
        dists.append(np.linalg.norm(pts[i] - palm))

    maxd = max(dists)
    dists = [d / maxd for d in dists]
    features.extend(dists)
    
    return features

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(request.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if not result.multi_hand_landmarks:
            return PredictResponse(gesture="None", confidence=0.0)
        
        # Extract features
        hand = result.multi_hand_landmarks[0]
        features = extract_features(hand.landmark)
        
        # Ensure 35 features
        if len(features) != 35:
            raise HTTPException(status_code=500, detail=f"Feature extraction error: got {len(features)} features")
        
        # Predict
        X = pd.DataFrame([features], columns=model.feature_names_in_)
        probs = model.predict_proba(X)[0]
        idx = np.argmax(probs)
        
        return PredictResponse(
            gesture=model.classes_[idx],
            confidence=float(probs[idx])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
