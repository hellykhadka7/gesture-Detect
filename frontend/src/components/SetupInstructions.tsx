import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Terminal, Folder, FileCode, CheckCircle2 } from 'lucide-react';

export const SetupInstructions = () => {
  const [isOpen, setIsOpen] = useState(false);

  const folderStructure = `
sign-language-app/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── requirements.txt  # Python dependencies
│   ├── gesture_model.pkl # Your trained model
│   └── data/
│       └── gestures.csv  # Training data
├── collect_data.py       # Data collection script
├── train_model.py        # Model training script
└── realtime_predict.py   # Original prediction script
  `.trim();

  const requirementsTxt = `
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
mediapipe==0.10.9
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.4.0
joblib==1.3.2
opencv-python==4.9.0.80
  `.trim();

  const mainPy = `
import base64
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
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
model = joblib.load("gesture_model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, static_image_mode=True)

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
  `.trim();

  return (
    <div className="glass-card">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-primary" />
          <span className="font-semibold">Backend Setup Instructions</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-5 h-5 text-muted-foreground" />
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-6 space-y-6 text-sm">
              {/* Step 1: Folder Structure */}
              <div className="space-y-2">
                <div className="flex items-center gap-2 font-medium">
                  <Folder className="w-4 h-4 text-primary" />
                  Step 1: Create folder structure
                </div>
                <pre className="p-4 rounded-lg bg-secondary overflow-x-auto font-mono text-xs">
                  {folderStructure}
                </pre>
              </div>

              {/* Step 2: requirements.txt */}
              <div className="space-y-2">
                <div className="flex items-center gap-2 font-medium">
                  <FileCode className="w-4 h-4 text-primary" />
                  Step 2: Create backend/requirements.txt
                </div>
                <pre className="p-4 rounded-lg bg-secondary overflow-x-auto font-mono text-xs">
                  {requirementsTxt}
                </pre>
              </div>

              {/* Step 3: main.py */}
              <div className="space-y-2">
                <div className="flex items-center gap-2 font-medium">
                  <FileCode className="w-4 h-4 text-primary" />
                  Step 3: Create backend/main.py
                </div>
                <pre className="p-4 rounded-lg bg-secondary overflow-x-auto font-mono text-xs max-h-96 overflow-y-auto">
                  {mainPy}
                </pre>
              </div>

              {/* Step 4: Run Commands */}
              <div className="space-y-2">
                <div className="flex items-center gap-2 font-medium">
                  <Terminal className="w-4 h-4 text-primary" />
                  Step 4: Run the backend
                </div>
                <pre className="p-4 rounded-lg bg-secondary overflow-x-auto font-mono text-xs">
{`# Navigate to backend folder
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy your trained model to backend folder
# gesture_model.pkl should be in backend/

# Run the server
python main.py

# Server will start at http://localhost:8000`}
                </pre>
              </div>

              {/* Step 5: Test */}
              <div className="space-y-2">
                <div className="flex items-center gap-2 font-medium">
                  <CheckCircle2 className="w-4 h-4 text-success" />
                  Step 5: Test the API
                </div>
                <p className="text-muted-foreground">
                  Open <code className="px-1 py-0.5 bg-secondary rounded">http://localhost:8000/health</code> in your browser. 
                  You should see <code className="px-1 py-0.5 bg-secondary rounded">{`{"status": "healthy"}`}</code>
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
