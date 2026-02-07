import { useState, useCallback, useRef } from 'react';

interface PredictionResult {
  gesture: string;
  confidence: number;
}

interface GestureRecord {
  gesture: string;
  confidence: number;
  timestamp: Date;
}

export const usePrediction = (backendUrl: string) => {
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<GestureRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const lastGestureRef = useRef<string | null>(null);
  const consecutiveCountRef = useRef(0);

  const predict = useCallback(async (imageBase64: string) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageBase64 }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setCurrentPrediction(data);
      setIsConnected(true);

      // Add to history with smoothing (only add if same gesture detected 2+ times)
      if (data.gesture !== 'None' && data.confidence > 0.5) {
        if (data.gesture === lastGestureRef.current) {
          consecutiveCountRef.current++;
          if (consecutiveCountRef.current >= 2) {
            setHistory(prev => {
              const newRecord: GestureRecord = {
                gesture: data.gesture,
                confidence: data.confidence,
                timestamp: new Date(),
              };
              // Avoid duplicates within 2 seconds
              const lastRecord = prev[0];
              if (lastRecord && 
                  lastRecord.gesture === data.gesture && 
                  Date.now() - lastRecord.timestamp.getTime() < 2000) {
                return prev;
              }
              return [newRecord, ...prev].slice(0, 50);
            });
          }
        } else {
          lastGestureRef.current = data.gesture;
          consecutiveCountRef.current = 1;
        }
      }

    } catch (err) {
      console.error('Prediction error:', err);
      setIsConnected(false);
      setError(err instanceof Error ? err.message : 'Connection failed');
      setCurrentPrediction(null);
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl]);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/health`);
      setIsConnected(response.ok);
    } catch {
      setIsConnected(false);
    }
  }, [backendUrl]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    lastGestureRef.current = null;
    consecutiveCountRef.current = 0;
  }, []);

  return {
    currentPrediction,
    isConnected,
    isLoading,
    history,
    error,
    predict,
    checkHealth,
    clearHistory,
  };
};
