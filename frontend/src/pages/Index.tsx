import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Hand, Github, Info } from 'lucide-react';
import { WebcamCapture } from '@/components/WebcamCapture';
import { PredictionDisplay } from '@/components/PredictionDisplay';
import { GestureHistory } from '@/components/GestureHistory';
import { ControlPanel } from '@/components/ControlPanel';
import { SetupInstructions } from '@/components/SetupInstructions';
import { usePrediction } from '@/hooks/usePrediction';

const Index = () => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [captureInterval, setCaptureInterval] = useState(500);
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");

  const {
    currentPrediction,
    isConnected,
    isLoading,
    history,
    predict,
    checkHealth,
    clearHistory,
  } = usePrediction(backendUrl);

  // Check backend health on mount and when URL changes
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const handleCapture = useCallback((imageData: string) => {
    predict(imageData);
  }, [predict]);

  const handleToggleCapture = useCallback(() => {
    setIsCapturing(prev => !prev);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <Hand className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold">SignLang AI</h1>
                <p className="text-xs text-muted-foreground">Real-time Sign Language Recognition</p>
              </div>
            </motion.div>
            
            <div className="flex items-center gap-2">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg hover:bg-secondary transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Webcam */}
          <div className="lg:col-span-2 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <WebcamCapture
                onCapture={handleCapture}
                isCapturing={isCapturing}
                captureInterval={captureInterval}
              />
            </motion.div>

            {/* Supported Gestures */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card"
            >
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Supported Gestures</h3>
              </div>
              <div className="flex gap-3 flex-wrap">
                {['A', 'B', 'C', 'D', 'E'].map((letter, index) => (
                  <motion.div
                    key={letter}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + index * 0.05 }}
                    className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center text-2xl font-bold hover:bg-primary hover:text-primary-foreground transition-colors cursor-default"
                  >
                    {letter}
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Setup Instructions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <SetupInstructions />
            </motion.div>
          </div>

          {/* Right Column - Controls & Results */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <PredictionDisplay
                gesture={currentPrediction?.gesture || null}
                confidence={currentPrediction?.confidence || null}
                isConnected={isConnected}
                isLoading={isLoading}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <ControlPanel
                isCapturing={isCapturing}
                onToggleCapture={handleToggleCapture}
                captureInterval={captureInterval}
                onIntervalChange={setCaptureInterval}
                onClearHistory={clearHistory}
                backendUrl={backendUrl}
                onBackendUrlChange={setBackendUrl}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GestureHistory history={history} />
            </motion.div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-muted-foreground">
            Built with MediaPipe + RandomForest • 5 Classes (A-E) • 35 Features
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
