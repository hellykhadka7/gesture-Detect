import { motion, AnimatePresence } from 'framer-motion';
import { Hand, AlertTriangle } from 'lucide-react';

interface PredictionDisplayProps {
  gesture: string | null;
  confidence: number | null;
  isConnected: boolean;
  isLoading: boolean;
}

export const PredictionDisplay = ({
  gesture,
  confidence,
  isConnected,
  isLoading,
}: PredictionDisplayProps) => {
  const hasValidPrediction = gesture && gesture !== 'None' && confidence !== null;
  const confidencePercent = confidence ? Math.round(confidence * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center gap-3 glass-card">
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
        <span className="text-sm font-medium">
          {isConnected ? 'Connected to Backend' : 'Backend Disconnected'}
        </span>
      </div>

      {/* Main Prediction Display */}
      <motion.div 
        layout
        className={`prediction-display ${!hasValidPrediction ? 'opacity-70' : ''}`}
        style={{
          background: hasValidPrediction 
            ? 'var(--gradient-primary)' 
            : 'hsl(var(--muted))'
        }}
      >
        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center py-4"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-12 h-12 border-4 border-primary-foreground/30 border-t-primary-foreground rounded-full"
              />
              <p className="mt-4 text-primary-foreground/80">Processing...</p>
            </motion.div>
          ) : hasValidPrediction ? (
            <motion.div
              key={gesture}
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.5, opacity: 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
              className="flex flex-col items-center"
            >
              <span className="gesture-letter">{gesture}</span>
              <span className="text-lg text-primary-foreground/90 mt-2">
                Detected Gesture
              </span>
            </motion.div>
          ) : (
            <motion.div
              key="no-hand"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center py-4 text-foreground"
            >
              <Hand className="w-16 h-16 text-muted-foreground mb-2" />
              <span className="text-xl font-semibold text-muted-foreground">
                No Hand Detected
              </span>
              <span className="text-sm text-muted-foreground mt-1">
                Show your hand to the camera
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Confidence Bar */}
      <div className="glass-card">
        <div className="flex justify-between items-center mb-3">
          <span className="text-sm font-medium text-muted-foreground">Confidence</span>
          <span className="text-lg font-bold font-mono">
            {hasValidPrediction ? `${confidencePercent}%` : '--'}
          </span>
        </div>
        <div className="confidence-bar">
          <motion.div
            className="confidence-fill"
            initial={{ width: 0 }}
            animate={{ width: `${hasValidPrediction ? confidencePercent : 0}%` }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
          />
        </div>
      </div>

      {/* Warning for low confidence */}
      <AnimatePresence>
        {hasValidPrediction && confidencePercent < 70 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 p-4 rounded-xl bg-warning/10 border border-warning/30"
          >
            <AlertTriangle className="w-5 h-5 text-warning" />
            <span className="text-sm text-warning">
              Low confidence - try adjusting hand position
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
