import { motion } from 'framer-motion';
import { Play, Pause, Settings, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

interface ControlPanelProps {
  isCapturing: boolean;
  onToggleCapture: () => void;
  captureInterval: number;
  onIntervalChange: (value: number) => void;
  onClearHistory: () => void;
  backendUrl: string;
  onBackendUrlChange: (url: string) => void;
}

export const ControlPanel = ({
  isCapturing,
  onToggleCapture,
  captureInterval,
  onIntervalChange,
  onClearHistory,
  backendUrl,
  onBackendUrlChange,
}: ControlPanelProps) => {
  return (
    <div className="glass-card space-y-6">
      <div className="flex items-center gap-2 text-lg font-semibold">
        <Settings className="w-5 h-5" />
        Controls
      </div>

      {/* Start/Stop Button */}
      <motion.div whileTap={{ scale: 0.98 }}>
        <Button
          onClick={onToggleCapture}
          className="w-full h-14 text-lg font-semibold"
          variant={isCapturing ? 'destructive' : 'default'}
        >
          {isCapturing ? (
            <>
              <Pause className="w-5 h-5 mr-2" />
              Stop Recognition
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Start Recognition
            </>
          )}
        </Button>
      </motion.div>

      {/* Capture Interval */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">Capture Interval</span>
          <span className="text-sm font-mono text-muted-foreground">
            {captureInterval}ms
          </span>
        </div>
        <Slider
          value={[captureInterval]}
          onValueChange={(value) => onIntervalChange(value[0])}
          min={200}
          max={2000}
          step={100}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          Lower = faster but more CPU usage
        </p>
      </div>

      {/* Backend URL */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Backend URL</label>
        <input
          type="text"
          value={backendUrl}
          onChange={(e) => onBackendUrlChange(e.target.value)}
          className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="http://localhost:8000"
        />
      </div>

      {/* Clear History */}
      <Button
        onClick={onClearHistory}
        variant="outline"
        className="w-full"
      >
        <Trash2 className="w-4 h-4 mr-2" />
        Clear History
      </Button>
    </div>
  );
};
