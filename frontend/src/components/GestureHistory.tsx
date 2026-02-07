import { motion, AnimatePresence } from 'framer-motion';

interface GestureRecord {
  gesture: string;
  confidence: number;
  timestamp: Date;
}

interface GestureHistoryProps {
  history: GestureRecord[];
}

export const GestureHistory = ({ history }: GestureHistoryProps) => {
  if (history.length === 0) {
    return (
      <div className="glass-card">
        <h3 className="text-lg font-semibold mb-4">Recent Gestures</h3>
        <p className="text-sm text-muted-foreground text-center py-8">
          No gestures detected yet
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card">
      <h3 className="text-lg font-semibold mb-4">Recent Gestures</h3>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        <AnimatePresence initial={false}>
          {history.slice(0, 10).map((record, index) => (
            <motion.div
              key={record.timestamp.getTime()}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center justify-between p-3 rounded-lg bg-secondary/50"
            >
              <div className="flex items-center gap-3">
                <span className="w-10 h-10 flex items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold text-xl">
                  {record.gesture}
                </span>
                <span className="text-sm text-muted-foreground">
                  {record.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <span className="text-sm font-mono font-medium">
                {Math.round(record.confidence * 100)}%
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};
