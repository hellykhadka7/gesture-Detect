import { useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Camera, CameraOff, RefreshCw, ShieldAlert, MonitorX, Webcam } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useWebcamStream, WebcamErrorType } from '@/hooks/useWebcamStream';

interface WebcamCaptureProps {
  onCapture: (imageData: string) => void;
  isCapturing: boolean;
  captureInterval?: number;
}

// Error icon and message based on error type
function getErrorDisplay(errorType: WebcamErrorType | null, errorMessage: string | null) {
  switch (errorType) {
    case "permission_denied":
      return {
        icon: <ShieldAlert className="w-16 h-16 text-destructive mb-4" />,
        title: "Camera Access Denied",
        message: errorMessage || "Please allow camera permissions in your browser settings.",
        hint: "Click the camera icon in your browser's address bar to change permissions.",
      };
    case "device_not_found":
      return {
        icon: <Webcam className="w-16 h-16 text-muted-foreground mb-4" />,
        title: "No Camera Found",
        message: errorMessage || "No camera device was detected.",
        hint: "Please connect a webcam and try again.",
      };
    case "element_not_mounted":
      return {
        icon: <MonitorX className="w-16 h-16 text-muted-foreground mb-4" />,
        title: "Display Error",
        message: "The video display failed to initialize.",
        hint: "Click 'Retry' or refresh the page.",
      };
    case "stream_ended":
      return {
        icon: <CameraOff className="w-16 h-16 text-muted-foreground mb-4" />,
        title: "Camera Disconnected",
        message: errorMessage || "The camera stream was interrupted.",
        hint: "Another application may be using the camera. Close it and retry.",
      };
    default:
      return {
        icon: <CameraOff className="w-16 h-16 text-muted-foreground mb-4" />,
        title: "Camera Error",
        message: errorMessage || "An error occurred while accessing the camera.",
        hint: "Please try again or refresh the page.",
      };
  }
}

export const WebcamCapture = ({ 
  onCapture, 
  isCapturing, 
  captureInterval = 500 
}: WebcamCaptureProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  
  const { 
    videoCallbackRef, 
    hasPermission, 
    error, 
    errorType,
    debugInfo, 
    needsUserGesture, 
    isRetrying,
    start, 
    retry 
  } = useWebcamStream({
    autoStart: true,
  });

  // Combined callback ref that stores the element and passes to hook
  const combinedVideoRef = useCallback((node: HTMLVideoElement | null) => {
    videoElementRef.current = node;
    videoCallbackRef(node);
  }, [videoCallbackRef]);

  const captureFrame = useCallback(() => {
    const video = videoElementRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Check video is ready (HAVE_ENOUGH_DATA = 4)
    if (!ctx || video.readyState < 4) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Mirror the image for natural feel
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];
    onCapture(base64Data);
  }, [onCapture]);

  // Capture interval effect
  useEffect(() => {
    if (!isCapturing || !hasPermission) return;
    
    const intervalId = setInterval(captureFrame, captureInterval);
    return () => clearInterval(intervalId);
  }, [isCapturing, hasPermission, captureFrame, captureInterval]);

  const errorDisplay = getErrorDisplay(errorType, error);
  const showVideo = hasPermission === true;
  const showLoading = hasPermission === null;
  const showError = hasPermission === false;

  return (
    <div className="relative">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="webcam-container aspect-video overflow-hidden"
      >
        {/* Video element is ALWAYS in the DOM so callback ref fires immediately */}
        <video
          ref={combinedVideoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${showVideo ? 'block' : 'hidden'}`}
          style={{ transform: 'scaleX(-1)' }}
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Loading state overlay */}
        {showLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-card">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            >
              <Camera className="w-16 h-16 text-primary" />
            </motion.div>
            <p className="text-muted-foreground mt-4">Requesting camera access...</p>
            {debugInfo && (
              <p className="text-xs text-muted-foreground mt-2 font-mono">{debugInfo}</p>
            )}
          </div>
        )}

        {/* Error state overlay */}
        {showError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-card p-8">
            {errorDisplay.icon}
            <h3 className="text-lg font-semibold text-foreground mb-2">{errorDisplay.title}</h3>
            <p className="text-muted-foreground text-center mb-2">{errorDisplay.message}</p>
            <p className="text-xs text-muted-foreground text-center mb-4">{errorDisplay.hint}</p>
            
            <Button 
              onClick={retry} 
              disabled={isRetrying}
              variant="outline"
              className="gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
              {isRetrying ? 'Retrying...' : 'Retry'}
            </Button>
            
            {debugInfo && (
              <p className="text-xs text-muted-foreground mt-4 font-mono bg-muted px-2 py-1 rounded">
                {debugInfo}
              </p>
            )}
          </div>
        )}

        {/* User gesture required overlay */}
        {showVideo && needsUserGesture && (
          <div className="absolute inset-0 flex items-center justify-center bg-card/70 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3 p-6 text-center">
              <Camera className="w-12 h-12 text-primary" />
              <p className="text-sm text-muted-foreground">
                Click to start the camera preview.
              </p>
              <Button onClick={start} className="gap-2">
                <Camera className="w-4 h-4" />
                Start Camera
              </Button>
              {debugInfo && (
                <p className="text-xs text-muted-foreground font-mono">{debugInfo}</p>
              )}
            </div>
          </div>
        )}
        
        {/* Recording indicator */}
        {showVideo && isCapturing && !needsUserGesture && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-4 right-4 flex items-center gap-2 bg-card/80 backdrop-blur-sm rounded-full px-3 py-1.5"
          >
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="w-2 h-2 rounded-full bg-destructive"
            />
            <span className="text-sm font-medium">Live</span>
          </motion.div>
        )}

        {/* Debug info overlay */}
        {showVideo && debugInfo && !needsUserGesture && (
          <div className="absolute bottom-3 left-3 rounded-md bg-card/80 backdrop-blur-sm px-3 py-1.5">
            <p className="text-xs text-muted-foreground font-mono">{debugInfo}</p>
          </div>
        )}
      </motion.div>
    </div>
  );
};
