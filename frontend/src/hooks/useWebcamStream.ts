import { useCallback, useEffect, useRef, useState } from "react";

export type WebcamErrorType = 
  | "permission_denied" 
  | "device_not_found" 
  | "element_not_mounted" 
  | "overconstrained"
  | "stream_ended"
  | "unknown";

export interface UseWebcamStreamOptions {
  constraints?: MediaStreamConstraints;
  autoStart?: boolean;
}

export interface UseWebcamStreamResult {
  videoCallbackRef: (node: HTMLVideoElement | null) => void;
  streamRef: React.MutableRefObject<MediaStream | null>;
  hasPermission: boolean | null;
  error: string | null;
  errorType: WebcamErrorType | null;
  debugInfo: string | null;
  needsUserGesture: boolean;
  isRetrying: boolean;
  start: () => Promise<void>;
  retry: () => Promise<void>;
  stop: () => void;
}

const defaultConstraints: MediaStreamConstraints = {
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    facingMode: { ideal: "user" },
  },
  audio: false,
};

function categorizeError(err: unknown): { type: WebcamErrorType; message: string } {
  const name = (err as any)?.name || "";
  const message = (err as any)?.message || String(err);

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return {
      type: "permission_denied",
      message: "Camera access was denied. Please allow camera permissions in your browser settings and refresh.",
    };
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return {
      type: "device_not_found",
      message: "No camera found. Please connect a camera and try again.",
    };
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return {
      type: "overconstrained",
      message: "Camera does not support the requested settings. Trying with default settings...",
    };
  }
  return {
    type: "unknown",
    message: message || "An unknown error occurred while accessing the camera.",
  };
}

export function useWebcamStream(options: UseWebcamStreamOptions = {}): UseWebcamStreamResult {
  const { constraints = defaultConstraints, autoStart = true } = options;

  // Use a ref to store the video element from the callback ref
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorType, setErrorType] = useState<WebcamErrorType | null>(null);
  const [debugInfo, setDebugInfo] = useState<string | null>(null);
  const [needsUserGesture, setNeedsUserGesture] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const startInFlightRef = useRef(false);
  const mountedRef = useRef(true);
  const autoStartAttemptedRef = useRef(false);

  // Stop all tracks and clean up stream
  const stop = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => {
        track.stop();
        setDebugInfo(`Stopped track: ${track.kind}`);
      });
      streamRef.current = null;
    }
    // Clear srcObject from video element
    if (videoElementRef.current) {
      videoElementRef.current.srcObject = null;
    }
  }, []);

  // Attach stream to video element and attempt playback
  const attachStreamToVideo = useCallback((stream: MediaStream) => {
    const videoEl = videoElementRef.current;
    
    if (!videoEl) {
      setError("Video element not mounted. Please wait and retry.");
      setErrorType("element_not_mounted");
      setDebugInfo("Video element missing when trying to attach stream.");
      setHasPermission(false);
      // Stop the stream since we can't use it
      stream.getTracks().forEach((t) => t.stop());
      return false;
    }

    // Check if video element is in a valid state
    setDebugInfo(`Attaching stream. Video readyState: ${videoEl.readyState}`);
    
    videoEl.srcObject = stream;
    streamRef.current = stream;

    const attemptPlay = () => {
      // Wait for video to be ready
      if (videoEl.readyState < 2) {
        setDebugInfo(`Waiting for video metadata... readyState: ${videoEl.readyState}`);
        return;
      }

      const playPromise = videoEl.play();
      if (playPromise) {
        playPromise
          .then(() => {
            setDebugInfo("Video playing successfully.");
            setNeedsUserGesture(false);
          })
          .catch((e) => {
            if (e.name === "AbortError") {
              // Play was interrupted, this is usually fine
              setDebugInfo("Play interrupted, will retry on user gesture.");
            } else if (e.name === "NotAllowedError") {
              setNeedsUserGesture(true);
              setDebugInfo("Autoplay blocked. Click 'Start Camera' to continue.");
            } else {
              setDebugInfo(`Play failed: ${e.name} - ${e.message}`);
            }
          });
      }
    };

    // Set up event listeners
    videoEl.onloadedmetadata = () => {
      setDebugInfo("Video metadata loaded. Attempting playback...");
      attemptPlay();
    };

    videoEl.onplaying = () => {
      setDebugInfo("Video is now playing.");
    };

    videoEl.onpause = () => {
      setDebugInfo("Video paused.");
    };

    videoEl.onerror = () => {
      setDebugInfo("Video element encountered an error.");
    };

    // Try to play immediately if already have metadata
    attemptPlay();

    // Watch for tracks ending unexpectedly
    stream.getVideoTracks().forEach((track) => {
      track.onended = () => {
        if (!mountedRef.current) return;
        setError("Camera stream ended unexpectedly. Another app may be using the camera.");
        setErrorType("stream_ended");
        setDebugInfo("Video track ended.");
        setHasPermission(false);
      };
    });

    return true;
  }, []);

  // Main start function with retry logic
  const start = useCallback(async () => {
    if (startInFlightRef.current) {
      setDebugInfo("Start already in progress, skipping...");
      return;
    }

    // Check if video element is mounted
    if (!videoElementRef.current) {
      setError("Video element not ready. Please wait a moment and try again.");
      setErrorType("element_not_mounted");
      setDebugInfo("Cannot start: video element not mounted yet.");
      return;
    }

    startInFlightRef.current = true;
    setNeedsUserGesture(false);
    setError(null);
    setErrorType(null);

    const requestStream = async (c: MediaStreamConstraints, label: string) => {
      setDebugInfo(`Requesting camera access (${label})...`);
      return navigator.mediaDevices.getUserMedia(c);
    };

    try {
      let stream: MediaStream | null = null;

      // Attempt 1: Use provided constraints
      try {
        stream = await requestStream(constraints, "with constraints");
      } catch (e: unknown) {
        const { type } = categorizeError(e);
        
        // Only retry with looser constraints for certain errors
        if (type === "overconstrained" || type === "device_not_found") {
          setDebugInfo("Retrying with minimal constraints...");
          
          // Attempt 2: Minimal constraints
          try {
            stream = await requestStream({ video: true, audio: false }, "minimal");
          } catch (e2: unknown) {
            const { type: type2 } = categorizeError(e2);
            
            if (type2 === "device_not_found") {
              // Attempt 3: Enumerate devices and try first camera
              setDebugInfo("Enumerating available cameras...");
              const devices = await navigator.mediaDevices.enumerateDevices();
              const cameras = devices.filter((d) => d.kind === "videoinput");
              
              if (cameras.length === 0) {
                throw e2; // No cameras found
              }
              
              setDebugInfo(`Found ${cameras.length} camera(s). Trying first one...`);
              stream = await requestStream(
                { video: { deviceId: { exact: cameras[0].deviceId } }, audio: false },
                "specific device"
              );
            } else {
              throw e2;
            }
          }
        } else {
          throw e;
        }
      }

      if (!stream) {
        throw new Error("Failed to acquire camera stream");
      }

      // Check if component is still mounted
      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        setDebugInfo("Component unmounted, stopped stream.");
        return;
      }

      setHasPermission(true);
      setDebugInfo("Stream acquired successfully.");

      // Attach to video element
      attachStreamToVideo(stream);

    } catch (err: unknown) {
      if (!mountedRef.current) return;

      const { type, message } = categorizeError(err);
      setHasPermission(false);
      setError(message);
      setErrorType(type);
      setDebugInfo(`Failed: ${(err as any)?.name || "Unknown"} - ${(err as any)?.message || String(err)}`);
    } finally {
      startInFlightRef.current = false;
    }
  }, [constraints, attachStreamToVideo]);

  // Explicit retry function
  const retry = useCallback(async () => {
    setIsRetrying(true);
    setError(null);
    setErrorType(null);
    setDebugInfo("Retrying camera access...");
    
    // Stop any existing stream first
    stop();
    
    // Small delay to ensure cleanup
    await new Promise((resolve) => setTimeout(resolve, 100));
    
    await start();
    setIsRetrying(false);
  }, [start, stop]);

  // Callback ref for video element - triggers start immediately when element mounts
  const videoCallbackRef = useCallback((node: HTMLVideoElement | null) => {
    const previousNode = videoElementRef.current;
    videoElementRef.current = node;
    
    if (node && !previousNode) {
      // Video element just mounted - start immediately if autoStart is enabled
      setDebugInfo("Video element mounted.");
      
      if (autoStart && !autoStartAttemptedRef.current && !startInFlightRef.current) {
        autoStartAttemptedRef.current = true;
        // Use requestAnimationFrame to ensure DOM is painted
        requestAnimationFrame(() => {
          if (mountedRef.current && videoElementRef.current) {
            start();
          }
        });
      }
    } else if (!node && previousNode) {
      setDebugInfo("Video element unmounted.");
    }
  }, [autoStart, start]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      // Stop all tracks to prevent zombie camera sessions
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
        streamRef.current = null;
      }
      // Clear video element
      if (videoElementRef.current) {
        videoElementRef.current.srcObject = null;
      }
    };
  }, []);

  return {
    videoCallbackRef,
    streamRef,
    hasPermission,
    error,
    errorType,
    debugInfo,
    needsUserGesture,
    isRetrying,
    start,
    retry,
    stop,
  };
}
