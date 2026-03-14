import { useState, useEffect } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraConfig, CameraHandle } from './types';

export function useCamera(config: CameraConfig): CameraHandle {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    setIsReady(false);

    WebGPUCameraModule.startCameraPreview(
      config.device,
      config.width,
      config.height,
      config.fps,
    );

    // Camera needs time to produce first frame
    const timer = setTimeout(() => setIsReady(true), 500);

    return () => {
      clearTimeout(timer);
      WebGPUCameraModule.stopCameraPreview();
      setIsReady(false);
    };
  }, [config.device, config.width, config.height, config.fps]);

  return {
    isReady,
    width: config.width,
    height: config.height,
    fps: config.fps,
  };
}
