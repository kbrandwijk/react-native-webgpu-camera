import { useEffect, useState } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraConfig, CameraHandle } from './types';

export function useCamera(config: CameraConfig): CameraHandle {
  const [isReady, setIsReady] = useState(false);
  const [resolvedWidth, setResolvedWidth] = useState(config.format?.width ?? 0);
  const [resolvedHeight, setResolvedHeight] = useState(config.format?.height ?? 0);
  const [resolvedFps, setResolvedFps] = useState(config.format?.maxFps ?? 30);

  const device = config.device;
  const nativeHandle = config.format?.nativeHandle ?? -1;
  const colorSpace = config.colorSpace ?? 'sRGB';
  const useDepth = config.useDepth ?? false;

  useEffect(() => {
    WebGPUCameraModule.startCameraPreview(device, nativeHandle, colorSpace, useDepth);

    // Camera needs time to produce first frame
    const timeout = setTimeout(() => {
      // Query actual dimensions (important when using default format)
      const info = WebGPUCameraModule.getActiveFormatInfo();
      if (info) {
        setResolvedWidth(info.width);
        setResolvedHeight(info.height);
        setResolvedFps(info.fps);
      }
      setIsReady(true);
    }, 500);

    return () => {
      clearTimeout(timeout);
      WebGPUCameraModule.stopCameraPreview();
      setIsReady(false);
    };
  }, [device, nativeHandle, colorSpace, useDepth]);

  return {
    isReady,
    width: resolvedWidth,
    height: resolvedHeight,
    fps: resolvedFps,
    colorSpace,
  };
}
