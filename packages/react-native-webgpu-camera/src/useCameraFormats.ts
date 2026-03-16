import { useEffect, useState } from 'react';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type { CameraFormat, ColorSpace, StabilizationMode } from './types';

function mapNativeFormat(raw: Record<string, any>): CameraFormat {
  return {
    width: raw.width as number,
    height: raw.height as number,
    minFps: raw.minFps as number,
    maxFps: raw.maxFps as number,
    pixelFormat: (raw.pixelFormat as CameraFormat['pixelFormat']) ?? 'bgra',
    supportedColorSpaces: (raw.supportedColorSpaces as ColorSpace[]) ?? ['sRGB'],
    isHDR: (raw.isHDR as boolean) ?? false,
    stabilizationModes: (raw.stabilizationModes as StabilizationMode[]) ?? ['off'],
    fieldOfView: (raw.fieldOfView as number) ?? 0,
    isBinned: (raw.isBinned as boolean) ?? false,
    isMultiCamSupported: (raw.isMultiCamSupported as boolean) ?? false,
    nativeHandle: raw.nativeHandle as number,
  };
}

export function useCameraFormats(device: 'back' | 'front'): CameraFormat[] {
  const [formats, setFormats] = useState<CameraFormat[]>([]);

  useEffect(() => {
    const raw = WebGPUCameraModule.getFormats(device);
    setFormats(raw.map(mapNativeFormat));
  }, [device]);

  return formats;
}
