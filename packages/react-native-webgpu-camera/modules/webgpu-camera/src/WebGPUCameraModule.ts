import { NativeModule, requireNativeModule } from 'expo-modules-core';

export interface FrameDimensions {
  width: number;
  height: number;
  bytesPerRow: number;
}

interface WebGPUCameraModuleInterface extends NativeModule {
  startCameraPreview(deviceId: string, width: number, height: number): void;
  stopCameraPreview(): void;
  getCurrentFrameHandle(): number;
  getCurrentFramePixels(): Uint8Array;
  getFrameDimensions(): FrameDimensions;
  getFrameCounter(): number;
  startTestRecorder(outputPath: string, width: number, height: number): number;
  stopTestRecorder(): string;
  getThermalState(): string;
}

export default requireNativeModule<WebGPUCameraModuleInterface>(
  'WebGPUCamera'
);
