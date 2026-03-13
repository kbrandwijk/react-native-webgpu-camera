import { NativeModule, requireNativeModule } from 'expo-modules-core';

export interface FrameDimensions {
  width: number;
  height: number;
  bytesPerRow: number;
}

interface WebGPUCameraModuleInterface extends NativeModule {
  startCameraPreview(deviceId: string, width: number, height: number, fps: number): void;
  stopCameraPreview(): void;
  getCurrentFrameHandle(): number;
  getCurrentFramePixels(): Uint8Array;
  getFrameDimensions(): FrameDimensions;
  getFrameCounter(): number;
  startTestRecorder(outputPath: string, width: number, height: number): number;
  stopTestRecorder(): string;
  appendFrameToRecorder(pixels: Uint8Array, width: number, height: number): void;
  getThermalState(): string;
  // Dawn compute pipeline
  setupComputePipeline(wgslCode: string, width: number, height: number): boolean;
  cleanupComputePipeline(): void;
  isComputeReady(): boolean;
}

export default requireNativeModule<WebGPUCameraModuleInterface>(
  'WebGPUCamera'
);
