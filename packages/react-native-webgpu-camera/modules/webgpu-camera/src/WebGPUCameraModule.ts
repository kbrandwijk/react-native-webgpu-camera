import { NativeModule, requireNativeModule } from 'expo-modules-core';

interface WebGPUCameraModuleInterface extends NativeModule {
  startCameraPreview(deviceId: string, width: number, height: number, fps: number): void;
  stopCameraPreview(): void;
  getThermalState(): string;
  // Dawn compute pipeline
  setupMultiPassPipeline(config: {
    shaders: string[];
    width: number;
    height: number;
    buffers: [number, number, number][];
    useCanvas: boolean;
    sync: boolean;
  }): boolean;
  cleanupComputePipeline(): void;
  isComputeReady(): boolean;
}

export default requireNativeModule<WebGPUCameraModuleInterface>(
  'WebGPUCamera'
);
