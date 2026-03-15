import { NativeModule, requireNativeModule } from 'expo-modules-core';

interface WebGPUCameraModuleInterface extends NativeModule {
  getFormats(deviceId: string): Record<string, any>[];
  startCameraPreview(
    deviceId: string,
    nativeHandle: number,
    colorSpace: string,
  ): void;
  stopCameraPreview(): void;
  getThermalState(): string;
  setupMultiPassPipeline(config: {
    shaders: string[];
    width: number;
    height: number;
    buffers: [number, number, number][];
    useCanvas: boolean;
    sync: boolean;
    resources: {
      type: string;
      data: ArrayBuffer;
      width: number;
      height: number;
      depth: number;
    }[];
    passInputs: {
      passIndex: number;
      bindings: {
        index: number;
        type: string;
        resourceHandle?: number;
        sourcePass?: number;
        sourceBuffer?: number;
      }[];
    }[];
    textureOutputPasses: number[];
  }): boolean;
  getActiveFormatInfo(): { width: number; height: number; fps: number } | null;
  cleanupComputePipeline(): void;
  isComputeReady(): boolean;
}

export default requireNativeModule<WebGPUCameraModuleInterface>(
  'WebGPUCamera'
);
