// react-native-webgpu-camera

// Hooks
export { useCamera } from './useCamera';
export { useCameraFormats } from './useCameraFormats';
export { useGPUFrameProcessor } from './useGPUFrameProcessor';

// Types
export type {
  CameraConfig,
  CameraHandle,
  CameraFormat,
  ColorSpace,
  StabilizationMode,
  CameraStream,
  ProcessorFrame,
  RenderFrame,
  FrameProcessor,
  GPUFrameProcessorResult,
  ProcessorConfig,
  BufferHandle,
  NullableBuffers,
  TypedArrayConstructor,
} from './types';

// Re-export WebGPU canvas types for convenience
export type { WebGPUCanvasRef } from '@shopify/react-native-skia';
export { WebGPUCanvas } from '@shopify/react-native-skia';

// GPU Resources
export { GPUResource } from './GPUResource';
export type { ResourceHandle, ModelResourceHandle, ModelOptions } from './GPUResource';

// Utilities
export { parseCubeFile } from './parseCubeFile';

// Native module (advanced usage)
export { default as WebGPUCameraModule } from '../modules/webgpu-camera/src/WebGPUCameraModule';
export type { FrameDimensions } from '../modules/webgpu-camera/src/WebGPUCameraModule';
