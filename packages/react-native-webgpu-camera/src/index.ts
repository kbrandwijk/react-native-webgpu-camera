// react-native-webgpu-camera

// Hooks
export { useCamera } from './useCamera';
export { useGPUFrameProcessor } from './useGPUFrameProcessor';

// Types
export type {
  CameraConfig,
  CameraHandle,
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

// Native module (advanced usage)
export { default as WebGPUCameraModule } from '../modules/webgpu-camera/src/WebGPUCameraModule';
export type { FrameDimensions } from '../modules/webgpu-camera/src/WebGPUCameraModule';
