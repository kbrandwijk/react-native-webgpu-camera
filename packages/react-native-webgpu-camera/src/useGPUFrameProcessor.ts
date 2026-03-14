import { useEffect, useState, useRef } from 'react';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import type { SkImage } from '@shopify/react-native-skia';
import WebGPUCameraModule from '../modules/webgpu-camera/src/WebGPUCameraModule';
import type {
  CameraHandle,
  CameraStream,
  FrameProcessor,
  GPUFrameProcessorResult,
} from './types';

export function useGPUFrameProcessor(
  camera: CameraHandle,
  processor: FrameProcessor,
): GPUFrameProcessorResult {
  const [error, setError] = useState<string | null>(null);
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImage | null>(null);
  const processorRef = useRef(processor);
  processorRef.current = processor;

  // Setup compute pipeline when camera is ready
  useEffect(() => {
    if (!camera.isReady) return;

    // v1: Extract WGSL by calling the processor once with a capture proxy.
    // The processor runs as a configuration function at setup time.
    // When frame.canvas support lands, the processor will run per-frame
    // in the worklet instead.
    let capturedWgsl: string | null = null;
    const captureFrame = {
      runShader(wgslCode: string) {
        if (!capturedWgsl) capturedWgsl = wgslCode;
      },
      canvas: new Proxy({} as any, {
        get() { return () => {}; }, // no-op stub for v1
      }),
      width: camera.width,
      height: camera.height,
    };
    try {
      processorRef.current(captureFrame);
    } catch {
      // Processor may reference worklet-only APIs during capture — safe to ignore
    }

    if (!capturedWgsl) {
      setError('No shader provided — call frame.runShader(wgslCode) in your processor');
      return;
    }

    // Setup native compute pipeline (compiles shader, creates GPU resources, installs JSI)
    const ok = WebGPUCameraModule.setupComputePipeline(
      capturedWgsl,
      camera.width,
      camera.height,
    );
    if (!ok) {
      setError('Compute pipeline setup failed');
      return;
    }
    setError(null);

    // Create stream host object — shared across Reanimated runtimes
    stream.value = globalThis.__webgpuCamera_createStream();

    return () => {
      // Dispose final frame to prevent GPU memory leak
      currentFrame.value?.dispose();
      currentFrame.value = null;
      stream.value = null;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, [camera.isReady, camera.width, camera.height, camera.fps]);

  // Frame callback — runs on UI thread every display frame
  useFrameCallback(() => {
    'worklet';
    const s = stream.value;
    if (!s) return;

    const img = s.nextImage();
    if (img) {
      currentFrame.value?.dispose();
      currentFrame.value = img;
    }
  });

  return { currentFrame, error };
}
