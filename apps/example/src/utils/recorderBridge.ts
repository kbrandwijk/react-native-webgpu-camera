import { Paths } from 'expo-file-system';
// NOTE: Verify this import path resolves in the monorepo. If not, try:
// import WebGPUCameraModule from '@/../../packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';
import WebGPUCameraModule from 'react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';

export interface RecorderState {
  isRecording: boolean;
  outputPath: string | null;
  surfaceHandle: number;
  path: 'surface-record' | 'readback-record' | 'unknown';
}

export function startRecording(
  width: number,
  height: number,
  filename = 'spike4_test.mp4'
): RecorderState {
  const docUri = Paths.document.uri;
  // Paths.document.uri is a file:// URI — strip scheme for native path
  const docPath = docUri.startsWith('file://') ? docUri.slice(7) : docUri;
  const outputPath = `${docPath}${filename}`;
  const surfaceHandle = WebGPUCameraModule.startTestRecorder(outputPath, width, height);

  const path = surfaceHandle !== 0 ? 'surface-record' : 'readback-record';

  console.log(`[Recorder] Started: ${path}, output: ${outputPath}`);
  console.log(`[Recorder] Surface handle: ${surfaceHandle}`);

  return { isRecording: true, outputPath, surfaceHandle, path };
}

export function stopRecording(): string {
  const filePath = WebGPUCameraModule.stopTestRecorder();
  console.log(`[Recorder] Stopped, file: ${filePath}`);
  return filePath;
}
