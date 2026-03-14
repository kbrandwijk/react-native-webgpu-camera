import { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  ScrollView,
  useWindowDimensions,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage, type SkImage as SkImageType } from '@shopify/react-native-skia';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import { useSpikeMetrics, SpikeResults } from '@/hooks/useSpikeMetrics';
import { startRecording, stopRecording, RecorderState } from '@/utils/recorderBridge';
import WebGPUCameraModule from 'react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';

interface CameraStream {
  nextImage(): SkImageType | null;
  dispose(): void;
}

declare global {
  // Returns a JsiSkImage (SkImage host object) directly — rAF fallback path
  function __webgpuCamera_nextImage(): SkImageType | null;
  // Returns a CameraStreamHostObject for worklet path
  function __webgpuCamera_createStream(): CameraStream;
}

const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;

function PreviewCanvas({ currentFrame }: { currentFrame: SharedValue<SkImageType | null> }) {
  const { width: screenW, height: screenH } = useWindowDimensions();
  return (
    <Canvas style={StyleSheet.absoluteFill}>
      <Fill color="black" />
      <Group transform={[
        { translateX: screenW },
        { rotate: Math.PI / 2 },
      ]}>
        <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
      </Group>
    </Canvas>
  );
}

type SharedValue<T> = ReturnType<typeof useSharedValue<T>>;

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [recorderState, setRecorderState] = useState<RecorderState | null>(null);
  const [results, setResults] = useState<SpikeResults | null>(null);
  const metrics = useSpikeMetrics();

  const recorderPathRef = useRef('pending');

  // Shared values: accessible from both JS and UI (worklet) runtimes
  const stream = useSharedValue<CameraStream | null>(null);
  const currentFrame = useSharedValue<SkImageType | null>(null);

  // useFrameCallback runs on UI thread every display frame.
  // The stream host object is shared across runtimes by Reanimated.
  // Skia Canvas watches currentFrame and redraws automatically — no setState.
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

  const startPipeline = useCallback(() => {
    setIsRunning(true);
    setResults(null);
    metrics.reset();

    // 1. Setup native compute pipeline (compiles WGSL, creates GPU resources, installs JSI)
    const ok = WebGPUCameraModule.setupComputePipeline(SOBEL_WGSL, CAMERA_WIDTH, CAMERA_HEIGHT);
    if (!ok) {
      setStatus('compute setup failed');
      console.error('[CameraSpikeScreen] Compute pipeline setup failed');
      return;
    }
    setStatus('compute ready');

    // 2. Create stream host object (JSI) — passable into worklets
    stream.value = globalThis.__webgpuCamera_createStream();

    // 3. Start camera with fps — each frame runs compute on native thread
    WebGPUCameraModule.startCameraPreview('back', CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS);
    console.log('[CameraSpikeScreen] Native pipeline started');
  }, [metrics.reset]);

  const stopPipeline = useCallback(() => {
    setIsRunning(false);
    stream.value = null;

    WebGPUCameraModule.stopCameraPreview();
    WebGPUCameraModule.cleanupComputePipeline();

    if (recorderState?.isRecording) {
      stopRecording();
      setRecorderState(null);
      recorderPathRef.current = 'pending';
    }

    metrics.logSummary({
      spike1Path: 'zero-copy',
      spike2Path: 'worklet-compute',
      spike3Path: 'graphite',
      spike4Path: recorderPathRef.current === 'pending' ? 'unknown' : recorderPathRef.current as SpikeResults['spike4Path'],
    });

    const summary = metrics.getSummary();
    if (summary) {
      setResults({
        ...summary,
        spike1Path: 'zero-copy',
        spike2Path: 'worklet-compute',
        spike3Path: 'graphite',
        spike4Path: recorderPathRef.current === 'pending' ? 'unknown' : recorderPathRef.current as SpikeResults['spike4Path'],
      });
    }
    setStatus('idle');
  }, [metrics, recorderState]);

  const handleStartRecording = useCallback(() => {
    const state = startRecording(CAMERA_WIDTH, CAMERA_HEIGHT);
    setRecorderState(state);
    recorderPathRef.current = state.path;
    setTimeout(() => {
      const filePath = stopRecording();
      setRecorderState(null);
      recorderPathRef.current = 'pending';
      console.log(`[CameraSpikeScreen] Recording saved: ${filePath}`);
    }, 5000);
  }, []);

  useEffect(() => {
    return () => {
      stream.value = null;
      WebGPUCameraModule.cleanupComputePipeline();
    };
  }, []);

  return (
    <View style={styles.container}>
      <PreviewCanvas currentFrame={currentFrame} />

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          Pipeline: {status}
        </Text>
      </View>

      <View style={styles.controls}>
        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={isRunning ? stopPipeline : startPipeline}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>

        {isRunning && (
          <Pressable
            style={[styles.button, recorderState?.isRecording && styles.buttonActive]}
            onPress={handleStartRecording}
            disabled={recorderState?.isRecording}
          >
            <Text style={styles.buttonText}>
              {recorderState?.isRecording ? 'Recording...' : 'Record 5s'}
            </Text>
          </Pressable>
        )}
      </View>

      {results && (
        <ScrollView style={styles.results}>
          <Text style={styles.resultsTitle}>Spike Results</Text>
          <Text style={styles.resultsText}>{JSON.stringify(results, null, 2)}</Text>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  statusBar: {
    position: 'absolute', top: 44, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 4, padding: 8,
  },
  statusText: { color: '#aaa', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'center', gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 24,
    paddingHorizontal: 24, paddingVertical: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: { backgroundColor: 'rgba(255,80,80,0.4)', borderColor: 'rgba(255,80,80,0.6)' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  results: {
    position: 'absolute', top: 220, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.8)', borderRadius: 8, padding: 12, maxHeight: 300,
  },
  resultsTitle: { color: '#0f0', fontSize: 16, fontWeight: '700', marginBottom: 8 },
  resultsText: { color: '#0f0', fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
});
