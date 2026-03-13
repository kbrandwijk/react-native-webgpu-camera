import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  ScrollView,
} from 'react-native';
import { runOnUI, runOnJS, useSharedValue } from 'react-native-reanimated';
import { useSpikeMetrics, SpikeResults } from '@/hooks/useSpikeMetrics';
import { useGPUPipeline } from '@/hooks/useGPUPipeline';
import { SpikeOverlay } from '@/components/SpikeOverlay';
import { startRecording, stopRecording, RecorderState } from '@/utils/recorderBridge';
import WebGPUCameraModule from 'react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule';

const CAMERA_WIDTH = 1920;
const CAMERA_HEIGHT = 1080;
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;
const RUN_DURATION_S = 60;

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [recorderState, setRecorderState] = useState<RecorderState | null>(null);
  const [spikeStatus, setSpikeStatus] = useState({
    spike1: 'pending',
    spike2: 'pending',
    spike3: 'pending',
    spike4: 'pending',
    fps: 0,
    elapsed: 0,
  });
  const [results, setResults] = useState<SpikeResults | null>(null);
  const metrics = useSpikeMetrics();
  const pipeline = useGPUPipeline(CAMERA_WIDTH, CAMERA_HEIGHT);

  // Shared values for worklet thread access (avoids stale closures)
  const isRunningRef = useSharedValue(false);
  // Use shared value (not React ref) so worklet can read recorder status
  const recorderPathSV = useSharedValue<string>('pending');

  // Push status updates from worklet -> JS thread
  const updateStatus = useCallback((status: typeof spikeStatus) => {
    setSpikeStatus(status);
  }, []);

  // autoStop is called from the worklet via runOnJS when elapsed >= RUN_DURATION_S.
  // Sets shared value to stop the loop; the useEffect cleanup handles full teardown.
  // Avoids circular dependency with stopPipeline (defined below).
  const autoStop = useCallback(() => {
    isRunningRef.value = false;
    setIsRunning(false);
  }, [isRunningRef]);

  // Access worklet-compatible shared values from the pipeline hook
  const { workletResources } = pipeline;

  // Shared values for worklet-side state (React refs are NOT accessible from worklets)
  const lastFrameCounterSV = useSharedValue(0);
  const startTimeSV = useSharedValue(0);
  const spike2StatusSV = useSharedValue('pending');
  const spike3StatusSV = useSharedValue('pending');

  // Callbacks for runOnJS — must be stable references
  const recordFrameJS = useCallback((timing: { importMs: number; computeMs: number; skiaMs: number; totalMs: number }) => {
    metrics.recordFrame(timing);
  }, [metrics]);

  const recordThermalJS = useCallback((thermal: string) => {
    metrics.recordThermalChange(thermal);
  }, [metrics]);

  // Worklet render loop — runs on UI thread via runOnUI
  // This validates Spike 2: WebGPU compute dispatch from worklet thread
  const startRenderLoop = useCallback(() => {
    // Capture current spike status for worklet
    spike2StatusSV.value = pipeline.state.computeSupported ? 'worklet-compute' : 'pending';
    spike3StatusSV.value = pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'pending';
    startTimeSV.value = performance.now();

    const tick = () => {
      'worklet';
      if (!isRunningRef.value) return;

      // Read frame counter from native module (JSI call — works from worklet)
      const frameCounter = WebGPUCameraModule.getFrameCounter();

      if (frameCounter > lastFrameCounterSV.value) {
        lastFrameCounterSV.value = frameCounter;

        // Get frame data from native module (JSI calls — work from worklet)
        const pixels = WebGPUCameraModule.getCurrentFramePixels();
        const dims = WebGPUCameraModule.getFrameDimensions();

        // Read WebGPU resources from shared values (NOT React refs)
        const device = workletResources.device.value;
        const computePipeline = workletResources.computePipeline.value;
        const inputTexture = workletResources.inputTexture.value;
        const bindGroup = workletResources.bindGroup.value;
        const w = workletResources.width.value;
        const h = workletResources.height.value;

        if (pixels.length > 0 && dims.width > 0 && device && computePipeline && inputTexture) {
          const t0 = performance.now();

          // Upload camera frame to input texture — THIS IS THE SPIKE 2 VALIDATION
          // WebGPU JSI calls executing on the worklet/UI thread
          device.queue.writeTexture(
            { texture: inputTexture },
            pixels,
            { bytesPerRow: dims.bytesPerRow },
            { width: w, height: h },
          );
          const tImport = performance.now();

          // Dispatch Sobel compute on worklet thread
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(computePipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(
            Math.ceil(w / 16),
            Math.ceil(h / 16),
          );
          pass.end();
          device.queue.submit([encoder.finish()]);
          const tCompute = performance.now();

          const skiaMs = 0; // Measured separately via Skia rendering

          // Push timing data to JS thread for metrics
          runOnJS(recordFrameJS)({
            importMs: tImport - t0,
            computeMs: tCompute - tImport,
            skiaMs,
            totalMs: tCompute - t0 + skiaMs,
          });

          const elapsed = Math.floor((performance.now() - startTimeSV.value) / 1000);

          runOnJS(updateStatus)({
            spike1: 'copy-fallback',
            spike2: spike2StatusSV.value,
            spike3: spike3StatusSV.value,
            spike4: recorderPathSV.value,
            fps: 0, // Updated from metrics on JS side
            elapsed,
          });

          // Check thermal state every ~1s
          if (frameCounter % 30 === 0) {
            const thermal = WebGPUCameraModule.getThermalState();
            runOnJS(recordThermalJS)(thermal);
          }

          // Auto-stop after RUN_DURATION_S
          if (elapsed >= RUN_DURATION_S) {
            runOnJS(autoStop)();
            return;
          }
        }
      }

      // Schedule next tick on UI thread.
      // RISK: Reanimated's worklet runtime may not support setTimeout natively.
      // Fallbacks: requestAnimationFrame, or move loop to JS with runOnJS.
      setTimeout(tick, FRAME_INTERVAL);
    };

    runOnUI(tick)();
  }, [pipeline, isRunningRef, workletResources, updateStatus, autoStop, recordFrameJS, recordThermalJS, lastFrameCounterSV, startTimeSV, spike2StatusSV, spike3StatusSV, recorderPathSV]);

  const startPipeline = useCallback(async () => {
    isRunningRef.value = true;
    setIsRunning(true);
    setResults(null);
    metrics.reset();
    lastFrameCounterSV.value = 0;
    startTimeSV.value = 0;

    // Initialize GPU pipeline
    await pipeline.initialize();

    // Start camera
    WebGPUCameraModule.startCameraPreview('back', CAMERA_WIDTH, CAMERA_HEIGHT);

    console.log('[CameraSpikeScreen] Pipeline started');

    // Start render loop after a brief delay for camera to warm up
    setTimeout(() => {
      startRenderLoop();
    }, 500);
  }, [pipeline, metrics, startRenderLoop, isRunningRef, lastFrameCounterSV, startTimeSV]);

  const stopPipeline = useCallback(() => {
    isRunningRef.value = false;
    setIsRunning(false);

    // Stop camera
    WebGPUCameraModule.stopCameraPreview();

    // Stop recorder if active
    if (recorderState?.isRecording) {
      stopRecording();
      setRecorderState(null);
      recorderPathSV.value = 'pending';
    }

    // Log results
    metrics.logSummary({
      spike1Path: 'copy-fallback',
      spike2Path: pipeline.state.computeSupported ? 'worklet-compute' : 'unknown',
      spike3Path: pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'unknown',
      spike4Path: recorderPathSV.value === 'pending' ? 'unknown' : recorderPathSV.value,
    });

    const summary = metrics.getSummary();
    if (summary) {
      setResults({
        ...summary,
        spike1Path: 'copy-fallback',
        spike2Path: pipeline.state.computeSupported ? 'worklet-compute' : 'unknown',
        spike3Path: pipeline.state.deviceSource === 'graphite' ? 'graphite' : 'unknown',
        spike4Path: recorderPathSV.value === 'pending' ? 'unknown' : recorderPathSV.value,
      });
    }

    // Cleanup GPU resources
    pipeline.cleanup();
  }, [pipeline, metrics, recorderState, isRunningRef, recorderPathSV]);

  const handleStartRecording = useCallback(() => {
    const state = startRecording(CAMERA_WIDTH, CAMERA_HEIGHT);
    setRecorderState(state);
    recorderPathSV.value = state.path;
    setSpikeStatus(s => ({ ...s, spike4: state.path }));

    // Stop after 5 seconds
    setTimeout(() => {
      const filePath = stopRecording();
      setRecorderState(null);
      recorderPathSV.value = 'pending';
      console.log(`[CameraSpikeScreen] Recording saved: ${filePath}`);
    }, 5000);
  }, [recorderPathSV]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isRunningRef.value = false;
      pipeline.cleanup();
    };
  }, [pipeline, isRunningRef]);

  return (
    <View style={styles.container}>
      {/* Skia overlay — validates Spike 3 */}
      <SpikeOverlay
        fps={spikeStatus.fps}
        spike1Status={spikeStatus.spike1}
        spike2Status={spikeStatus.spike2}
        spike3Status={spikeStatus.spike3}
        spike4Status={spikeStatus.spike4}
        elapsed={spikeStatus.elapsed}
        isRecording={recorderState?.isRecording ?? false}
      />

      {/* Pipeline status */}
      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          GPU: {pipeline.state.status} | Source: {pipeline.state.deviceSource}
          {pipeline.state.error ? ` | Error: ${pipeline.state.error}` : ''}
        </Text>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={isRunning ? stopPipeline : startPipeline}
        >
          <Text style={styles.buttonText}>
            {isRunning ? 'Stop' : 'Start Pipeline'}
          </Text>
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

      {/* Results display */}
      {results && (
        <ScrollView style={styles.results}>
          <Text style={styles.resultsTitle}>Spike Results</Text>
          <Text style={styles.resultsText}>
            {JSON.stringify(results, null, 2)}
          </Text>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  statusBar: {
    position: 'absolute',
    top: 44,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 4,
    padding: 8,
  },
  statusText: {
    color: '#aaa',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  controls: {
    position: 'absolute',
    bottom: 60,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 24,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: {
    backgroundColor: 'rgba(255,80,80,0.4)',
    borderColor: 'rgba(255,80,80,0.6)',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  results: {
    position: 'absolute',
    top: 220,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 8,
    padding: 12,
    maxHeight: 300,
  },
  resultsTitle: {
    color: '#0f0',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultsText: {
    color: '#0f0',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
});
