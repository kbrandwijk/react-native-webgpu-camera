import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  ScrollView,
} from 'react-native';
import { useSpikeMetrics, SpikeResults } from '@/hooks/useSpikeMetrics';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import * as FileSystem from 'expo-file-system';

// These imports will be wired up during spike implementation:
// import { Canvas } from 'react-native-wgpu';
// import { runOnUI } from 'react-native-reanimated';
// import { WebGPUCameraModule } from 'react-native-webgpu-camera';

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
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

  const startPipeline = useCallback(async () => {
    setIsRunning(true);
    setResults(null);
    metrics.reset();

    // TODO: Wire up the full pipeline during spike implementation:
    // 1. Start camera via Rust module: WebGPUCameraModule.startCameraPreview('back', 1920, 1080)
    // 2. Get GPUDevice from react-native-wgpu Canvas onCreateSurface
    // 3. Create Sobel compute pipeline from SOBEL_WGSL
    // 4. In worklet render loop:
    //    a. Poll Rust for frame handle (Spike 1)
    //    b. Import as GPUTexture or fall back to writeTexture
    //    c. Dispatch Sobel compute shader (Spike 2)
    //    d. Draw Skia overlay (Spike 3)
    //    e. Render pass to canvas + recorder if active (Spike 4)
    //    f. Record timing via metrics.recordFrame()
    // 5. After 60 seconds, call metrics.logSummary() and stop

    console.log('[CameraSpikeScreen] Pipeline started (shell only - wire up during spike implementation)');
    console.log('[CameraSpikeScreen] Sobel shader loaded:', SOBEL_WGSL.substring(0, 50) + '...');
  }, [metrics]);

  const stopPipeline = useCallback(() => {
    // TODO: WebGPUCameraModule.stopCameraPreview()
    setIsRunning(false);

    metrics.logSummary({
      spike1Path: spikeStatus.spike1 as SpikeResults['spike1Path'],
      spike2Path: spikeStatus.spike2 as SpikeResults['spike2Path'],
      spike3Path: spikeStatus.spike3 as SpikeResults['spike3Path'],
      spike4Path: spikeStatus.spike4 as SpikeResults['spike4Path'],
    });

    const summary = metrics.getSummary();
    if (summary) {
      setResults(summary);
    }
  }, [metrics, spikeStatus]);

  const startRecording = useCallback(async () => {
    const outputPath = `${FileSystem.documentDirectory}spike4_test.mp4`;
    setIsRecording(true);

    // TODO: Wire up during spike implementation:
    // const surfaceHandle = WebGPUCameraModule.startTestRecorder(outputPath, 1920, 1080);
    // if (surfaceHandle !== 0) {
    //   setSpikeStatus(s => ({ ...s, spike4: 'surface-record' }));
    // } else {
    //   setSpikeStatus(s => ({ ...s, spike4: 'readback-record' }));
    // }

    console.log(`[CameraSpikeScreen] Recording to: ${outputPath}`);

    // Stop after 5 seconds
    setTimeout(() => {
      // TODO: const filePath = WebGPUCameraModule.stopTestRecorder();
      setIsRecording(false);
      console.log('[CameraSpikeScreen] Recording stopped (stub)');
    }, 5000);
  }, []);

  return (
    <View style={styles.container}>
      {/* WebGPU Canvas will go here — full screen camera preview + compute output */}
      {/* <Canvas style={StyleSheet.absoluteFill} /> */}
      <View style={[StyleSheet.absoluteFill, styles.placeholder]}>
        <Text style={styles.placeholderText}>
          WebGPU Canvas{'\n'}(wire up during spike implementation)
        </Text>
      </View>

      {/* Status overlay */}
      <View style={styles.overlay}>
        <Text style={styles.overlayTitle}>Spike Validation</Text>
        <Text style={styles.statusText}>
          Spike 1 (zero-copy): {spikeStatus.spike1}
        </Text>
        <Text style={styles.statusText}>
          Spike 2 (worklet compute): {spikeStatus.spike2}
        </Text>
        <Text style={styles.statusText}>
          Spike 3 (Skia Graphite): {spikeStatus.spike3}
        </Text>
        <Text style={styles.statusText}>
          Spike 4 (recorder): {spikeStatus.spike4}
        </Text>
        <Text style={styles.statusText}>
          FPS: {spikeStatus.fps.toFixed(1)} | Elapsed: {spikeStatus.elapsed}s
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
            style={[styles.button, isRecording && styles.buttonActive]}
            onPress={startRecording}
            disabled={isRecording}
          >
            <Text style={styles.buttonText}>
              {isRecording ? 'Recording...' : 'Record 5s'}
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
  placeholder: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#111',
  },
  placeholderText: {
    color: '#444',
    fontSize: 18,
    textAlign: 'center',
  },
  overlay: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8,
    padding: 12,
  },
  overlayTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
  },
  statusText: {
    color: '#fff',
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    marginBottom: 4,
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
    top: 250,
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
