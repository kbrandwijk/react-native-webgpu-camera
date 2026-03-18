import { useState } from 'react';
import { View, Text, StyleSheet, Pressable, Platform, ActivityIndicator } from 'react-native';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Paths, File } from 'expo-file-system';
// @ts-ignore
import { requireNativeModule } from 'expo-modules-core';

const WebGPUCameraModule = requireNativeModule('WebGPUCamera') as {
  getDawnPointers(): { device: string; instance: string; dawnProcTable: string };
};

const MODEL_URL = 'https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx';
const MODEL_PATH = `${Paths.document.uri}/mobilenetv2-12.onnx`;

export default function OrtTest() {
  const [status, setStatus] = useState('Tap Run to start');
  const [result, setResult] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const runTest = async () => {
    if (running) return;
    setRunning(true);
    setResult(null);

    try {
      // Download model if not cached
      const modelFile = new File(MODEL_PATH);
      console.log('[ORT] Model path:', MODEL_PATH);
      console.log('[ORT] File exists:', modelFile.exists, modelFile.exists ? `(${modelFile.size} bytes)` : '');
      // Delete if too small (previous failed download)
      if (modelFile.exists && modelFile.size < 1000) {
        console.log('[ORT] Deleting corrupt file');
        modelFile.delete();
      }
      if (!modelFile.exists) {
        setStatus('Downloading MobileNetV2 (~5MB)...');
        console.log('[ORT] Downloading from:', MODEL_URL);
        const response = await fetch(MODEL_URL, {
          headers: { 'User-Agent': 'onnxruntime-react-native' },
        });
        console.log('[ORT] Response status:', response.status);
        const bytes = new Uint8Array(await response.arrayBuffer());
        console.log('[ORT] Downloaded bytes:', bytes.length);
        modelFile.write(bytes);
        console.log('[ORT] Written to disk, exists now:', modelFile.exists);
      }

      // Get Dawn device pointers for WebGPU EP
      const dawn = WebGPUCameraModule.getDawnPointers();
      console.log('[ORT] Dawn pointers:', JSON.stringify(dawn));

      // Create session with WebGPU EP
      console.log('[ORT] Creating session from:', MODEL_PATH);
      setStatus('Creating inference session (WebGPU)...');
      const epConfig = {
        name: 'webgpu' as const,
        deviceId: '1',
        device: dawn.device,
        instance: dawn.instance,
        dawnProcTable: dawn.dawnProcTable,
      };
      console.log('[ORT] EP config:', JSON.stringify(epConfig));
      const t0 = performance.now();
      let session;
      try {
        session = await InferenceSession.create(MODEL_PATH, {
          executionProviders: [epConfig as any],
        });
      } catch (e: any) {
        console.error('[ORT] Session create failed:', e.message);
        throw e;
      }
      const loadTime = performance.now() - t0;
      console.log('[ORT] Session created in', loadTime.toFixed(0), 'ms');

      const inputNames = session.inputNames;
      const outputNames = session.outputNames;
      console.log('[ORT] Inputs:', inputNames, 'Outputs:', outputNames);

      // Create random input (MobileNetV2: 1x3x224x224)
      setStatus('Running inference...');
      const inputData = new Float32Array(1 * 3 * 224 * 224);
      for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random();
      }
      const inputTensor = new Tensor('float32', inputData, [1, 3, 224, 224]);
      console.log('[ORT] Running inference...');

      const t1 = performance.now();
      const feeds: Record<string, Tensor> = {};
      feeds[inputNames[0]] = inputTensor;
      console.log('[ORT] Calling session.run()...');
      let output;
      try {
        output = await session.run(feeds);
      } catch (e: any) {
        console.error('[ORT] session.run() failed:', e.message, e.stack?.slice(0, 500));
        throw e;
      }
      const inferenceTime = performance.now() - t1;
      console.log('[ORT] Inference done in', inferenceTime.toFixed(0), 'ms');
      console.log('[ORT] Output keys:', Object.keys(output));

      const outputTensor = output[outputNames[0]];
      const outputData = outputTensor.data as Float32Array;

      // Top-5
      const scores = Array.from(outputData);
      const indexed = scores.map((s, i) => ({ score: s, index: i }));
      indexed.sort((a, b) => b.score - a.score);
      const top5 = indexed.slice(0, 5);

      setStatus('Done!');
      setResult(
        `Load: ${loadTime.toFixed(0)}ms | Inference: ${inferenceTime.toFixed(0)}ms\n` +
        `Input: ${inputNames[0]} [1,3,224,224]\n` +
        `Output: ${outputNames[0]} [${outputTensor.dims.join(',')}]\n\n` +
        `Top-5 (random input):\n` +
        top5.map((t, i) => `  ${i + 1}. class ${t.index}: ${t.score.toFixed(4)}`).join('\n')
      );
    } catch (e: any) {
      console.error('[ORT] Error:', e.message, e.stack?.slice(0, 500));
      setStatus(`Error: ${e.message}`);
      setResult(`${e.message}\n\n${e.stack?.slice(0, 800) || 'no stack'}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ONNX Runtime Test</Text>
      <Text style={styles.status}>{status}</Text>
      {running && <ActivityIndicator size="small" color="#aaa" style={{ marginTop: 8 }} />}
      {result && <Text style={styles.result}>{result}</Text>}
      <Pressable style={[styles.button, running && styles.buttonDisabled]} onPress={runTest}>
        <Text style={styles.buttonText}>{running ? 'Running...' : 'Run Model'}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#000',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 20,
  },
  status: {
    fontSize: 14,
    color: '#aaa',
    marginBottom: 10,
    textAlign: 'center',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  result: {
    fontSize: 12,
    color: '#0f0',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    marginBottom: 20,
    textAlign: 'left',
    padding: 12,
    backgroundColor: '#111',
    borderRadius: 8,
    width: '100%',
  },
  button: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 8,
  },
  buttonDisabled: { opacity: 0.4 },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
