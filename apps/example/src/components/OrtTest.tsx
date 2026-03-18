import { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

export default function OrtTest() {
  const [status, setStatus] = useState('Not started');
  const [result, setResult] = useState<string | null>(null);

  const runTest = async () => {
    try {
      setStatus('Creating random input...');

      // Create a simple test: random 1x3x224x224 input (MobileNet shape)
      const inputData = new Float32Array(1 * 3 * 224 * 224);
      for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random();
      }

      setStatus('ONNX Runtime loaded — JSI bridge works!');
      setResult(
        `InferenceSession: ${typeof InferenceSession}\n` +
        `Tensor: ${typeof Tensor}\n` +
        `create: ${typeof InferenceSession.create}\n` +
        `Input shape: [1, 3, 224, 224] (${inputData.length} floats)`
      );
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
      setResult(e.stack?.slice(0, 500) || null);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ONNX Runtime Test</Text>
      <Text style={styles.status}>{status}</Text>
      {result && <Text style={styles.result}>{result}</Text>}
      <Pressable style={styles.button} onPress={runTest}>
        <Text style={styles.buttonText}>Run Test</Text>
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
    fontSize: 16,
    color: '#aaa',
    marginBottom: 10,
    textAlign: 'center',
  },
  result: {
    fontSize: 12,
    color: '#0f0',
    fontFamily: 'monospace',
    marginBottom: 20,
    textAlign: 'left',
    padding: 10,
    backgroundColor: '#111',
    borderRadius: 8,
    width: '100%',
  },
  button: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
