import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage } from '@shopify/react-native-skia';
import { useCamera, useGPUFrameProcessor } from 'react-native-webgpu-camera';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import { SOBEL_COLOR_WGSL } from '@/shaders/sobel-color.wgsl';
import DepthEstimation from '@/components/DepthEstimation';

const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;

const SHADERS = [
  { name: 'Sobel', wgsl: [SOBEL_WGSL] },
  { name: 'Sobel Color', wgsl: [SOBEL_COLOR_WGSL] },
  { name: 'Multi-pass', wgsl: [SOBEL_WGSL, SOBEL_COLOR_WGSL] },
] as const;

function CameraPreview({ shaderChain }: { shaderChain: readonly string[] }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  const { currentFrame, error } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    for (const wgsl of shaderChain) {
      frame.runShader(wgsl);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="red" />
        <SkImage image={currentFrame} x={0} y={0} width={screenW} height={screenH} fit="contain" />
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Pipeline running' : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [shaderIndex, setShaderIndex] = useState(0);
  const [showDepth, setShowDepth] = useState(false);
  const shader = SHADERS[shaderIndex];

  if (showDepth) {
    return <DepthEstimation onBack={() => setShowDepth(false)} />;
  }

  return (
    <View style={styles.container}>
      {/* key forces re-mount because pipeline setup captures shader chain */}
      {isRunning && <CameraPreview key={shader.name} shaderChain={shader.wgsl} />}

      <View style={styles.controls}>
        {isRunning && (
          <Pressable
            style={styles.button}
            onPress={() => setShaderIndex((i) => (i + 1) % SHADERS.length)}
          >
            <Text style={styles.buttonText}>{shader.name}</Text>
          </Pressable>
        )}

        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={() => setIsRunning(!isRunning)}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>

        {!isRunning && (
          <Pressable style={styles.button} onPress={() => setShowDepth(true)}>
            <Text style={styles.buttonText}>Depth AI</Text>
          </Pressable>
        )}
      </View>
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
});
