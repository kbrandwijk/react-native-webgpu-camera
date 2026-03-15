import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage, Skia, Picture, createPicture } from '@shopify/react-native-skia';
import { useDerivedValue } from 'react-native-reanimated';
import { useCamera, useGPUFrameProcessor } from 'react-native-webgpu-camera';
import { PASSTHROUGH_WGSL } from '@/shaders/passthrough.wgsl';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import { SOBEL_COLOR_WGSL } from '@/shaders/sobel-color.wgsl';
import { HISTOGRAM_WGSL } from '@/shaders/histogram.wgsl';
import DepthEstimation from '@/components/DepthEstimation';

const CAMERA_WIDTH = 3840;
const CAMERA_HEIGHT = 2160;
const CAMERA_FPS = 120;

type ShaderMode =
  | { name: string; wgsl: readonly string[]; type: 'simple' }
  | { name: string; type: 'histogram' }
  | { name: string; type: 'histogram-onframe' };

const SHADERS: ShaderMode[] = [
  { name: 'None', wgsl: [], type: 'simple' },
  { name: 'Passthrough', wgsl: [PASSTHROUGH_WGSL], type: 'simple' },
  { name: 'Sobel', wgsl: [SOBEL_WGSL], type: 'simple' },
  { name: 'Sobel Color', wgsl: [SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Multi-pass', wgsl: [SOBEL_WGSL, SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Histogram', type: 'histogram' },
  { name: 'Hist (burn)', type: 'histogram-onframe' },
];

function CameraPreview({ shaderChain }: { shaderChain: readonly string[] }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, (frame) => {
    'worklet';
    for (const wgsl of shaderChain) {
      frame.runShader(wgsl);
    }
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[FPS] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms bind=${m.bindGroup.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms buf=${m.buffers.toFixed(2)}ms skImg=${m.makeImage.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <Group transform={[
          { translateX: screenW },
          { rotate: Math.PI / 2 },
        ]}>
          <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Pipeline running' : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramPreview() {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  // Histogram with buffer readback (sync required until async readback is implemented)
  const { currentFrame, buffers, error } = useGPUFrameProcessor(camera, {
    sync: false,
    pipeline: (frame) => {
      'worklet';
      const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
      return { hist };
    },
  });

  const emptyPicture = createPicture(() => {});

  // Portrait texture dimensions (same coordinate space as onFrame)
  const texPortraitW = CAMERA_HEIGHT; // 2160
  const texPortraitH = CAMERA_WIDTH;  // 3840

  // Scale factor: screen points → texture pixels
  const overlayScale = screenW / texPortraitW;

  // Create a Skia Picture from buffer data — draws in texture coordinates
  const histPicture = useDerivedValue(() => {
    const hist = buffers.value.hist as Uint32Array | null;
    // console.log('[Histogram oveblue is rlay] buffer data:', hist ? `${hist.length} elements, first=${hist[0]}` : 'null');
    if (!hist) return emptyPicture;

    // Same coordinates as onFrame burn-in
    const histW = 1200;
    const histH = 500;
    const x0 = texPortraitW - histW - 60;
    const y0 = texPortraitH - histH - 120;

    return createPicture((canvas) => {
      // Background
      const bgPaint = Skia.Paint();
      bgPaint.setColor(Skia.Color('black'));
      bgPaint.setAlphaf(0.7);
      canvas.drawRect(Skia.XYWHRect(x0 - 8, y0 - 8, histW + 16, histH + 16), bgPaint);

      // Find max for normalization
      let maxVal = 1;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > maxVal) maxVal = hist[i];
      }

      // Bars
      const barPaint = Skia.Paint();
      barPaint.setColor(Skia.Color('white'));
      const barW = histW / 256;
      for (let i = 0; i < 256; i++) {
        const barH = (hist[i] / maxVal) * histH;
        if (barH < 1) continue;
        canvas.drawRect(
          Skia.XYWHRect(x0 + i * barW, y0 + histH - barH, Math.max(barW, 1), barH),
          barPaint,
        );
      }
    });
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <Group transform={[
          { translateX: screenW },
          { rotate: Math.PI / 2 },
        ]}>
          <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
        </Group>
        {/* Histogram overlay — drawn in texture coords, scaled to screen */}
        <Group transform={[{ scale: overlayScale }]}>
          <Picture picture={histPicture} />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Histogram running' : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramOnFramePreview() {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT,
    fps: CAMERA_FPS,
  });

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    sync: false,
    pipeline: (frame) => {
      'worklet';
      const hist = frame.runShader(HISTOGRAM_WGSL, { output: Uint32Array, count: 256 });
      return { hist };
    },
    onFrame: (() => {
      const bgPaint = Skia.Paint();
      bgPaint.setColor(Skia.Color('black'));
      bgPaint.setAlphaf(0.6);
      const barPaint = Skia.Paint();
      barPaint.setColor(Skia.Color('white'));
      // Cache normalized bar heights so we can redraw even when hist is null
      const barHeights = new Float32Array(256);
      let hasData = false;

      return (frame: any, { hist }: any) => {
        'worklet';
        // Update cached heights when new data arrives
        if (hist) {
          let maxVal = 1;
          for (let i = 0; i < 256; i++) {
            if (hist[i] > maxVal) maxVal = hist[i];
          }
          for (let i = 0; i < 256; i++) {
            barHeights[i] = hist[i] / maxVal;
          }
          hasData = true;
        }
        if (!hasData) return;

        // Always draw — compute overwrites finalTex every frame
        const histW = 1200;
        const histH = 500;
        const x0 = (frame.width - histW) / 2;
        const y0 = (frame.height - histH) / 2;

        frame.canvas.drawRect(Skia.XYWHRect(x0 - 8, y0 - 8, histW + 16, histH + 16), bgPaint);

        const barW = histW / 256;
        for (let i = 0; i < 256; i++) {
          const barH = barHeights[i] * histH;
          if (barH < 1) continue;
          frame.canvas.drawRect(
            Skia.XYWHRect(x0 + i * barW, y0 + histH - barH, Math.max(barW, 1), barH),
            barPaint,
          );
        }
      };
    })(),
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[Hist-burn] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms bind=${m.bindGroup.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms buf=${m.buffers.toFixed(2)}ms skImg=${m.makeImage.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
    }
  });

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        <Group transform={[
          { translateX: screenW },
          { rotate: Math.PI / 2 },
        ]}>
          <SkImage image={currentFrame} x={0} y={0} width={screenH} height={screenW} fit="cover" />
        </Group>
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>
          {error ? `Error: ${error}` : camera.isReady ? 'Histogram (burn-in) running' : 'Starting camera...'}
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
      {isRunning && shader.type === 'histogram' && <HistogramPreview key={shader.name} />}
      {isRunning && shader.type === 'histogram-onframe' && <HistogramOnFramePreview key={shader.name} />}
      {isRunning && shader.type === 'simple' && <CameraPreview key={shader.name} shaderChain={shader.wgsl} />}

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
