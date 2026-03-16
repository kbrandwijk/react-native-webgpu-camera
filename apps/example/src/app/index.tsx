import { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
  Modal,
  FlatList,
} from 'react-native';
import { Canvas, Fill, Group, Image as SkImage, Skia, Picture, createPicture } from '@shopify/react-native-skia';
import { useDerivedValue } from 'react-native-reanimated';
import { Asset } from 'expo-asset';
import { useCamera, useGPUFrameProcessor, useCameraFormats, GPUResource } from 'react-native-webgpu-camera';
import type { CameraFormat, ColorSpace } from 'react-native-webgpu-camera';
import { PASSTHROUGH_WGSL } from '@/shaders/passthrough.wgsl';
import { SOBEL_WGSL } from '@/shaders/sobel.wgsl';
import { SOBEL_COLOR_WGSL } from '@/shaders/sobel-color.wgsl';
import { HISTOGRAM_WGSL } from '@/shaders/histogram.wgsl';
import { LUT_WGSL } from '@/shaders/lut.wgsl';
import DepthEstimation from '@/components/DepthEstimation';

type ShaderMode =
  | { name: string; wgsl: readonly string[]; type: 'simple' }
  | { name: string; type: 'histogram' }
  | { name: string; type: 'histogram-onframe' }
  | { name: string; type: 'applelog' };

const SHADERS: ShaderMode[] = [
  { name: 'None', wgsl: [], type: 'simple' },
  { name: 'Passthrough', wgsl: [PASSTHROUGH_WGSL], type: 'simple' },
  { name: 'Sobel', wgsl: [SOBEL_WGSL], type: 'simple' },
  { name: 'Sobel Color', wgsl: [SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Multi-pass', wgsl: [SOBEL_WGSL, SOBEL_COLOR_WGSL], type: 'simple' },
  { name: 'Histogram', type: 'histogram' },
  { name: 'Hist (burn)', type: 'histogram-onframe' },
  { name: 'LUT', type: 'applelog' },
];

function CameraPreview({ shaderChain, format, colorSpace }: { shaderChain: readonly string[]; format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
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
          {error ? `Error: ${error}` : camera.isReady ? `${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramPreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
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
  const texPortraitW = camera.height; // e.g. 2160 in landscape
  const texPortraitH = camera.width;  // e.g. 3840 in landscape

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
          {error ? `Error: ${error}` : camera.isReady ? `Histogram ${camera.width}x${camera.height}` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function HistogramOnFramePreview({ format, colorSpace }: { format?: CameraFormat; colorSpace?: ColorSpace }) {
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
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
          {error ? `Error: ${error}` : camera.isReady ? `Hist-burn ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

function AppleLogPreview({ format, colorSpace, lutResource }: { format?: CameraFormat; colorSpace?: ColorSpace; lutResource: ReturnType<typeof GPUResource.texture3D> | null }) {
  console.log('[AppleLogPreview] MOUNT, lutResource:', !!lutResource, 'format:', format?.width, 'x', format?.height, 'colorSpace:', colorSpace);
  const { width: screenW, height: screenH } = useWindowDimensions();

  const camera = useCamera({
    device: 'back',
    format,
    colorSpace,
  });

  // TODO: Enable after native rebuild with .cube parser support
  // const resources = lutResource ? { lut: lutResource } : undefined;
  const resources = undefined;

  const { currentFrame, fps, displayFps, metrics, error } = useGPUFrameProcessor(camera, {
    resources,
    pipeline: (frame, res: any) => {
      'worklet';
      // TODO: Enable after native rebuild with .cube parser support
      // if (res?.lut) {
      //   frame.runShader(LUT_WGSL, { inputs: { lut: res.lut } });
      // }
    },
  });

  useDerivedValue(() => {
    const m = metrics.value;
    if (fps.value > 0 && m) {
      console.log(`[AppleLog] ${fps.value}fps (display=${displayFps.value}) | lock=${m.lockWait.toFixed(2)}ms import=${m.import.toFixed(2)}ms bind=${m.bindGroup.toFixed(2)}ms compute=${m.compute.toFixed(2)}ms buf=${m.buffers.toFixed(2)}ms skImg=${m.makeImage.toFixed(2)}ms total=${m.total.toFixed(2)}ms wall=${m.wall.toFixed(2)}ms`);
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
          {error ? `Error: ${error}` : camera.isReady ? `Apple Log ${camera.width}x${camera.height} @ ${camera.fps}fps` : 'Starting camera...'}
        </Text>
      </View>
    </>
  );
}

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [shaderIndex, setShaderIndex] = useState(0);
  const [showDepth, setShowDepth] = useState(false);
  const [lutResource, setLutResource] = useState<ReturnType<typeof GPUResource.texture3D> | null>(null);
  const shader = SHADERS[shaderIndex];

  // Load bundled .cube LUT on mount — pass file URI, native side parses .cube
  useEffect(() => {
    (async () => {
      const [asset] = await Asset.loadAsync(require('../../assets/AppleLogToRec709-v1.0.cube'));
      if (!asset.localUri) return;
      console.log('[LUT] asset localUri:', asset.localUri);
      setLutResource(GPUResource.texture3D(asset.localUri, {
        width: 0, height: 0, depth: 0,
        format: 'rgba32float',
      }));
    })();
  }, []);

  // Format enumeration — expand into one entry per (format, colorSpace) permutation
  const formats = useCameraFormats('back');
  type FormatEntry = { format: CameraFormat; colorSpace: ColorSpace };
  const formatEntries = formats.flatMap((f) =>
    (f.supportedColorSpaces ?? ['sRGB']).map((cs) => ({ format: f, colorSpace: cs as ColorSpace }))
  );
  const [selected, setSelected] = useState<FormatEntry | undefined>();

  // Auto-select best sRGB format
  useEffect(() => {
    if (formatEntries.length === 0) return;
    const best =
      formatEntries.find(e => e.colorSpace === 'sRGB' && e.format.width >= 3840 && e.format.maxFps >= 120) ??
      formatEntries.find(e => e.colorSpace === 'sRGB' && e.format.width >= 1920 && e.format.maxFps >= 60) ??
      formatEntries.find(e => e.colorSpace === 'sRGB') ??
      formatEntries[0];
    setSelected(best);
  }, [formats]);

  const selectedFormat = selected?.format;
  const selectedColorSpace = selected?.colorSpace ?? 'sRGB';

  if (showDepth) {
    return <DepthEstimation onBack={() => setShowDepth(false)} />;
  }

  const [showFormatPicker, setShowFormatPicker] = useState(false);

  const formatLabel = selected
    ? `${selected.format.width}x${selected.format.height} @${Math.round(selected.format.maxFps)} ${selected.colorSpace}`
    : 'Auto';

  return (
    <View style={styles.container}>
      {/* key forces re-mount because pipeline setup captures shader chain */}
      {isRunning && shader.type === 'histogram' && <HistogramPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'histogram-onframe' && <HistogramOnFramePreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'simple' && <CameraPreview key={`${shader.name}-${selectedColorSpace}`} shaderChain={shader.wgsl} format={selectedFormat} colorSpace={selectedColorSpace} />}
      {isRunning && shader.type === 'applelog' && <AppleLogPreview key={`${shader.name}-${selectedColorSpace}`} format={selectedFormat} colorSpace={selectedColorSpace} lutResource={lutResource} />}

      <View style={styles.controls}>
        {isRunning && (
          <Pressable
            style={styles.button}
            onPress={() => setShaderIndex((i) => (i + 1) % SHADERS.length)}
          >
            <Text style={styles.buttonText}>{shader.name} ({shaderIndex + 1}/{SHADERS.length})</Text>
          </Pressable>
        )}

        {!isRunning && (
          <Pressable
            style={styles.button}
            onPress={() => setShowFormatPicker(true)}
          >
            <Text style={styles.buttonText}>{formatLabel}</Text>
          </Pressable>
        )}

        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={() => { console.log('[CameraSpikeScreen] toggle isRunning, shader:', shader.name, 'type:', shader.type); setIsRunning(!isRunning); }}
        >
          <Text style={styles.buttonText}>{isRunning ? 'Stop' : 'Start Pipeline'}</Text>
        </Pressable>

        {!isRunning && (
          <Pressable style={styles.button} onPress={() => setShowDepth(true)}>
            <Text style={styles.buttonText}>Depth AI</Text>
          </Pressable>
        )}
      </View>

      <Modal visible={showFormatPicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Camera Format</Text>
            <FlatList
              data={formatEntries}
              keyExtractor={(e) => `${e.format.nativeHandle}-${e.colorSpace}`}
              renderItem={({ item: e }) => {
                const isSelected = selected?.format.nativeHandle === e.format.nativeHandle && selected?.colorSpace === e.colorSpace;
                return (
                  <Pressable
                    style={[styles.formatRow, isSelected && styles.formatRowSelected]}
                    onPress={() => { setSelected(e); setShowFormatPicker(false); }}
                  >
                    <Text style={[styles.formatText, isSelected && styles.formatTextSelected]}>
                      {e.format.width}x{e.format.height}
                    </Text>
                    <Text style={[styles.formatDetail, isSelected && styles.formatTextSelected]}>
                      {Math.round(e.format.minFps)}-{Math.round(e.format.maxFps)}fps {e.colorSpace}{e.format.isBinned ? ' bin' : ''}
                    </Text>
                  </Pressable>
                );
              }}
            />
            <Pressable style={styles.modalClose} onPress={() => setShowFormatPicker(false)}>
              <Text style={styles.buttonText}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
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
    flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 20,
    paddingHorizontal: 16, paddingVertical: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: { backgroundColor: 'rgba(255,80,80,0.4)', borderColor: 'rgba(255,80,80,0.6)' },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  modalOverlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a1a', borderTopLeftRadius: 16, borderTopRightRadius: 16,
    maxHeight: '60%', paddingBottom: 40,
  },
  modalTitle: {
    color: '#fff', fontSize: 18, fontWeight: '700',
    padding: 16, textAlign: 'center',
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.15)',
  },
  formatRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: 'rgba(255,255,255,0.08)',
  },
  formatRowSelected: { backgroundColor: 'rgba(80,140,255,0.25)' },
  formatText: { color: '#fff', fontSize: 16, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  formatDetail: { color: '#888', fontSize: 13, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  formatTextSelected: { color: '#6af' },
  modalClose: {
    alignSelf: 'center', marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 20,
    paddingHorizontal: 32, paddingVertical: 12,
  },
});
