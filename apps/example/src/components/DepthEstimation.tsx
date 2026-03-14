import { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
  useWindowDimensions,
  ActivityIndicator,
} from 'react-native';
import { Canvas, Image as SkImage, Fill } from '@shopify/react-native-skia';
import { Skia, AlphaType, ColorType } from '@shopify/react-native-skia';

type Status = 'idle' | 'loading-model' | 'running' | 'done' | 'error';

// Test image URL — aerial photo with strong depth cues
const TEST_IMAGE_URL =
  'https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Empire_State_Building_%28aerial_view%29.jpg/800px-Empire_State_Building_%28aerial_view%29.jpg';

/**
 * Convert a transformers.js RawImage (grayscale or RGB) to an SkImage
 * by building RGBA pixel data and using Skia.Image.MakeImage.
 */
function rawImageToSkia(raw: { data: Uint8Array | Uint8ClampedArray; width: number; height: number; channels: number }) {
  const { width, height, channels, data } = raw;
  const pixelCount = width * height;
  const rgba = new Uint8Array(pixelCount * 4);

  if (channels === 1) {
    // Grayscale → RGBA
    for (let i = 0; i < pixelCount; i++) {
      const v = data[i];
      rgba[i * 4 + 0] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
    }
  } else if (channels === 3) {
    // RGB → RGBA
    for (let i = 0; i < pixelCount; i++) {
      rgba[i * 4 + 0] = data[i * 3 + 0];
      rgba[i * 4 + 1] = data[i * 3 + 1];
      rgba[i * 4 + 2] = data[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
  } else if (channels === 4) {
    rgba.set(data);
  }

  const skData = Skia.Data.fromBytes(rgba);
  return Skia.Image.MakeImage(
    { width, height, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Opaque },
    skData,
    width * 4,
  );
}

export default function DepthEstimation({ onBack }: { onBack: () => void }) {
  const { width: screenW, height: screenH } = useWindowDimensions();
  const [status, setStatus] = useState<Status>('idle');
  const [statusText, setStatusText] = useState('Tap "Run Depth" to start');
  const [depthImage, setDepthImage] = useState<ReturnType<typeof Skia.Image.MakeImage> | null>(null);
  const [sourceImage, setSourceImage] = useState<ReturnType<typeof Skia.Image.MakeImage> | null>(null);
  const [showDepth, setShowDepth] = useState(true);

  const runDepthEstimation = useCallback(async () => {
    try {
      setStatus('loading-model');
      setStatusText('Loading transformers.js + depth model (first run downloads ~50MB)...');

      const modelStart = performance.now();

      const { pipeline, RawImage, env } = await import('@huggingface/transformers');

      // Disable local model caching (not available in RN)
      env.allowLocalModels = false;

      setStatusText('Downloading model weights via WebGPU...');

      const depthEstimator = await pipeline(
        'depth-estimation',
        'onnx-community/depth-anything-v2-small',
        { device: 'webgpu' },
      );

      const modelTime = performance.now() - modelStart;
      setStatusText(`Model loaded in ${(modelTime / 1000).toFixed(1)}s. Fetching test image...`);
      setStatus('running');

      const inferenceStart = performance.now();

      // Fetch and decode test image
      const image = await RawImage.read(TEST_IMAGE_URL);
      setStatusText(`Image loaded (${image.width}x${image.height}). Running inference...`);

      // Run depth estimation
      const { depth, predicted_depth } = await depthEstimator(image);

      const inferenceTime = performance.now() - inferenceStart;

      // Convert depth output (grayscale RawImage) to Skia image
      const skDepth = rawImageToSkia(depth);
      setDepthImage(skDepth);

      // Convert source image to Skia image for comparison
      const skSource = rawImageToSkia(image);
      setSourceImage(skSource);

      setStatus('done');
      setStatusText(
        `Model: ${(modelTime / 1000).toFixed(1)}s | Inference: ${(inferenceTime / 1000).toFixed(1)}s | Depth: ${depth.width}x${depth.height}`,
      );
    } catch (e: any) {
      console.error('[DepthEstimation]', e);
      setStatus('error');
      setStatusText(`Error: ${e.message}`);
    }
  }, []);

  const displayImage = showDepth ? depthImage : sourceImage;

  return (
    <View style={styles.container}>
      <Canvas style={StyleSheet.absoluteFill}>
        <Fill color="black" />
        {displayImage && (
          <SkImage
            image={displayImage}
            x={0}
            y={0}
            width={screenW}
            height={screenH}
            fit="contain"
          />
        )}
      </Canvas>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>{statusText}</Text>
        {(status === 'loading-model' || status === 'running') && (
          <ActivityIndicator size="small" color="#aaa" style={{ marginTop: 4 }} />
        )}
      </View>

      <View style={styles.controls}>
        <Pressable style={styles.button} onPress={onBack}>
          <Text style={styles.buttonText}>Back</Text>
        </Pressable>

        {status === 'done' && (
          <Pressable
            style={styles.button}
            onPress={() => setShowDepth((v) => !v)}
          >
            <Text style={styles.buttonText}>
              {showDepth ? 'Source' : 'Depth'}
            </Text>
          </Pressable>
        )}

        <Pressable
          style={[styles.button, (status === 'loading-model' || status === 'running') && styles.buttonDisabled]}
          onPress={status !== 'loading-model' && status !== 'running' ? runDepthEstimation : undefined}
        >
          <Text style={styles.buttonText}>
            {status === 'idle' ? 'Run Depth' : status === 'done' || status === 'error' ? 'Run Again' : 'Running...'}
          </Text>
        </Pressable>
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
  statusText: {
    color: '#aaa', fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  controls: {
    position: 'absolute', bottom: 60, left: 16, right: 16,
    flexDirection: 'row', justifyContent: 'center', gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 24,
    paddingHorizontal: 24, paddingVertical: 14,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonDisabled: { opacity: 0.4 },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
