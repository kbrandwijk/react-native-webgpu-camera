import React from 'react';
import { StyleSheet } from 'react-native';
import { Canvas, Text, RoundedRect, useFont } from '@shopify/react-native-skia';

interface SpikeOverlayProps {
  fps: number;
  spike1Status: string;
  spike2Status: string;
  spike3Status: string;
  spike4Status: string;
  elapsed: number;
  isRecording: boolean;
}

// Spike 3 validation: test Skia Graphite shared context
export function validateGraphiteSharedContext(): {
  graphiteActive: boolean;
  sharedDevice: boolean;
  textureRoundTrip: boolean;
  path: 'graphite-direct' | 'graphite-composite' | 'ganesh-fallback' | 'unknown';
} {
  const result = {
    graphiteActive: false,
    sharedDevice: false,
    textureRoundTrip: false,
    path: 'unknown' as const,
  };

  // Test 1: Is navigator.gpu available? (Graphite installs it)
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    result.graphiteActive = true;
    console.log('[Spike3] navigator.gpu exists — Graphite likely active');
  } else {
    console.log('[Spike3] navigator.gpu NOT available — Graphite not active');
    return { ...result, path: 'ganesh-fallback' };
  }

  try {
    const g = globalThis as any;
    if (g.__SKIA_GRAPHITE_ACTIVE__ === true) {
      result.sharedDevice = true;
      result.textureRoundTrip = true;
      return { ...result, path: 'graphite-direct' };
    }

    result.sharedDevice = false;
    result.textureRoundTrip = false;
    return { ...result, path: 'graphite-composite' };
  } catch {
    return { ...result, path: 'ganesh-fallback' };
  }
}

export function SpikeOverlay({
  fps,
  spike1Status,
  spike2Status,
  spike3Status,
  spike4Status,
  elapsed,
  isRecording,
}: SpikeOverlayProps) {
  const font = useFont(null, 13);

  if (!font) return null;

  const x = 16;
  const y = 60;
  const lineHeight = 18;

  return (
    <Canvas style={styles.overlay} pointerEvents="none">
      <RoundedRect x={x} y={y} width={260} height={140} r={8} color="rgba(0,0,0,0.6)" />
      <Text x={x + 12} y={y + 20} text="Spike Validation" font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight} text={`S1 (camera→GPU): ${spike1Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 2} text={`S2 (compute): ${spike2Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 3} text={`S3 (Skia): ${spike3Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 4} text={`S4 (recorder): ${spike4Status}`} font={font} color="white" />
      <Text x={x + 12} y={y + 20 + lineHeight * 5} text={`FPS: ${fps.toFixed(1)} | ${elapsed}s`} font={font} color="white" />
      {isRecording && (
        <RoundedRect x={0} y={0} width={9999} height={9999} r={0} color="rgba(255,0,0,0.15)" />
      )}
    </Canvas>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
});
