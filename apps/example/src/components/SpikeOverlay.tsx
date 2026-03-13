import React from 'react';
import { StyleSheet, View, Text as RNText } from 'react-native';

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
  return (
    <View style={styles.overlay} pointerEvents="none">
      <View style={styles.box}>
        <RNText style={styles.title}>Spike Validation</RNText>
        <RNText style={styles.line}>S1 (camera→GPU): {spike1Status}</RNText>
        <RNText style={styles.line}>S2 (compute): {spike2Status}</RNText>
        <RNText style={styles.line}>S3 (Skia): {spike3Status}</RNText>
        <RNText style={styles.line}>S4 (recorder): {spike4Status}</RNText>
        <RNText style={styles.line}>FPS: {fps.toFixed(1)} | {elapsed}s</RNText>
      </View>
      {isRecording && <View style={styles.recordingFlash} />}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
  },
  box: {
    position: 'absolute',
    top: 60,
    left: 16,
    width: 260,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8,
    padding: 12,
  },
  title: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 4,
  },
  line: {
    color: '#fff',
    fontSize: 13,
    lineHeight: 18,
  },
  recordingFlash: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255,0,0,0.15)',
  },
});
