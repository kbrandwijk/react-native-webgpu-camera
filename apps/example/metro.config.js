// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Register asset extensions
config.resolver.assetExts.push('cube');
config.resolver.assetExts.push('onnx');

// Stub out onnxruntime-web and onnxruntime-node for transformers.js
// We use onnxruntime-react-native instead, registered via Symbol.for('onnxruntime')
const path = require('path');
const emptyModule = path.resolve(__dirname, 'src/utils/empty-module.js');
const originalResolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === 'onnxruntime-web' || moduleName === 'onnxruntime-node') {
    return { type: 'sourceFile', filePath: emptyModule };
  }
  if (originalResolveRequest) {
    return originalResolveRequest(context, moduleName, platform);
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
