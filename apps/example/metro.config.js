// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Register .cube as an asset extension so require() resolves LUT files
config.resolver.assetExts.push('cube');

module.exports = config;
