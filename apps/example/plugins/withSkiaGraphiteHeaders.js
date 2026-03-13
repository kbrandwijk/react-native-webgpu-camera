/**
 * Expo config plugin that copies Dawn/WebGPU headers from
 * react-native-skia-graphite-headers into @shopify/react-native-skia
 * so the Graphite build can find webgpu/webgpu_cpp.h.
 */
const { withDangerousMod } = require("expo/config-plugins");
const path = require("path");
const fs = require("fs");

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

module.exports = function withSkiaGraphiteHeaders(config) {
  return withDangerousMod(config, [
    "ios",
    (config) => {
      const projectRoot = config.modRequest.projectRoot;

      // Resolve package paths — works with bun hoisted node_modules
      let headersDir, skiaDir;
      try {
        headersDir = path.dirname(
          require.resolve("react-native-skia-graphite-headers/package.json", {
            paths: [projectRoot],
          })
        );
        skiaDir = path.dirname(
          require.resolve("@shopify/react-native-skia/package.json", {
            paths: [projectRoot],
          })
        );
      } catch (e) {
        console.warn(
          "[withSkiaGraphiteHeaders] Could not resolve packages:",
          e.message
        );
        return config;
      }

      const src = path.join(headersDir, "libs", "skia", "cpp", "dawn");
      const dest = path.join(skiaDir, "cpp", "dawn");

      if (!fs.existsSync(src)) {
        console.warn(
          "[withSkiaGraphiteHeaders] Source headers not found at",
          src
        );
        return config;
      }

      console.log(`[withSkiaGraphiteHeaders] Copying Dawn headers`);
      console.log(`  from: ${src}`);
      console.log(`  to:   ${dest}`);
      copyDirSync(src, dest);

      return config;
    },
  ]);
};
