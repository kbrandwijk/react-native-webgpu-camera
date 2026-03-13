/**
 * Expo config plugin that copies Dawn/WebGPU and Skia private headers from
 * react-native-skia-graphite-headers into @shopify/react-native-skia/cpp/
 * so the Graphite build can find webgpu/webgpu_cpp.h, ContextOptionsPriv.h, etc.
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

      const headersCppDir = path.join(headersDir, "libs", "skia", "cpp");
      const skiaCppDir = path.join(skiaDir, "cpp");

      // Copy both dawn/ (WebGPU headers) and skia/ (Skia private headers)
      for (const subdir of ["dawn", "skia"]) {
        const src = path.join(headersCppDir, subdir);
        const dest = path.join(skiaCppDir, subdir);
        if (fs.existsSync(src)) {
          console.log(`[withSkiaGraphiteHeaders] Copying ${subdir} headers`);
          console.log(`  from: ${src}`);
          console.log(`  to:   ${dest}`);
          copyDirSync(src, dest);
        }
      }

      return config;
    },
  ]);
};
