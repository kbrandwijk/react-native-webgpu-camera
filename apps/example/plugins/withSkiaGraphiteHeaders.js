/**
 * Expo config plugin that copies Skia private headers from the submodule's
 * externals/skia into @shopify/react-native-skia/cpp/skia/ so the Graphite
 * build can find ContextOptionsPriv.h and other private headers.
 *
 * With the submodule approach, Dawn/WebGPU headers and xcframeworks are
 * handled by `install-skia-graphite` / `build-skia`. This plugin only
 * needs to ensure Skia private headers are in place if they weren't
 * copied by the install script.
 *
 * Previous workarounds (sk_sp<const SkData> patching, graphite-headers
 * package copying) are no longer needed since we build/install from the
 * submodule where headers and binaries always match.
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

      let skiaDir;
      try {
        skiaDir = path.dirname(
          require.resolve("@shopify/react-native-skia/package.json", {
            paths: [projectRoot],
          })
        );
      } catch (e) {
        console.warn(
          "[withSkiaGraphiteHeaders] Could not resolve @shopify/react-native-skia:",
          e.message
        );
        return config;
      }

      // Check if Dawn headers are in place (from install-skia-graphite)
      const dawnDir = path.join(skiaDir, "cpp", "dawn");
      if (!fs.existsSync(dawnDir)) {
        console.warn(
          "[withSkiaGraphiteHeaders] Dawn headers not found at",
          dawnDir
        );
        console.warn(
          "  Run: bun run install:skia-graphite"
        );
      }

      // Check if Skia private headers are in place
      const skiaPrivateHeaders = path.join(
        skiaDir,
        "cpp",
        "skia",
        "src",
        "gpu",
        "graphite"
      );
      if (!fs.existsSync(skiaPrivateHeaders)) {
        // Try copying from externals/skia in the submodule
        const submoduleSkiaSrc = path.join(
          skiaDir,
          "..",
          "..",
          "externals",
          "skia",
          "src",
          "gpu",
          "graphite"
        );
        if (fs.existsSync(submoduleSkiaSrc)) {
          console.log(
            "[withSkiaGraphiteHeaders] Copying Skia private headers from submodule"
          );
          const dest = path.join(skiaDir, "cpp", "skia", "src", "gpu", "graphite");
          copyDirSync(submoduleSkiaSrc, dest);
        }
      }

      return config;
    },
  ]);
};
