# react-native-webgpu-camera

## Project

React Native camera library replacing VisionCamera with WebGPU compute + Skia Graphite pipeline.
Currently in Phase 0 spike validation.

## Structure

- `packages/react-native-webgpu-camera/` — core library
  - `modules/webgpu-camera/` — local Expo module (Dawn compute pipeline in C++/ObjC++)
  - `modules/webgpu-camera/ios/` — native iOS code (DawnComputePipeline, CameraStreamHostObject, Swift module)
- `packages/react-native-skia/` — git submodule of @shopify/react-native-skia (workspace-linked)
- `apps/example/` — Expo 55 spike validation app
- `docs/superpowers/specs/` — design docs and setup notes

## Tech Stack

- Bun workspaces monorepo
- Expo 55 (RN 0.83, New Architecture only)
- @shopify/react-native-skia with SK_GRAPHITE=1 (git submodule, Graphite bundles Dawn — provides `navigator.gpu` via its own JSI bridge, no separate react-native-wgpu needed)
- react-native-reanimated >=4.2.1 (worklet threading)
- Dawn (WebGPU) compute pipeline in C++/ObjC++ (no Rust — removed in favor of direct Dawn API)

## First-Time Setup

### 1. Install dependencies

```bash
bun install
```

### 2. Init Skia submodule

```bash
git submodule update --init --depth 1
```

### 3. Install Skia Graphite prebuilt binaries

```bash
bun run install:skia-graphite
```

This downloads prebuilt Graphite xcframeworks (iOS + macOS), Android libs, and checks out the matching Skia revision.

### 4. Fix Dawn headers (workaround for upstream bug)

The `install:skia-graphite` script has a bug where Dawn/WebGPU headers don't get copied due to a path mismatch in the archive extraction. Manual fix:

```bash
cd packages/react-native-skia/packages/skia
curl -sL "https://github.com/Shopify/react-native-skia/releases/download/skia-graphite-m142/skia-graphite-headers-skia-graphite-m142.tar.gz" -o /tmp/headers.tar.gz
tar xzf /tmp/headers.tar.gz --strip-components=3 -C . "packages/skia/cpp"
cd -
```

Verify: `ls packages/react-native-skia/packages/skia/cpp/dawn/include/webgpu/webgpu_cpp.h` should exist.

### Alternative: Build Skia from source

Instead of steps 3-4, you can build Skia from source with Graphite (~20-30 min). This guarantees header/binary match and avoids the Dawn headers bug:

```bash
bun run build:skia
```

## Build Workflow

### Building the app

1. `eas build --platform ios --profile development` — EAS Build for device (or `--local` for local builds)
2. TypeScript/WGSL changes use Metro fast refresh (no rebuild)

### TypeScript checks

```bash
bunx tsc
```

## Key Files for Skia Graphite Integration

- `apps/example/eas.json` — sets `SK_GRAPHITE=1` env var for EAS builds
- `packages/react-native-skia/packages/skia/react-native-skia.podspec` — includes `cpp/dawn/include` in `HEADER_SEARCH_PATHS`
- `packages/react-native-skia/packages/skia/cpp/dawn/include/` — Dawn/WebGPU headers (must be populated, see setup step 4)
- `packages/react-native-skia/packages/skia/libs/ios/` — prebuilt Skia Graphite xcframeworks
- `.easignore` — excludes Android libs, tests, and source from EAS upload (~2 GB savings)

## Known Issues / Workarounds

- **Dawn headers bug**: `install-skia-graphite.ts` `copyDawnHeaders()` silently fails due to archive path mismatch. Manual tar extraction required (see setup step 4).
- **macOS libs required**: Cannot exclude `libs/macos/` from `.easignore` — the Skia podspec `prepare_command` checks `hasIos && hasMacos` and falls back to npm packages if either is missing.
- **Upstream issue**: [Shopify/react-native-skia#3750](https://github.com/Shopify/react-native-skia/issues/3750) — podspec prepare_command doesn't copy headers from graphite-headers package.

## Conventions

- Device builds via EAS Build, not `expo run:ios/android`
- No CI/CD setup yet (spike phase)
- Spike validation on physical devices only
- Always commit `bun.lock`
