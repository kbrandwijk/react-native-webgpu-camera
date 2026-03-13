# react-native-webgpu-camera

## Project

React Native camera library replacing VisionCamera with WebGPU compute + Skia Graphite pipeline.
Currently in Phase 0 spike validation.

## Structure

- `packages/react-native-webgpu-camera/` — core library
  - `modules/webgpu-camera/` — local Expo module with Rust native code
  - `modules/webgpu-camera/rust/` — Rust crate (uniffi-bindgen-react-native)
  - `modules/webgpu-camera/ios/rust/` — UniFFI-generated Swift bindings + static lib (copied by `generate-bindings.sh`)
- `packages/react-native-skia/` — git submodule of @shopify/react-native-skia (workspace-linked)
- `apps/example/` — Expo 55 spike validation app
  - `plugins/withSkiaGraphiteHeaders.js` — Expo config plugin (legacy, may no longer be needed)
- `scripts/` — build scripts for Rust
- `docs/superpowers/specs/` — design docs and setup notes

## Tech Stack

- Bun workspaces monorepo
- Expo 55 (RN 0.83, New Architecture only)
- @shopify/react-native-skia with SK_GRAPHITE=1 (git submodule, Graphite bundles Dawn — provides `navigator.gpu` via its own JSI bridge, no separate react-native-wgpu needed)
- react-native-reanimated >=4.2.1 (worklet threading)
- Rust + UniFFI 0.29 (binding generation via `cargo run --bin uniffi-bindgen`)

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

### 5. Build Rust (if Rust code exists)

```bash
./scripts/build-rust.sh
./scripts/generate-bindings.sh
```

### Alternative: Build Skia from source

Instead of steps 3-4, you can build Skia from source with Graphite (~20-30 min). This guarantees header/binary match and avoids the Dawn headers bug:

```bash
bun run build:skia
```

## Build Workflow

### After Rust changes

1. `./scripts/build-rust.sh` — cross-compile Rust for iOS (device + sim) and Android (arm64 + x86_64), copies `.a`/`.so` to `prebuilt/`
2. `./scripts/generate-bindings.sh` — generates UniFFI Swift/Kotlin bindings AND copies artifacts (Swift, FFI header, modulemap, static lib) into `modules/webgpu-camera/ios/rust/`

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

## UniFFI iOS Integration

Generated Swift bindings live in `modules/webgpu-camera/ios/rust/`:
- `webgpu_camera.swift` — generated Swift bindings
- `webgpu_cameraFFI.h` — C FFI header
- `webgpu_cameraFFI.modulemap` — Swift module map
- `libwebgpu_camera.a` — static library

The podspec (`ios/WebGPUCamera.podspec`) uses `vendored_libraries` and `SWIFT_INCLUDE_PATHS` to wire these in. No explicit Swift `import` needed — UniFFI free functions are visible at module scope.

## Known Issues / Workarounds

- **Dawn headers bug**: `install-skia-graphite.ts` `copyDawnHeaders()` silently fails due to archive path mismatch. Manual tar extraction required (see setup step 4).
- **macOS libs required**: Cannot exclude `libs/macos/` from `.easignore` — the Skia podspec `prepare_command` checks `hasIos && hasMacos` and falls back to npm packages if either is missing.
- **Upstream issue**: [Shopify/react-native-skia#3750](https://github.com/Shopify/react-native-skia/issues/3750) — podspec prepare_command doesn't copy headers from graphite-headers package.

## Conventions

- Device builds via EAS Build, not `expo run:ios/android`
- Prebuilt Rust libraries committed to repo under `packages/react-native-webgpu-camera/prebuilt/`
- No CI/CD setup yet (spike phase)
- Spike validation on physical devices only
- Always commit `bun.lock`
