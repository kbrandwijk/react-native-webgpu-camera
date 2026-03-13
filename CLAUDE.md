# react-native-webgpu-camera

## Project

React Native camera library replacing VisionCamera with WebGPU compute + Skia Graphite pipeline.
Currently in Phase 0 spike validation.

## Structure

- `packages/react-native-webgpu-camera/` — core library
  - `modules/webgpu-camera/` — local Expo module with Rust native code
  - `modules/webgpu-camera/rust/` — Rust crate (uniffi-bindgen-react-native)
- `apps/example/` — Expo 55 spike validation app
- `scripts/` — build scripts for Rust and Skia

## Tech Stack

- Bun workspaces monorepo
- Expo 55 (RN 0.83, New Architecture only)
- react-native-wgpu ^0.5.8 (WebGPU via Dawn)
- @shopify/react-native-skia with SK_GRAPHITE=1
- react-native-reanimated >=4.2.1 (worklet threading)
- Rust + uniffi-bindgen-react-native ^0.30.0

## Build Workflow

1. `./scripts/build-rust.sh` — cross-compile Rust
2. `./scripts/generate-bindings.sh` — UniFFI binding generation
3. `./scripts/build-skia-graphite.sh` — build Skia with Graphite
4. `eas build --platform ios --profile development` — EAS Build for device
5. TypeScript/WGSL changes use Metro fast refresh (no rebuild)

## Conventions

- Device builds via EAS Build, not `expo run:ios/android`
- Prebuilt Rust libraries committed to repo under `packages/react-native-webgpu-camera/prebuilt/`
- No CI/CD setup yet (spike phase)
- Spike validation on physical devices only
