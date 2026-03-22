# react-native-webgpu-camera

## Project

React Native camera library replacing VisionCamera with WebGPU compute + Skia Graphite pipeline.
Currently in Phase 0 spike validation.

## Structure

- `packages/react-native-webgpu-camera/` — core library
  - `modules/webgpu-camera/` — local Expo module (Dawn compute pipeline in C++/ObjC++)
  - `modules/webgpu-camera/ios/` — native iOS code (DawnComputePipeline, CameraStreamHostObject, Swift module)
- `packages/react-native-skia/` — git submodule of @shopify/react-native-skia (workspace-linked)
- `packages/gpu-video-shaders/` — webgpu-video-shaders package (separate repo, gitignored)
- `apps/example/` — Expo 55 spike validation app
- `apps/web/` — Browser WebGPU demo app for webgpu-video-shaders (separate, gitignored)
- `docs/superpowers/specs/` — design docs and setup notes

## Tech Stack

- Bun workspaces monorepo
- Expo 55 (RN 0.83, New Architecture only)
- @shopify/react-native-skia with Graphite (git submodule, Graphite bundles Dawn — provides `navigator.gpu`, `WebGPUCanvas`, and `Skia.getDevice()` via JSI)
- react-native-reanimated >=4.2.1 (worklet threading)
- Dawn (WebGPU) compute pipeline in C++/ObjC++ (no Rust — removed in favor of direct Dawn API)
- webgpu-video-shaders — libplacebo video processing algorithms as WGSL shader generators

## First-Time Setup

### 1. Install dependencies

```bash
bun install
```

### 2. Init Skia submodule

```bash
git submodule update --init --depth 1
```

### 3. Install Skia Graphite (choose one)

**Option A: Prebuilt binaries (fast, ~2 min)**

```bash
bun run install:skia-graphite
```

Downloads prebuilt Graphite xcframeworks, Dawn/WebGPU headers, and creates the `.graphite` marker file. The updated script (PR #3757) handles Dawn headers correctly — no manual fix needed.

**Option B: Build from source (~20-30 min)**

```bash
bun run build:skia
```

Guarantees header/binary match. Required if prebuilt binaries aren't available for your platform.

## Build Workflow

### Building the app

1. `eas build --platform ios --profile development` — EAS Build for device (or `--local` for local builds)
2. TypeScript/WGSL changes use Metro fast refresh (no rebuild)

### TypeScript checks

```bash
bunx tsc
```

## Key Files

- `.easignore` — excludes large files from EAS upload

## Conventions

- Device builds via EAS Build, not `expo run:ios/android`
- No CI/CD setup yet (spike phase)
- Spike validation on physical devices only
- Always commit `bun.lock`
