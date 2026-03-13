# Spike Scaffold Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the monorepo, core package, Expo module with Rust/UniFFI, and example app with a camera screen shell. This plan builds the complete project structure with stub implementations. The actual spike validation (wiring the WebGPU render loop, implementing Rust camera code, testing on physical devices) is the follow-up implementation work.

**Architecture:** Bun workspaces monorepo. Core library at `packages/react-native-webgpu-camera` containing a local Expo module with Rust native code via uniffi-bindgen-react-native. Example Expo 55 app at `apps/example` with a single camera screen. Device builds via EAS Build.

**Tech Stack:** Expo 55 (RN 0.83), react-native-wgpu 0.5.8, @shopify/react-native-skia (Graphite), react-native-reanimated 4.2.1+, Rust + UniFFI, Bun.

**Spec:** `docs/superpowers/specs/2026-03-12-spike-scaffold-design.md`

---

## File Map

### Root (monorepo config)

| File | Action | Purpose |
|------|--------|---------|
| `package.json` | Create | Bun workspace root with workspace definitions and scripts |
| `tsconfig.json` | Create | Base TypeScript config shared by all packages |
| `.gitignore` | Create | Ignore node_modules, native build artifacts, bun.lock patterns |
| `CLAUDE.md` | Create | Project conventions for agentic workers |
| `scripts/build-rust.sh` | Create | Cross-compile Rust for iOS and Android targets |
| `scripts/generate-bindings.sh` | Create | Run UniFFI binding generation |
| `scripts/build-skia-graphite.sh` | Create | Wrapper: `SK_GRAPHITE=1 bun run build-skia` |

### Core package (`packages/react-native-webgpu-camera/`)

| File | Action | Purpose |
|------|--------|---------|
| `package.json` | Create | Package manifest with dependencies and exports |
| `tsconfig.json` | Create | TypeScript config extending root |
| `src/index.ts` | Create | Public API re-exports (minimal for spike) |

### Expo module (`packages/react-native-webgpu-camera/modules/webgpu-camera/`)

| File | Action | Purpose |
|------|--------|---------|
| `expo-module.config.json` | Create | Expo autolinking config |
| `src/WebGPUCameraModule.ts` | Create | TypeScript interface to native module |
| `ios/WebGPUCameraModule.swift` | Create | iOS Swift bridge calling Rust |
| `android/src/main/java/.../WebGPUCameraModule.kt` | Create | Android Kotlin bridge calling Rust |
| `index.ts` | Create | Module entry point |

### Rust crate (`packages/react-native-webgpu-camera/modules/webgpu-camera/rust/`)

| File | Action | Purpose |
|------|--------|---------|
| `Cargo.toml` | Create | Rust crate config with uniffi dependency |
| `ubrn.config.yaml` | Create | uniffi-bindgen-react-native build config |
| `src/lib.rs` | Create | UniFFI-annotated API (spike scope only) |
| `src/camera/mod.rs` | Create | Platform-agnostic camera trait |
| `src/camera/ios.rs` | Create | AVCaptureSession wrapper |
| `src/camera/android.rs` | Create | Camera2 wrapper |
| `src/recorder/mod.rs` | Create | Platform-agnostic recorder trait |
| `src/recorder/ios.rs` | Create | AVAssetWriter wrapper |
| `src/recorder/android.rs` | Create | MediaRecorder wrapper |
| `src/thermal.rs` | Create | Thermal state query per platform |

### Example app (`apps/example/`)

| File | Action | Purpose |
|------|--------|---------|
| `package.json` | Create | App manifest linking to core package |
| `app.json` | Create | Expo config |
| `eas.json` | Create | EAS Build profiles |
| `tsconfig.json` | Create | TypeScript config |
| `app/_layout.tsx` | Create | Root layout (Expo Router) |
| `app/(tabs)/_layout.tsx` | Create | Tab navigation layout |
| `app/(tabs)/index.tsx` | Create | Camera spike screen |
| `src/shaders/sobel.wgsl.ts` | Create | Sobel edge detection compute shader string |
| `src/hooks/useSpikeMetrics.ts` | Create | FPS, frame drop, memory tracking |

---

## Chunk 1: Monorepo Foundation

### Task 1: Initialize git and root workspace

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `.gitignore`
- Create: `CLAUDE.md`

- [ ] **Step 1: Create root `package.json`**

```json
{
  "name": "react-native-webgpu-camera-monorepo",
  "private": true,
  "workspaces": [
    "packages/*",
    "apps/*"
  ],
  "scripts": {
    "build": "bun run --filter 'react-native-webgpu-camera' build",
    "build:rust": "./scripts/build-rust.sh",
    "build:bindings": "./scripts/generate-bindings.sh",
    "build:skia": "./scripts/build-skia-graphite.sh",
    "lint": "bun run --filter '*' lint",
    "typecheck": "bun run --filter '*' typecheck",
    "example:ios": "cd apps/example && eas build --platform ios --profile development",
    "example:android": "cd apps/example && eas build --platform android --profile development"
  }
}
```

- [ ] **Step 2: Create root `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "esnext",
    "module": "esnext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "dist",
    "baseUrl": "."
  },
  "exclude": ["node_modules", "dist", "android", "ios"]
}
```

- [ ] **Step 3: Create `.gitignore`**

```
node_modules/
dist/
.expo/
ios/
android/
*.jsbundle
bun.lock

# Rust build artifacts (NOT the prebuilts — those are committed)
**/rust/target/

# Prebuilt Rust libraries ARE committed (for EAS Build compatibility)
# !packages/react-native-webgpu-camera/prebuilt/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# EAS
.eas/
```

- [ ] **Step 4: Create `CLAUDE.md`**

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add package.json tsconfig.json .gitignore CLAUDE.md
git commit -m "feat: initialize monorepo root with bun workspaces"
```

---

### Task 2: Create build scripts

**Files:**
- Create: `scripts/build-rust.sh`
- Create: `scripts/generate-bindings.sh`
- Create: `scripts/build-skia-graphite.sh`

- [ ] **Step 1: Create `scripts/build-rust.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Cross-compile Rust for iOS and Android targets
# Run from monorepo root

RUST_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera/rust"
PREBUILT_DIR="packages/react-native-webgpu-camera/prebuilt"

echo "=== Building Rust for iOS ==="
cd "$RUST_DIR"
cargo build --release --target aarch64-apple-ios
cargo build --release --target aarch64-apple-ios-sim
cd -

echo "=== Building Rust for Android ==="
cd "$RUST_DIR"
cargo ndk -t arm64-v8a -t x86_64 build --release
cd -

echo "=== Copying prebuilts ==="
mkdir -p "$PREBUILT_DIR/ios" "$PREBUILT_DIR/android/arm64-v8a" "$PREBUILT_DIR/android/x86_64"

cp "$RUST_DIR/target/aarch64-apple-ios/release/libwebgpu_camera.a" \
   "$PREBUILT_DIR/ios/libwebgpu_camera.a"
cp "$RUST_DIR/target/aarch64-apple-ios-sim/release/libwebgpu_camera.a" \
   "$PREBUILT_DIR/ios/libwebgpu_camera_sim.a"
cp "$RUST_DIR/target/aarch64-linux-android/release/libwebgpu_camera.so" \
   "$PREBUILT_DIR/android/arm64-v8a/libwebgpu_camera.so"
cp "$RUST_DIR/target/x86_64-linux-android/release/libwebgpu_camera.so" \
   "$PREBUILT_DIR/android/x86_64/libwebgpu_camera.so"

echo "=== Rust build complete ==="
```

- [ ] **Step 2: Create `scripts/generate-bindings.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Generate UniFFI TypeScript + native bindings
# Run from monorepo root

MODULE_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera"

cd "$MODULE_DIR"
npx ubrn build ios --and-generate
npx ubrn build android --and-generate
cd -

echo "=== Bindings generated ==="
```

- [ ] **Step 3: Create `scripts/build-skia-graphite.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Build Skia with Graphite backend
# Run from monorepo root

SK_GRAPHITE=1 bun run build-skia

echo "=== Skia Graphite build complete ==="
```

- [ ] **Step 4: Make scripts executable**

Run: `chmod +x scripts/build-rust.sh scripts/generate-bindings.sh scripts/build-skia-graphite.sh`

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: add build scripts for Rust, UniFFI bindings, and Skia Graphite"
```

---

## Chunk 2: Core Package & Expo Module Scaffold

### Task 3: Create core package scaffold

**Files:**
- Create: `packages/react-native-webgpu-camera/package.json`
- Create: `packages/react-native-webgpu-camera/tsconfig.json`
- Create: `packages/react-native-webgpu-camera/src/index.ts`

- [ ] **Step 1: Create directory structure**

Run: `mkdir -p packages/react-native-webgpu-camera/src`

- [ ] **Step 2: Create `packages/react-native-webgpu-camera/package.json`**

Check react-native-wgpu's exact peer dependency name first:

Run: `npm view react-native-wgpu peerDependencies --json`

Use the output to determine if the worklet package is `react-native-worklets` or `react-native-worklets-core`. Then create:

```json
{
  "name": "react-native-webgpu-camera",
  "version": "0.0.1",
  "private": true,
  "main": "src/index.ts",
  "types": "src/index.ts",
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint src/"
  },
  "peerDependencies": {
    "react": "*",
    "react-native": ">=0.83.0",
    "react-native-wgpu": "^0.5.8",
    "@shopify/react-native-skia": "*",
    "react-native-reanimated": ">=4.2.1",
    "expo": ">=55.0.0"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "react-native-wgpu": "^0.5.8",
    "@shopify/react-native-skia": "*",
    "react-native-reanimated": ">=4.2.1"
  }
}
```

**Note:** The `react-native-worklets` peer dep from react-native-wgpu may be satisfied transitively by react-native-reanimated. Verify after `bun install`.

- [ ] **Step 3: Create `packages/react-native-webgpu-camera/tsconfig.json`**

```json
{
  "extends": "../../tsconfig.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Create `packages/react-native-webgpu-camera/src/index.ts`**

```typescript
// react-native-webgpu-camera
// Phase 0 spike — minimal exports

export { WebGPUCameraModule } from '../modules/webgpu-camera';
```

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/
git commit -m "feat: scaffold core package with package.json and TypeScript config"
```

---

### Task 4: Create local Expo module scaffold

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/expo-module.config.json`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/index.ts`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/src/WebGPUCameraModule.ts`

- [ ] **Step 1: Create directory structure**

Run: `mkdir -p packages/react-native-webgpu-camera/modules/webgpu-camera/src`

- [ ] **Step 2: Create `expo-module.config.json`**

```json
{
  "platforms": ["ios", "android"],
  "ios": {
    "modules": ["WebGPUCameraModule"]
  },
  "android": {
    "modules": ["expo.modules.webgpucamera.WebGPUCameraModule"]
  }
}
```

- [ ] **Step 3: Create `src/WebGPUCameraModule.ts`**

This is the TypeScript interface that maps to the native module. The actual native implementation will be generated by UniFFI, but we define the JS-facing API here:

```typescript
import { NativeModule, requireNativeModule } from 'expo-modules-core';

// Type definitions matching the Rust API
export interface FrameDimensions {
  width: number;
  height: number;
  bytesPerRow: number;
}

interface WebGPUCameraModuleInterface extends NativeModule {
  startCameraPreview(deviceId: string, width: number, height: number): void;
  stopCameraPreview(): void;
  getCurrentFrameHandle(): number;
  getCurrentFramePixels(): Uint8Array;
  getFrameDimensions(): FrameDimensions;
  startTestRecorder(outputPath: string, width: number, height: number): number;
  stopTestRecorder(): string;
  getThermalState(): string;
}

export default requireNativeModule<WebGPUCameraModuleInterface>(
  'WebGPUCamera'
);
```

- [ ] **Step 4: Create `index.ts`**

```typescript
export { default as WebGPUCameraModule } from './src/WebGPUCameraModule';
export type { FrameDimensions } from './src/WebGPUCameraModule';
```

- [ ] **Step 5: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/
git commit -m "feat: scaffold local Expo module with TypeScript interface"
```

---

### Task 5: Create Expo module native stubs (iOS)

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/ios/WebGPUCameraModule.swift`

- [ ] **Step 1: Create iOS directory**

Run: `mkdir -p packages/react-native-webgpu-camera/modules/webgpu-camera/ios`

- [ ] **Step 2: Create `ios/WebGPUCameraModule.swift`**

This is a stub that will later call into the Rust library. For now it returns placeholder values so the module loads:

```swift
import ExpoModulesCore

public class WebGPUCameraModule: Module {
  public func definition() -> ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { (deviceId: String, width: Int, height: Int) in
      // TODO: Call Rust start_camera_preview via UniFFI bindings
      print("[WebGPUCamera] startCameraPreview(\(deviceId), \(width)x\(height))")
    }

    Function("stopCameraPreview") {
      // TODO: Call Rust stop_camera_preview
      print("[WebGPUCamera] stopCameraPreview")
    }

    Function("getCurrentFrameHandle") { () -> Int in
      // TODO: Call Rust get_current_frame_handle
      // Returns 0 until Rust is wired up
      return 0
    }

    Function("getCurrentFramePixels") { () -> Data in
      // TODO: Call Rust get_current_frame_pixels
      return Data()
    }

    Function("getFrameDimensions") { () -> [String: Any] in
      // TODO: Call Rust get_frame_dimensions
      return ["width": 0, "height": 0, "bytesPerRow": 0]
    }

    Function("startTestRecorder") { (outputPath: String, width: Int, height: Int) -> Int in
      // TODO: Call Rust start_test_recorder
      return 0
    }

    Function("stopTestRecorder") { () -> String in
      // TODO: Call Rust stop_test_recorder
      return ""
    }

    Function("getThermalState") { () -> String in
      let state = ProcessInfo.processInfo.thermalState
      switch state {
      case .nominal: return "nominal"
      case .fair: return "fair"
      case .serious: return "serious"
      case .critical: return "critical"
      @unknown default: return "nominal"
      }
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/ios/
git commit -m "feat: add iOS Expo module stub with thermal state implementation"
```

---

### Task 6: Create Expo module native stubs (Android)

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera/WebGPUCameraModule.kt`

- [ ] **Step 1: Create Android directory structure**

Run: `mkdir -p packages/react-native-webgpu-camera/modules/webgpu-camera/android/src/main/java/expo/modules/webgpucamera`

- [ ] **Step 2: Create `WebGPUCameraModule.kt`**

```kotlin
package expo.modules.webgpucamera

import android.os.PowerManager
import android.content.Context
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class WebGPUCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("WebGPUCamera")

    Function("startCameraPreview") { deviceId: String, width: Int, height: Int ->
      // TODO: Call Rust start_camera_preview via UniFFI bindings
      println("[WebGPUCamera] startCameraPreview($deviceId, ${width}x${height})")
    }

    Function("stopCameraPreview") {
      // TODO: Call Rust stop_camera_preview
      println("[WebGPUCamera] stopCameraPreview")
    }

    Function("getCurrentFrameHandle") {
      // TODO: Call Rust get_current_frame_handle
      0L
    }

    Function("getCurrentFramePixels") {
      // TODO: Call Rust get_current_frame_pixels
      ByteArray(0)
    }

    Function("getFrameDimensions") {
      // TODO: Call Rust get_frame_dimensions
      mapOf("width" to 0, "height" to 0, "bytesPerRow" to 0)
    }

    Function("startTestRecorder") { outputPath: String, width: Int, height: Int ->
      // TODO: Call Rust start_test_recorder
      0L
    }

    Function("stopTestRecorder") {
      // TODO: Call Rust stop_test_recorder
      ""
    }

    Function("getThermalState") {
      val context = appContext.reactContext ?: return@Function "nominal"
      val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
      when (pm?.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "nominal"
        PowerManager.THERMAL_STATUS_LIGHT -> "fair"
        PowerManager.THERMAL_STATUS_MODERATE -> "fair"
        PowerManager.THERMAL_STATUS_SEVERE -> "serious"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "critical"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "critical"
        else -> "nominal"
      }
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/android/
git commit -m "feat: add Android Expo module stub with thermal state implementation"
```

---

## Chunk 3: Rust Crate Setup

### Task 7: Initialize Rust crate with UniFFI

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/Cargo.toml`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/lib.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/ubrn.config.yaml`

- [ ] **Step 1: Verify Rust toolchain and uniffi version compatibility**

Run: `rustc --version`

Expected: >= 1.63.0 (required for `const Mutex::new()` in static context).
If older, run: `rustup update stable`

Run: `npm view uniffi-bindgen-react-native@0.30.0 --json 2>/dev/null | head -20`

Also check what uniffi crate version 0.30.x expects:

Run: `curl -s https://raw.githubusercontent.com/jhugman/uniffi-bindgen-react-native/main/Cargo.toml 2>/dev/null | grep uniffi || echo "Check manually at https://github.com/jhugman/uniffi-bindgen-react-native"`

Use the output to determine the correct `uniffi` crate version for Cargo.toml.

- [ ] **Step 2: Create directory structure**

Run: `mkdir -p packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/recorder`

- [ ] **Step 3: Create `Cargo.toml`**

```toml
[package]
name = "webgpu-camera"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["staticlib", "cdylib"]

[dependencies]
# Pin to version matching uniffi-bindgen-react-native 0.30.x
# Verify exact version from Step 1
uniffi = { version = "0.29", features = ["cli"] }

[build-dependencies]
uniffi = { version = "0.29", features = ["build"] }
```

**Note:** The `uniffi` version MUST match what `uniffi-bindgen-react-native` 0.30.x expects. Update the version from Step 1's findings.

- [ ] **Step 4: Create `ubrn.config.yaml`**

```yaml
---
rust:
  directory: ./
  manifestPath: Cargo.toml

android:
  targets:
    - arm64-v8a
    - x86_64
  apiLevel: 28

ios:
  targets:
    - aarch64-apple-ios
    - aarch64-apple-ios-sim
```

- [ ] **Step 5: Create `src/lib.rs`**

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

uniffi::setup_scaffolding!();

// Thread-safe slot for latest camera frame handle.
// Camera callback updates this; JS polls it on render tick.
static CURRENT_FRAME_HANDLE: AtomicU64 = AtomicU64::new(0);
static CURRENT_FRAME_PIXELS: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static FRAME_DIMS: Mutex<FrameDimensions> = Mutex::new(FrameDimensions {
    width: 0,
    height: 0,
    bytes_per_row: 0,
});

#[derive(uniffi::Record, Clone)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

#[uniffi::export]
pub fn start_camera_preview(device_id: String, width: u32, height: u32) {
    // Platform-specific implementation will be added per-platform
    // For now, log the call
    println!(
        "[webgpu-camera] start_camera_preview({}, {}x{})",
        device_id, width, height
    );

    // Update frame dimensions
    let mut dims = FRAME_DIMS.lock().unwrap();
    dims.width = width;
    dims.height = height;
    dims.bytes_per_row = width * 4; // BGRA8

    #[cfg(target_os = "ios")]
    camera::ios::start_preview(&device_id, width, height);

    #[cfg(target_os = "android")]
    camera::android::start_preview(&device_id, width, height);
}

#[uniffi::export]
pub fn stop_camera_preview() {
    #[cfg(target_os = "ios")]
    camera::ios::stop_preview();

    #[cfg(target_os = "android")]
    camera::android::stop_preview();
}

#[uniffi::export]
pub fn get_current_frame_handle() -> u64 {
    CURRENT_FRAME_HANDLE.load(Ordering::Relaxed)
}

#[uniffi::export]
pub fn get_current_frame_pixels() -> Vec<u8> {
    CURRENT_FRAME_PIXELS.lock().unwrap().clone()
}

#[uniffi::export]
pub fn get_frame_dimensions() -> FrameDimensions {
    FRAME_DIMS.lock().unwrap().clone()
}

#[uniffi::export]
pub fn start_test_recorder(output_path: String, width: u32, height: u32) -> u64 {
    #[cfg(target_os = "ios")]
    return recorder::ios::start_recorder(&output_path, width, height);

    #[cfg(target_os = "android")]
    return recorder::android::start_recorder(&output_path, width, height);

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = (output_path, width, height);
        0
    }
}

#[uniffi::export]
pub fn stop_test_recorder() -> String {
    #[cfg(target_os = "ios")]
    return recorder::ios::stop_recorder();

    #[cfg(target_os = "android")]
    return recorder::android::stop_recorder();

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    String::new()
}

#[uniffi::export]
pub fn get_thermal_state() -> String {
    #[cfg(target_os = "ios")]
    return thermal::ios_thermal_state();

    #[cfg(target_os = "android")]
    return thermal::android_thermal_state();

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    "nominal".to_string()
}

mod camera;
mod recorder;
mod thermal;
```

- [ ] **Step 6: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/rust/
git commit -m "feat: initialize Rust crate with UniFFI-annotated spike API"
```

---

### Task 8: Create Rust platform modules (stubs)

**Files:**
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/mod.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/ios.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/camera/android.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/recorder/mod.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/recorder/ios.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/recorder/android.rs`
- Create: `packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/thermal.rs`

- [ ] **Step 1: Create `src/camera/mod.rs`**

```rust
#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(target_os = "android")]
pub mod android;
```

- [ ] **Step 2: Create `src/camera/ios.rs`**

```rust
//! iOS camera implementation using AVCaptureSession.
//!
//! Pipeline:
//! AVCaptureSession -> AVCaptureVideoDataOutput
//!   -> CMSampleBufferGetImageBuffer() -> CVPixelBuffer
//!   -> CVPixelBufferGetIOSurface() -> IOSurface handle
//!   -> Store handle in CURRENT_FRAME_HANDLE atomic
//!   -> Store pixel copy in CURRENT_FRAME_PIXELS mutex

use crate::{CURRENT_FRAME_HANDLE, CURRENT_FRAME_PIXELS};
use std::sync::atomic::Ordering;

pub fn start_preview(device_id: &str, width: u32, height: u32) {
    // TODO: Implement AVCaptureSession setup
    // 1. Create AVCaptureSession
    // 2. Add AVCaptureVideoDataOutput
    // 3. Set delegate callback that:
    //    a. Gets CVPixelBuffer from CMSampleBuffer
    //    b. Gets IOSurface handle via CVPixelBufferGetIOSurface
    //    c. Stores handle in CURRENT_FRAME_HANDLE
    //    d. Optionally copies pixels for fallback path
    println!(
        "[webgpu-camera/ios] start_preview({}, {}x{}) — stub",
        device_id, width, height
    );
}

pub fn stop_preview() {
    CURRENT_FRAME_HANDLE.store(0, Ordering::Relaxed);
    CURRENT_FRAME_PIXELS.lock().unwrap().clear();
    println!("[webgpu-camera/ios] stop_preview — stub");
}
```

- [ ] **Step 3: Create `src/camera/android.rs`**

```rust
//! Android camera implementation using Camera2.
//!
//! Pipeline:
//! Camera2 -> ImageReader -> Image.getHardwareBuffer()
//!   -> HardwareBuffer handle
//!   -> Store handle in CURRENT_FRAME_HANDLE atomic
//!   -> Store pixel copy in CURRENT_FRAME_PIXELS mutex

use crate::{CURRENT_FRAME_HANDLE, CURRENT_FRAME_PIXELS};
use std::sync::atomic::Ordering;

pub fn start_preview(device_id: &str, width: u32, height: u32) {
    // TODO: Implement Camera2 setup via JNI
    // 1. Open CameraDevice
    // 2. Create ImageReader with HardwareBuffer usage
    // 3. Set OnImageAvailableListener that:
    //    a. Acquires latest image
    //    b. Gets HardwareBuffer handle
    //    c. Stores handle in CURRENT_FRAME_HANDLE
    //    d. Optionally copies pixels for fallback path
    println!(
        "[webgpu-camera/android] start_preview({}, {}x{}) — stub",
        device_id, width, height
    );
}

pub fn stop_preview() {
    CURRENT_FRAME_HANDLE.store(0, Ordering::Relaxed);
    CURRENT_FRAME_PIXELS.lock().unwrap().clear();
    println!("[webgpu-camera/android] stop_preview — stub");
}
```

- [ ] **Step 4: Create `src/recorder/mod.rs`**

```rust
#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(target_os = "android")]
pub mod android;
```

- [ ] **Step 5: Create `src/recorder/ios.rs`**

```rust
//! iOS video recording using AVAssetWriter.
//!
//! Returns IOSurface handle from AVAssetWriterInputPixelBufferAdaptor
//! that Dawn can render to directly.

pub fn start_recorder(output_path: &str, width: u32, height: u32) -> u64 {
    // TODO: Implement AVAssetWriter setup
    // 1. Create AVAssetWriter with .mp4 output
    // 2. Create AVAssetWriterInput for video
    // 3. Create AVAssetWriterInputPixelBufferAdaptor
    // 4. Return IOSurface handle from the adaptor's pixel buffer pool
    println!(
        "[webgpu-camera/ios] start_recorder({}, {}x{}) — stub",
        output_path, width, height
    );
    0
}

pub fn stop_recorder() -> String {
    // TODO: Finalize AVAssetWriter, return output path
    println!("[webgpu-camera/ios] stop_recorder — stub");
    String::new()
}
```

- [ ] **Step 6: Create `src/recorder/android.rs`**

```rust
//! Android video recording using MediaRecorder.
//!
//! Returns Surface handle from MediaRecorder.getSurface()
//! that Dawn can render to directly.

pub fn start_recorder(output_path: &str, width: u32, height: u32) -> u64 {
    // TODO: Implement MediaRecorder setup via JNI
    // 1. Create MediaRecorder
    // 2. Configure video source as SURFACE
    // 3. Set output format, encoder, size, frame rate
    // 4. Prepare and get Surface
    // 5. Return Surface handle (ANativeWindow pointer)
    println!(
        "[webgpu-camera/android] start_recorder({}, {}x{}) — stub",
        output_path, width, height
    );
    0
}

pub fn stop_recorder() -> String {
    // TODO: Stop MediaRecorder, return output path
    println!("[webgpu-camera/android] stop_recorder — stub");
    String::new()
}
```

- [ ] **Step 7: Create `src/thermal.rs`**

Thermal state is implemented directly in Swift/Kotlin (Expo module stubs in
Tasks 5 and 6) because it uses platform APIs that are simpler to call from
Swift/Kotlin than through Rust FFI. The Rust `get_thermal_state()` export
delegates to a placeholder — the Expo module's native `getThermalState()`
is the canonical implementation that JS calls.

```rust
//! Thermal state placeholder.
//! The real implementation lives in the Expo module's native code
//! (Swift/Kotlin) which has direct access to ProcessInfo / PowerManager.
//! This Rust stub exists only so the UniFFI API surface compiles.

pub fn ios_thermal_state() -> String {
    "nominal".to_string()
}

pub fn android_thermal_state() -> String {
    "nominal".to_string()
}
```

- [ ] **Step 8: Verify Rust crate compiles (host target only)**

Run: `cd packages/react-native-webgpu-camera/modules/webgpu-camera/rust && cargo check`

Expected: Compiles with warnings about unused code (platform-specific modules are gated by cfg).

- [ ] **Step 9: Commit**

```bash
git add packages/react-native-webgpu-camera/modules/webgpu-camera/rust/src/
git commit -m "feat: add Rust platform module stubs for camera, recorder, and thermal"
```

---

## Chunk 4: Example App

### Task 9: Create Expo 55 example app

**Files:**
- Create: `apps/example/package.json`
- Create: `apps/example/app.json`
- Create: `apps/example/eas.json`
- Create: `apps/example/tsconfig.json`

- [ ] **Step 1: Scaffold Expo app**

Run from monorepo root:

```bash
cd apps
npx create-expo-app@latest example --template default@sdk-55
cd ..
```

This generates the standard Expo 55 scaffold. We'll customize it next.

**Important:** After `create-expo-app`, delete the generated `node_modules` and `package-lock.json` inside `apps/example/` before running `bun install` from root. Otherwise the workspace resolution may fail.

Run: `rm -rf apps/example/node_modules apps/example/package-lock.json`

- [ ] **Step 2: Update `apps/example/package.json` — add workspace dependency**

Add to the `dependencies` section:

```json
{
  "react-native-webgpu-camera": "workspace:*",
  "react-native-wgpu": "^0.5.8",
  "@shopify/react-native-skia": "*",
  "react-native-reanimated": ">=4.2.1",
  "expo-camera": "*",
  "expo-file-system": "*"
}
```

Run: `npm view react-native-wgpu peerDependencies --json`

If `react-native-worklets` is a peer dep, add it too.

- [ ] **Step 3: Create `apps/example/eas.json`**

```json
{
  "cli": { "version": ">= 12.0.0" },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": { "buildConfiguration": "Debug" },
      "android": { "buildType": "apk" },
      "env": { "SK_GRAPHITE": "1" }
    }
  }
}
```

- [ ] **Step 4: Install dependencies**

Run from monorepo root: `bun install`

Verify workspace linking:

Run: `ls -la node_modules/react-native-webgpu-camera`

Expected: Symlink to `packages/react-native-webgpu-camera`.

- [ ] **Step 5: Commit**

```bash
git add apps/example/
git commit -m "feat: scaffold Expo 55 example app with workspace dependencies"
```

---

### Task 10: Create Sobel edge detection compute shader

**Files:**
- Create: `apps/example/src/shaders/sobel.wgsl.ts`

- [ ] **Step 1: Create directory**

Run: `mkdir -p apps/example/src/shaders`

- [ ] **Step 2: Create `apps/example/src/shaders/sobel.wgsl.ts`**

This is a non-trivial compute shader for Spike 2 validation — enough GPU work to be measurable:

```typescript
// Sobel edge detection compute shader.
// Reads from an input texture, writes edge-detected output to a storage texture.
// Used to validate Spike 2: compute dispatch from worklet thread.

export const SOBEL_WGSL = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(inputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);

  // Sobel kernels
  // Gx:          Gy:
  // -1  0  1     -1 -2 -1
  // -2  0  2      0  0  0
  // -1  0  1      1  2  1

  var gx = vec3f(0.0);
  var gy = vec3f(0.0);

  // Sample 3x3 neighborhood
  let tl = textureLoad(inputTex, coord + vec2i(-1, -1), 0).rgb;
  let tc = textureLoad(inputTex, coord + vec2i( 0, -1), 0).rgb;
  let tr = textureLoad(inputTex, coord + vec2i( 1, -1), 0).rgb;
  let ml = textureLoad(inputTex, coord + vec2i(-1,  0), 0).rgb;
  let mr = textureLoad(inputTex, coord + vec2i( 1,  0), 0).rgb;
  let bl = textureLoad(inputTex, coord + vec2i(-1,  1), 0).rgb;
  let bc = textureLoad(inputTex, coord + vec2i( 0,  1), 0).rgb;
  let br = textureLoad(inputTex, coord + vec2i( 1,  1), 0).rgb;

  gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  let edge = sqrt(gx * gx + gy * gy);
  let intensity = clamp(length(edge) / 3.0, 0.0, 1.0);

  textureStore(outputTex, coord, vec4f(intensity, intensity, intensity, 1.0));
}
`;
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/shaders/
git commit -m "feat: add Sobel edge detection WGSL compute shader for spike 2"
```

---

### Task 11: Create spike metrics hook

**Files:**
- Create: `apps/example/src/hooks/useSpikeMetrics.ts`

- [ ] **Step 1: Create directory**

Run: `mkdir -p apps/example/src/hooks`

- [ ] **Step 2: Create `apps/example/src/hooks/useSpikeMetrics.ts`**

```typescript
import { useRef, useCallback } from 'react';
import { WebGPUCameraModule } from 'react-native-webgpu-camera';

export interface SpikeResults {
  spike1Path: 'zero-copy' | 'copy-fallback' | 'unknown';
  spike1AvgMs: number;
  spike2Path: 'worklet-compute' | 'main-thread-compute' | 'unknown';
  spike2AvgMs: number;
  spike3Path: 'graphite' | 'ganesh-fallback' | 'unknown';
  spike3OverheadMs: number;
  spike4Path: 'surface-record' | 'readback-record' | 'unknown';
  sustainedFps: number;
  frameDrops: number;
  totalFrames: number;
  peakMemoryMb: number;
  thermalTransitions: string[];
}

interface FrameTiming {
  importMs: number;
  computeMs: number;
  skiaMs: number;
  totalMs: number;
}

export function useSpikeMetrics(durationSeconds = 60) {
  const startTime = useRef(0);
  const frameTimings = useRef<FrameTiming[]>([]);
  const lastFrameTime = useRef(0);
  const frameDropCount = useRef(0);
  const thermalTransitions = useRef<string[]>([]);
  const lastThermalState = useRef('nominal');

  const recordFrame = useCallback((timing: FrameTiming) => {
    const now = performance.now();

    if (startTime.current === 0) {
      startTime.current = now;
    }

    // Detect frame drops (>20ms gap = missed a frame at 60fps)
    if (lastFrameTime.current > 0) {
      const gap = now - lastFrameTime.current;
      if (gap > 20) {
        frameDropCount.current += Math.floor(gap / 16.67) - 1;
      }
    }
    lastFrameTime.current = now;

    frameTimings.current.push(timing);

    // Check thermal state every 60 frames (~1 second)
    if (frameTimings.current.length % 60 === 0) {
      const thermal = WebGPUCameraModule.getThermalState();
      if (thermal !== lastThermalState.current) {
        const elapsed = ((now - startTime.current) / 1000).toFixed(0);
        thermalTransitions.current.push(
          `${lastThermalState.current} -> ${thermal} at ${elapsed}s`
        );
        lastThermalState.current = thermal;
      }
    }
  }, []);

  const getSummary = useCallback((): SpikeResults | null => {
    const timings = frameTimings.current;
    if (timings.length === 0) return null;

    const elapsed = (performance.now() - startTime.current) / 1000;
    const avgImport =
      timings.reduce((s, t) => s + t.importMs, 0) / timings.length;
    const avgCompute =
      timings.reduce((s, t) => s + t.computeMs, 0) / timings.length;
    const avgSkia =
      timings.reduce((s, t) => s + t.skiaMs, 0) / timings.length;

    return {
      spike1Path: 'unknown', // Set by caller based on which path succeeded
      spike1AvgMs: avgImport,
      spike2Path: 'unknown',
      spike2AvgMs: avgCompute,
      spike3Path: 'unknown',
      spike3OverheadMs: avgSkia,
      spike4Path: 'unknown',
      sustainedFps: timings.length / elapsed,
      frameDrops: frameDropCount.current,
      totalFrames: timings.length,
      peakMemoryMb: 0, // Platform-specific, filled by caller
      thermalTransitions: thermalTransitions.current,
    };
  }, []);

  const logSummary = useCallback(
    (paths: Partial<SpikeResults>) => {
      const summary = getSummary();
      if (!summary) return;

      const merged = { ...summary, ...paths };

      console.log('=== SPIKE RESULTS ===');
      console.log(
        `Spike 1: ${merged.spike1Path} (${merged.spike1AvgMs.toFixed(2)}ms avg)`
      );
      console.log(
        `Spike 2: ${merged.spike2Path} (${merged.spike2AvgMs.toFixed(2)}ms avg)`
      );
      console.log(
        `Spike 3: ${merged.spike3Path} (${merged.spike3OverheadMs.toFixed(2)}ms overhead)`
      );
      console.log(`Spike 4: ${merged.spike4Path}`);
      console.log(`Sustained FPS: ${merged.sustainedFps.toFixed(1)}`);
      console.log(
        `Frame drops: ${merged.frameDrops}/${merged.totalFrames} (${((merged.frameDrops / merged.totalFrames) * 100).toFixed(1)}%)`
      );
      console.log(`Memory: ${merged.peakMemoryMb.toFixed(0)}MB peak`);
      console.log(
        `Thermal: ${merged.thermalTransitions.length === 0 ? 'nominal (no transitions)' : merged.thermalTransitions.join(', ')}`
      );
      console.log('========================');
    },
    [getSummary]
  );

  return { recordFrame, getSummary, logSummary };
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/example/src/hooks/
git commit -m "feat: add useSpikeMetrics hook for 60-second performance tracking"
```

---

### Task 12: Create camera spike screen

**Files:**
- Create: `apps/example/app/_layout.tsx`
- Create: `apps/example/app/(tabs)/_layout.tsx`
- Create: `apps/example/app/(tabs)/index.tsx`

- [ ] **Step 1: Create `app/_layout.tsx`**

```typescript
import { Stack } from 'expo-router';

export default function RootLayout() {
  return <Stack screenOptions={{ headerShown: false }} />;
}
```

- [ ] **Step 2: Create `app/(tabs)/_layout.tsx`**

```typescript
import { Tabs } from 'expo-router';

export default function TabLayout() {
  return (
    <Tabs screenOptions={{ headerShown: false }}>
      <Tabs.Screen name="index" options={{ title: 'Camera Spike' }} />
    </Tabs>
  );
}
```

- [ ] **Step 3: Create `app/(tabs)/index.tsx`**

This is the main spike validation screen. It wires together:
- Rust camera module (frame delivery)
- react-native-wgpu Canvas (WebGPU rendering)
- Reanimated worklet (off-thread compute)
- Skia overlay (debug metrics)
- Recorder Surface (video capture test)

```typescript
import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
} from 'react-native';
import { Canvas } from 'react-native-wgpu';
import { runOnUI } from 'react-native-reanimated';
import { WebGPUCameraModule } from 'react-native-webgpu-camera';
import { useSpikeMetrics, SpikeResults } from '../../src/hooks/useSpikeMetrics';
import { SOBEL_WGSL } from '../../src/shaders/sobel.wgsl';
import * as FileSystem from 'expo-file-system';

// Spike path tracking
type Spike1Path = 'zero-copy' | 'copy-fallback';
type Spike2Path = 'worklet-compute' | 'main-thread-compute';
type Spike3Path = 'graphite' | 'ganesh-fallback';
type Spike4Path = 'surface-record' | 'readback-record';

export default function CameraSpikeScreen() {
  const [isRunning, setIsRunning] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [spikeStatus, setSpikeStatus] = useState({
    spike1: 'pending' as string,
    spike2: 'pending' as string,
    spike3: 'pending' as string,
    spike4: 'pending' as string,
    fps: 0,
    elapsed: 0,
  });
  const [results, setResults] = useState<SpikeResults | null>(null);
  const metrics = useSpikeMetrics(60);
  const canvasRef = useRef<any>(null);

  // Start the camera + render pipeline
  const startPipeline = useCallback(async () => {
    setIsRunning(true);
    setResults(null);

    // Start camera via Rust module
    WebGPUCameraModule.startCameraPreview('back', 1920, 1080);

    // The render loop runs on a worklet thread via react-native-wgpu's
    // Canvas onCreateSurface callback. WebGPU objects are passed to
    // the worklet where compute + render happens each frame.
    //
    // Detailed wiring:
    // 1. Canvas.onCreateSurface provides GPUDevice + GPUCanvasContext
    // 2. We create compute pipeline (Sobel shader) once
    // 3. Each frame tick:
    //    a. Poll Rust for frame handle (Spike 1)
    //    b. Import as GPUTexture or fall back to writeTexture (Spike 1)
    //    c. Dispatch Sobel compute shader (Spike 2)
    //    d. Draw Skia overlay (Spike 3)
    //    e. Render pass to canvas + recorder if active (Spike 4)
    //    f. Record timing metrics (Spike 5)
    //
    // After 60 seconds, log summary and stop.
  }, []);

  const stopPipeline = useCallback(() => {
    WebGPUCameraModule.stopCameraPreview();
    setIsRunning(false);

    // Log final results
    metrics.logSummary({
      spike1Path: spikeStatus.spike1 as any,
      spike2Path: spikeStatus.spike2 as any,
      spike3Path: spikeStatus.spike3 as any,
      spike4Path: spikeStatus.spike4 as any,
    });
  }, [metrics, spikeStatus]);

  // Record 5 seconds for Spike 4
  const startRecording = useCallback(async () => {
    const outputPath = `${FileSystem.documentDirectory}spike4_test.mp4`;
    setIsRecording(true);

    const surfaceHandle = WebGPUCameraModule.startTestRecorder(
      outputPath,
      1920,
      1080
    );

    if (surfaceHandle !== 0) {
      setSpikeStatus((s) => ({ ...s, spike4: 'surface-record' }));
    } else {
      setSpikeStatus((s) => ({ ...s, spike4: 'readback-record' }));
    }

    // Stop after 5 seconds
    setTimeout(async () => {
      const filePath = WebGPUCameraModule.stopTestRecorder();
      setIsRecording(false);

      if (filePath) {
        console.log(`SPIKE4: Video saved to ${filePath}`);
        // Verify file exists and has non-zero size
        const info = await FileSystem.getInfoAsync(filePath);
        console.log(`SPIKE4: File size: ${info.exists ? (info as any).size : 0} bytes`);
      }
    }, 5000);
  }, []);

  return (
    <View style={styles.container}>
      {/* WebGPU Canvas — full screen camera preview + compute output */}
      <Canvas
        ref={canvasRef}
        style={StyleSheet.absoluteFill}
        // onCreateSurface callback wires up the render loop
        // This is where the GPUDevice is obtained and the
        // worklet render loop is started
      />

      {/* Status overlay */}
      <View style={styles.overlay}>
        <Text style={styles.statusText}>
          Spike 1 (zero-copy): {spikeStatus.spike1}
        </Text>
        <Text style={styles.statusText}>
          Spike 2 (worklet compute): {spikeStatus.spike2}
        </Text>
        <Text style={styles.statusText}>
          Spike 3 (Skia Graphite): {spikeStatus.spike3}
        </Text>
        <Text style={styles.statusText}>
          Spike 4 (recorder): {spikeStatus.spike4}
        </Text>
        <Text style={styles.statusText}>
          FPS: {spikeStatus.fps.toFixed(1)} | Elapsed: {spikeStatus.elapsed}s
        </Text>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable
          style={[styles.button, isRunning && styles.buttonActive]}
          onPress={isRunning ? stopPipeline : startPipeline}
        >
          <Text style={styles.buttonText}>
            {isRunning ? 'Stop' : 'Start Pipeline'}
          </Text>
        </Pressable>

        {isRunning && (
          <Pressable
            style={[styles.button, isRecording && styles.buttonActive]}
            onPress={startRecording}
            disabled={isRecording}
          >
            <Text style={styles.buttonText}>
              {isRecording ? 'Recording...' : 'Record 5s'}
            </Text>
          </Pressable>
        )}
      </View>

      {/* Results display */}
      {results && (
        <View style={styles.results}>
          <Text style={styles.resultsTitle}>Spike Results</Text>
          <Text style={styles.resultsText}>
            {JSON.stringify(results, null, 2)}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  overlay: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 8,
    padding: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    marginBottom: 4,
  },
  controls: {
    position: 'absolute',
    bottom: 60,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  button: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 24,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonActive: {
    backgroundColor: 'rgba(255,80,80,0.4)',
    borderColor: 'rgba(255,80,80,0.6)',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  results: {
    position: 'absolute',
    top: 220,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 8,
    padding: 12,
    maxHeight: 300,
  },
  resultsTitle: {
    color: '#0f0',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultsText: {
    color: '#0f0',
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
});
```

**Note:** The `Canvas` `onCreateSurface` callback wiring is intentionally left as a comment/TODO. The exact API depends on react-native-wgpu's current surface creation pattern, which the implementer must verify from react-native-wgpu's documentation or `with-webgpu` template. The structure above provides the full app shell; the WebGPU render loop wiring is the spike implementation work.

- [ ] **Step 4: Commit**

```bash
git add apps/example/app/
git commit -m "feat: add camera spike screen with full pipeline shell and controls"
```

---

## Chunk 5: Final Wiring & Verification

### Task 13: Verify monorepo resolves and TypeScript compiles

- [ ] **Step 1: Install all dependencies**

Run from monorepo root: `bun install`

Expected: Resolves all workspace dependencies. No version conflicts.

If `bun install` fails on workspace resolution, check that:
- `packages/react-native-webgpu-camera/package.json` has correct peer deps
- `apps/example/package.json` has `"react-native-webgpu-camera": "workspace:*"`
- Root `package.json` workspaces array includes `"packages/*"` and `"apps/*"`

- [ ] **Step 2: Run TypeScript check**

Run: `bun run typecheck`

Expected: May have errors from missing type declarations for native modules. Fix any import path issues. react-native-wgpu and Skia types may not be available until dependencies are installed.

- [ ] **Step 3: Verify Expo module is autolinked**

Run: `cd apps/example && npx expo config --type prebuild 2>&1 | grep -i webgpu`

Expected: The WebGPUCamera module should appear in the autolinking output.

If not found, verify `expo-module.config.json` exists and has correct module names. The `modules/` directory inside the core package must be discoverable by Expo autolinking.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve monorepo dependency and TypeScript issues"
```

---

### Task 14: Create prebuilt directory structure

**Files:**
- Create: `packages/react-native-webgpu-camera/prebuilt/.gitkeep`

- [ ] **Step 1: Create prebuilt directories**

```bash
mkdir -p packages/react-native-webgpu-camera/prebuilt/ios
mkdir -p packages/react-native-webgpu-camera/prebuilt/android/arm64-v8a
mkdir -p packages/react-native-webgpu-camera/prebuilt/android/x86_64
touch packages/react-native-webgpu-camera/prebuilt/ios/.gitkeep
touch packages/react-native-webgpu-camera/prebuilt/android/arm64-v8a/.gitkeep
touch packages/react-native-webgpu-camera/prebuilt/android/x86_64/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add packages/react-native-webgpu-camera/prebuilt/
git commit -m "feat: add prebuilt directory structure for Rust libraries"
```

---

### Task 15: Final verification and summary commit

- [ ] **Step 1: Verify complete directory structure**

Run: `find . -not -path './node_modules/*' -not -path './.git/*' -not -path '*/target/*' -type f | sort`

Verify the output matches the file map in the spec.

- [ ] **Step 2: Verify Rust compiles**

Run: `cd packages/react-native-webgpu-camera/modules/webgpu-camera/rust && cargo check && cd -`

- [ ] **Step 3: Run bun install and typecheck**

Run: `bun install && bun run typecheck`

- [ ] **Step 4: Final commit with complete scaffold**

Only if there are uncommitted changes from verification fixes:

```bash
git add -A
git commit -m "chore: final scaffold verification and cleanup"
```

---

## Next Steps (for implementer)

After this scaffold is complete, the implementer needs to do the following
on a physical device to complete the spike:

1. **Wire up Rust camera implementation** — Replace stubs in `camera/ios.rs`
   and `camera/android.rs` with actual AVCaptureSession / Camera2 code.

2. **Wire up WebGPU render loop** — In `index.tsx`, implement the
   `Canvas.onCreateSurface` callback that creates the compute pipeline,
   dispatches Sobel shader, and renders to the canvas.

3. **Test zero-copy import (Spike 1)** — Try importing the frame handle from
   Rust as a GPUTexture. If it fails, use the copy fallback.

4. **Test Skia overlay (Spike 3)** — Draw debug text via Skia on the
   processed frame. Detect Graphite vs Ganesh at runtime.

5. **Wire up recorder (Spike 4)** — Replace stubs in `recorder/ios.rs` and
   `recorder/android.rs`. Test the "Record 5s" button.

6. **Build and run on device** — `./scripts/build-rust.sh`, commit prebuilts,
   `eas build`, install dev client, run for 60 seconds, capture console output.
