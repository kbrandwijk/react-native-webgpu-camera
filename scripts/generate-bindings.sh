#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera/rust"
GEN_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera/generated"

mkdir -p "$GEN_DIR/swift" "$GEN_DIR/kotlin"

echo "=== Generating Swift bindings ==="
cd "$RUST_DIR"
cargo run --bin uniffi-bindgen -- generate \
  --library target/aarch64-apple-ios/release/libwebgpu_camera.a \
  --language swift \
  --out-dir "../generated/swift"

echo "=== Generating Kotlin bindings ==="
cargo run --bin uniffi-bindgen -- generate \
  --library target/aarch64-linux-android/release/libwebgpu_camera.so \
  --language kotlin \
  --out-dir "../generated/kotlin"
cd -

echo "=== Copying artifacts into native modules ==="
MODULE_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera"
PREBUILT_DIR="packages/react-native-webgpu-camera/prebuilt"

# iOS: copy generated Swift + FFI header/modulemap + static lib into ios/rust/
mkdir -p "$MODULE_DIR/ios/rust"
cp "$GEN_DIR/swift/webgpu_camera.swift" "$MODULE_DIR/ios/rust/"
cp "$GEN_DIR/swift/webgpu_cameraFFI.h" "$MODULE_DIR/ios/rust/"
cp "$GEN_DIR/swift/webgpu_cameraFFI.modulemap" "$MODULE_DIR/ios/rust/"
cp "$PREBUILT_DIR/ios/libwebgpu_camera.a" "$MODULE_DIR/ios/rust/"

# TODO: Android — copy generated Kotlin + .so into android module when wired

echo "=== Bindings generated and copied ==="
