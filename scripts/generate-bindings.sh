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

echo "=== Bindings generated ==="
