#!/usr/bin/env bash
set -euo pipefail

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
