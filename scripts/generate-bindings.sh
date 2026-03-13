#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="packages/react-native-webgpu-camera/modules/webgpu-camera"

cd "$MODULE_DIR"
npx ubrn build ios --and-generate
npx ubrn build android --and-generate
cd -

echo "=== Bindings generated ==="
