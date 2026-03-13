#!/usr/bin/env bash
set -euo pipefail

SK_GRAPHITE=1 bun run build-skia

echo "=== Skia Graphite build complete ==="
