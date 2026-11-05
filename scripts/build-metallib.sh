#!/bin/bash
# Compiles the NSS kernels into a metallib: scripts/build-metallib.sh OUT.metallib
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/Sources/NSSMetal/Shaders/nss.metal"
OUT=${1:?usage: build-metallib.sh OUT.metallib}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# For macOS 14 and later, like the package (by default the compiler targets this Mac's macOS, which older ones refuse).
xcrun metal -std=metal3.0 -O3 -mmacosx-version-min=14.0 -c "$SRC" -o "$TMP/nss.air"
xcrun metallib "$TMP/nss.air" -o "$OUT"
