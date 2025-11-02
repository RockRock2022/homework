#!/usr/bin/env bash
set -euo pipefail

# Project build script. Assumes MLIR/LLVM has already been configured and built
# by ./setup.sh (which creates llvm-project/build).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LLVM_PROJECT_DIR="$REPO_ROOT/third_party/llvm-project"
LLVM_BUILD_DIR="$LLVM_PROJECT_DIR/build"
PROJECT_BUILD_DIR="$REPO_ROOT/build"
NPROC="${NPROC:-$(nproc)}"

echo "[build] Repository root: $REPO_ROOT"
echo "[build] LLVM project dir: $LLVM_PROJECT_DIR"
echo "[build] LLVM build dir: $LLVM_BUILD_DIR"
echo "[build] Project build dir: $PROJECT_BUILD_DIR"

MLIR_CONFIG="$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
if [ ! -f "$MLIR_CONFIG" ]; then
    echo "ERROR: MLIR not found. Expected MLIRConfig.cmake at: $MLIR_CONFIG"
    echo "Run './setup.sh' first to configure and build LLVM/MLIR."
    exit 1
fi

echo "[build] Configuring project (Ninja generator)"
mkdir -p "$PROJECT_BUILD_DIR"
cmake -S "$REPO_ROOT" -B "$PROJECT_BUILD_DIR" -G "Ninja" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"

echo "[build] Building project (parallel: $NPROC)"
cmake --build "$PROJECT_BUILD_DIR" --parallel "$NPROC"

echo "[build] Verifying build"
BUILT_LIB="$PROJECT_BUILD_DIR/src/libMyMLIRTosaTransforms.so"
BUILT_BIN="$REPO_ROOT/bin/my-opt"
if [ -f "$BUILT_LIB" ]; then
    echo "Build successful: $BUILT_LIB"
    ls -lh "$BUILT_LIB"
    if [ -x "$BUILT_BIN" ]; then
        echo "my-opt available: $BUILT_BIN"
        ls -lh "$BUILT_BIN"
    else
        echo "WARN: my-opt not found in bin/. It may still be in the build tree."
        find "$PROJECT_BUILD_DIR" -maxdepth 3 -type f -name my-opt -perm -111 || true
    fi
    echo "Try: ./bin/my-opt --adjust-layout src/adjust_layout.mlir"
else
    echo "Build failed: $BUILT_LIB not found"
    echo "Available libs:" 
    find "$PROJECT_BUILD_DIR" -name "*.so" -o -name "*.a"
    exit 1
fi
