#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LLVM_PROJECT_DIR="$REPO_ROOT/third_party/llvm-project"
LLVM_SRC_DIR="$LLVM_PROJECT_DIR/llvm"
LLVM_BUILD_DIR="$LLVM_PROJECT_DIR/build"
NPROC="${NPROC:-$(nproc)}"

echo "[setup] Repository root: $REPO_ROOT"
echo "[setup] LLVM project: $LLVM_PROJECT_DIR"
echo "[setup] LLVM build dir: $LLVM_BUILD_DIR"

if ! command -v ninja >/dev/null 2>&1; then
  echo "WARNING: 'ninja' not found in PATH. Install ninja for best performance."
  echo "On Debian/Ubuntu: sudo apt install ninja-build"
fi

mkdir -p "$LLVM_BUILD_DIR"

echo "[setup] Configuring LLVM/MLIR with CMake (Ninja generator)"
cmake -S "$LLVM_SRC_DIR" -B "$LLVM_BUILD_DIR" -G "Ninja" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_BUILD_RUNTIME=OFF \
    -DLLVM_BUILD_TOOLS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

echo "[setup] Building MLIR targets (parallel: $NPROC)"
cmake --build "$LLVM_BUILD_DIR" --parallel "$NPROC" --target \
    LLVMCore MLIRSupport MLIRIR MLIRPass MLIRTransformUtils MLIRAnalysis MLIRDialect mlir-tblgen mlir-opt mlir-translate || \
    cmake --build "$LLVM_BUILD_DIR" --parallel "$NPROC"

MLIR_CONFIG="$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
if [ ! -f "$MLIR_CONFIG" ]; then
    echo "ERROR: MLIR configuration not found at $MLIR_CONFIG"
    echo "Check the configure/build output above for errors."
    exit 1
fi

echo "[setup] LLVM/MLIR build completed. MLIRConfig.cmake: $MLIR_CONFIG"
echo "You can now run './build.sh' to build the project against this MLIR build."
