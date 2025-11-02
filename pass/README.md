# MLIR TOSA Layout Adjustment Pass

This project implements a custom MLIR transformation pass that adjusts tensor layouts for TOSA (Tensor Operator Set Architecture) operations, targeting Conv2D operations. The pass converts between NCHW and NHWC data formats to optimize hardware performance and memory locality.

## Overview

The `AdjustLayoutPass` is a custom MLIR pass that:
- Targets TOSA Conv2D operations with specific weight shapes ([320, 320, 1, 1])
- Inserts transpose operations to convert data layouts:
  - Input tensors: NCHW → NHWC
  - Weight tensors: [OC, IC, H, W] → [OC, H, W, IC]
  - Output tensors: NHWC → NCHW
- Uses pattern matching and rewriting for efficient transformation

## Project Structure

```
pass/
├── CMakeLists.txt              # Top-level CMake configuration
├── setup.sh                    # LLVM/MLIR setup and build script
├── build.sh                    # Project build script
├── bin/                        # Output directory for executables
│   └── my-opt                  # Custom MLIR optimizer tool
├── build/                      # CMake build directory
├── src/                        # Source files
│   ├── CMakeLists.txt          # Source-level CMake configuration
│   ├── adjust_layout.td        # TableGen pass definition
│   ├── adjust_layout.hpp       # Pass header file
│   ├── adjust_layout.cpp       # Pass implementation
│   ├── adjust_layout.mlir      # Test case with expected transformations
│   └── my-opt.cpp              # MLIR optimizer tool entry point
└── third_party/
    └── llvm-project/           # LLVM/MLIR dependency
```

## Prerequisites

- **CMake** 3.20 or later
- **Ninja** build system (recommended)
- **C++17** compatible compiler (GCC, Clang)
- **Python** 3.6+ (for LLVM build)
- Sufficient disk space (~10GB for LLVM/MLIR build)

Install dependencies on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake ninja-build build-essential python3
```

## Build Instructions

### Step 1: Setup LLVM/MLIR

First, configure and build LLVM/MLIR with the MLIR project enabled:

```bash
./setup.sh
```

This script will:
- Configure LLVM/MLIR using CMake with Ninja generator
- Build only the necessary MLIR targets to save time
- Create the MLIR configuration files needed for the project
- Build time: ~30-60 minutes depending on your system

### Step 2: Build the Project

Once LLVM/MLIR is built, compile the custom pass and optimizer:

```bash
./build.sh
```

This script will:
- Configure the project using the pre-built MLIR libraries
- Generate TableGen files from `adjust_layout.td`
- Compile the pass library (`libMyMLIRTosaTransforms.so`)
- Build the `my-opt` tool in the `bin/` directory

## Usage

### Running the Optimizer

Use the `my-opt` tool to apply the layout adjustment pass to MLIR files:

```bash
./bin/my-opt --adjust-layout src/adjust_layout.mlir
```

## Development

### Key Components

- **adjust_layout.td**: TableGen definition of the pass interface
- **adjust_layout.cpp**: Pattern rewriter implementation
- **my-opt.cpp**: Tool entry point with pass registration
- **CMakeLists.txt**: Build configuration with MLIR dependencies

### Pass Implementation

The pass uses MLIR's pattern rewriting infrastructure:
- `OpRewritePattern<tosa::Conv2DOp>`: Matches Conv2D operations
- Pattern matching checks for specific weight shapes ([320, 320, 1, 1])
- Rewrites operations by inserting transpose ops around Conv2D
- Uses `GreedyPatternRewriteDriver` for efficient application

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [TOSA Specification](https://www.mlplatform.org/tosa/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
