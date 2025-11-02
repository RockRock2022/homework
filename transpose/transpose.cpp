#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

extern "C" void
transpose_nchwxy_to_nychwx(const uint8_t *inData, uint8_t *outData,
                           const int32_t inDims[6], const int32_t inStrides[6],
                           const int32_t outStrides[6], int32_t elementSize) {
  for (int32_t n = 0; n < inDims[0]; n++) {
    for (int32_t c = 0; c < inDims[1]; c++) {
      for (int32_t h = 0; h < inDims[2]; h++) {
        for (int32_t w = 0; w < inDims[3]; w++) {
          for (int32_t x = 0; x < inDims[4]; x++) {
            for (int32_t y = 0; y < inDims[5]; y++) {
              int32_t in_off = n * inStrides[0] + c * inStrides[1] +
                               h * inStrides[2] + w * inStrides[3] +
                               x * inStrides[4] + y * inStrides[5];
              int32_t out_off = n * outStrides[0] + y * outStrides[1] +
                                c * outStrides[2] + h * outStrides[3] +
                                w * outStrides[4] + x * outStrides[5];
              std::memcpy(outData + out_off, inData + in_off, elementSize);
            }
          }
        }
      }
    }
  }
}

void printTensor6D(const char *name, const uint8_t *data,
                   const int32_t dims[6]) {
  std::cout << name << " (N=" << dims[0] << ", C=" << dims[1]
            << ", H=" << dims[2] << ", W=" << dims[3] << ", X=" << dims[4]
            << ", Y=" << dims[5] << "):" << std::endl;

  int total = dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * dims[5];
  for (int i = 0; i < total; i++) {
    std::cout << std::setw(3) << (int)data[i] << " ";
    if ((i + 1) % dims[5] == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

void calculateStrides(int32_t strides[6], const int32_t dims[6],
                      int32_t elementSize) {
  strides[5] = elementSize;
  for (int i = 4; i >= 0; i--) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
}

int32_t getIndex6D(const int32_t coords[6], const int32_t dims[6]) {
  return coords[0] * (dims[1] * dims[2] * dims[3] * dims[4] * dims[5]) +
         coords[1] * (dims[2] * dims[3] * dims[4] * dims[5]) +
         coords[2] * (dims[3] * dims[4] * dims[5]) +
         coords[3] * (dims[4] * dims[5]) + coords[4] * dims[5] + coords[5];
}

bool verifyTranspose(const uint8_t *input, const uint8_t *output,
                     const int32_t inDims[6]) {
  for (int32_t n = 0; n < inDims[0]; n++) {
    for (int32_t c = 0; c < inDims[1]; c++) {
      for (int32_t h = 0; h < inDims[2]; h++) {
        for (int32_t w = 0; w < inDims[3]; w++) {
          for (int32_t x = 0; x < inDims[4]; x++) {
            for (int32_t y = 0; y < inDims[5]; y++) {
              int32_t inCoord[6] = {n, c, h, w, x, y};
              int32_t inIdx = getIndex6D(inCoord, inDims);
              int32_t outDims[6] = {inDims[0], inDims[5], inDims[1],
                                    inDims[2], inDims[3], inDims[4]};
              int32_t outCoord[6] = {n, y, c, h, w, x};
              int32_t outIdx = getIndex6D(outCoord, outDims);
              if (input[inIdx] != output[outIdx]) {
                std::cout << "Mismatch at [" << n << "," << c << "," << h << ","
                          << w << "," << x << "," << y << "]" << std::endl;
                std::cout << "Input[" << inIdx << "] = " << (int)input[inIdx]
                          << ", Output[" << outIdx
                          << "] = " << (int)output[outIdx] << std::endl;
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool test_small_tensor() {
  std::cout << "=== Test 1: Small Tensor (1x2x2x2x2x2) ===" << std::endl;
  int32_t inDims[6] = {1, 2, 2, 2, 2, 2};
  int32_t elementSize = 1;
  int32_t inStrides[6];
  calculateStrides(inStrides, inDims, elementSize);
  int32_t outDims[6] = {inDims[0], inDims[5], inDims[1],
                        inDims[2], inDims[3], inDims[4]};
  int32_t outStrides[6];
  calculateStrides(outStrides, outDims, elementSize);

  int total =
      inDims[0] * inDims[1] * inDims[2] * inDims[3] * inDims[4] * inDims[5];
  std::vector<uint8_t> input(total);
  for (int i = 0; i < total; i++) {
    input[i] = i;
  }
  std::vector<uint8_t> output(total, 0);

  transpose_nchwxy_to_nychwx(input.data(), output.data(), inDims, inStrides,
                             outStrides, elementSize);
  bool passed = verifyTranspose(input.data(), output.data(), inDims);
  std::cout << "Test 1: " << (passed ? "PASSED" : "FAILED") << std::endl
            << std::endl;
  return passed;
}

bool test_single_element() {
  std::cout << "=== Test 2: Single Element (1x1x1x1x1x1) ===" << std::endl;
  int32_t inDims[6] = {1, 1, 1, 1, 1, 1};
  int32_t elementSize = 1;
  int32_t inStrides[6], outStrides[6];
  calculateStrides(inStrides, inDims, elementSize);
  int32_t outDims[6] = {1, 1, 1, 1, 1, 1};
  calculateStrides(outStrides, outDims, elementSize);
  std::vector<uint8_t> input = {42};
  std::vector<uint8_t> output(1, 0);
  transpose_nchwxy_to_nychwx(input.data(), output.data(), inDims, inStrides,
                             outStrides, elementSize);
  bool passed = (output[0] == 42);
  std::cout << "Test 2: " << (passed ? "PASSED" : "FAILED") << std::endl
            << std::endl;
  return passed;
}

bool test_varying_y() {
  std::cout << "=== Test 3: Varying Y Dimension (1x2x2x2x2x3) ===" << std::endl;
  int32_t inDims[6] = {1, 2, 2, 2, 2, 3};
  int32_t elementSize = 1;
  int32_t inStrides[6], outStrides[6];
  calculateStrides(inStrides, inDims, elementSize);
  int32_t outDims[6] = {inDims[0], inDims[5], inDims[1],
                        inDims[2], inDims[3], inDims[4]};
  calculateStrides(outStrides, outDims, elementSize);
  int total =
      inDims[0] * inDims[1] * inDims[2] * inDims[3] * inDims[4] * inDims[5];
  std::vector<uint8_t> input(total);
  for (int i = 0; i < total; i++) {
    input[i] = i % 256;
  }
  std::vector<uint8_t> output(total, 0);
  transpose_nchwxy_to_nychwx(input.data(), output.data(), inDims, inStrides,
                             outStrides, elementSize);
  bool passed = verifyTranspose(input.data(), output.data(), inDims);
  std::cout << "Test 3: " << (passed ? "PASSED" : "FAILED") << std::endl
            << std::endl;
  return passed;
}

bool test_larger_tensor() {
  std::cout << "=== Test 4: Larger Tensor (2x3x4x2x2x3) ===" << std::endl;
  int32_t inDims[6] = {2, 3, 4, 2, 2, 3};
  int32_t elementSize = 1;
  int32_t inStrides[6], outStrides[6];
  calculateStrides(inStrides, inDims, elementSize);
  int32_t outDims[6] = {inDims[0], inDims[5], inDims[1],
                        inDims[2], inDims[3], inDims[4]};
  calculateStrides(outStrides, outDims, elementSize);
  int total =
      inDims[0] * inDims[1] * inDims[2] * inDims[3] * inDims[4] * inDims[5];
  std::vector<uint8_t> input(total);
  for (int i = 0; i < total; i++) {
    input[i] = i % 256;
  }
  std::vector<uint8_t> output(total, 0);
  transpose_nchwxy_to_nychwx(input.data(), output.data(), inDims, inStrides,
                             outStrides, elementSize);
  bool passed = verifyTranspose(input.data(), output.data(), inDims);
  std::cout << "Test 4: " << (passed ? "PASSED" : "FAILED") << std::endl
            << std::endl;
  return passed;
}

int main() {
  std::cout << "======================================" << std::endl;
  std::cout << "Transpose Simple 6D Test (NCHWXY->NYCHWX)" << std::endl;
  int passed = 0, total = 0;
  total++;
  if (test_single_element()) {
    passed++;
  }
  total++;
  if (test_small_tensor()) {
    passed++;
  }
  total++;
  if (test_varying_y()) {
    passed++;
  }
  total++;
  if (test_larger_tensor()) {
    passed++;
  }

  std::cout << "======================================" << std::endl;
  std::cout << "Test Results: " << passed << "/" << total << " passed"
            << std::endl;

  return (passed == total) ? 0 : 1;
}
