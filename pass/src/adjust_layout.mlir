// RUN: my-opt --adjust-layout %s | FileCheck %s

module {
  // CHECK-LABEL: @test_conv2d_f16
  func.func @test_conv2d_f16(%arg0: tensor<1x320x64x64xf16>) -> tensor<1x320x64x64xf16> {
    // CHECK: "tosa.const"() <{value = dense<0.000000e+00> : tensor<320x1x1x320xf16>}> : () -> tensor<320x1x1x320xf16>
    %0 = "tosa.const"() {value = dense<0.0> : tensor<320x320x1x1xf16>} : () -> tensor<320x320x1x1xf16>
    %1 = "tosa.const"() {value = dense<0.0> : tensor<320xf16>} : () -> tensor<320xf16>
    // CHECK: arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    // CHECK: tosa.transpose
    // CHECK-SAME: (tensor<1x320x64x64xf16>, tensor<4xi32>) -> tensor<1x64x64x320xf16>
    // CHECK: "tosa.conv2d"
    // CHECK-SAME: (tensor<1x64x64x320xf16>, tensor<320x1x1x320xf16>, tensor<320xf16>) -> tensor<1x64x64x320xf16>
    %2 = "tosa.conv2d"(%arg0, %0, %1) {
      acc_type = f16, 
      dilation = array<i64: 1, 1>, 
      pad = array<i64: 0, 0, 0, 0>, 
      stride = array<i64: 1, 1>
    } : (tensor<1x320x64x64xf16>, tensor<320x320x1x1xf16>, tensor<320xf16>) -> tensor<1x320x64x64xf16>
    // CHECK: tosa.transpose
    // CHECK-SAME: (tensor<1x64x64x320xf16>, tensor<4xi32>) -> tensor<1x320x64x64xf16>
    // CHECK: return 
    // CHECK-SAME: tensor<1x320x64x64xf16>
    return %2 : tensor<1x320x64x64xf16>
  }

}
