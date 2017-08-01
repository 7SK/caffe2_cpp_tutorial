#include "caffe2/operator/affine_scale_op.h"
#include "caffe2/core/context_hip.h"

namespace caffe2 {

//namespace {

__global__ void AffineScaleKernel(const int N, const int C, const float* X, const float* M, const float* S, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] * S[i / C] + M[i / C];
  }
}

__global__ void AffineScaleInverseKernel(const int N, const int C, const float* X, const float* M, const float* S, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = (X[i] - M[i / C]) / (S[i / C] + 1e-8);
  }
}

//}  // namespace

template <>
bool AffineScaleOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& M = Input(1);
  auto& S = Input(2);
  auto* Y = Output(0);
  DCHECK_EQ(M.size(), X.dim(0));
  DCHECK_EQ(S.size(), X.dim(0));
  Y->ResizeLike(X);
  if (X.size() > 0) {
    auto size = X.size() / X.dim(0);
    if (inverse_) {
      hipLaunchKernel(AffineScaleInverseKernel,CAFFE_GET_BLOCKS(X.size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
        X.size(), size, X.data<float>(), M.data<float>(), S.data<float>(), Y->mutable_data<float>());
    } else {
      hipLaunchKernel(AffineScaleKernel,CAFFE_GET_BLOCKS(X.size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
        X.size(), size, X.data<float>(), M.data<float>(), S.data<float>(), Y->mutable_data<float>());
    }
  }
  return true;
}

//namespace {

__global__ void AffineScaleGradientKernel(const int N, const int C, const float* dY, const float* S, float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * S[i / C];
  }
}

__global__ void AffineScaleInverseGradientKernel(const int N, const int C, const float* dY, const float* S, float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] / (S[i / C] + 1e-8);
  }
}

//}  // namespace

template <>
bool AffineScaleGradientOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& S = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(S.size(), X.dim(0));
  DCHECK_EQ(dY.size(), X.size());
  dX->ResizeLike(X);
  if (X.size() > 0) {
    auto size = X.size() / X.dim(0);
    if (inverse_) {
      hipLaunchKernel(AffineScaleInverseGradientKernel,CAFFE_GET_BLOCKS(dY.size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
        dY.size(), size, dY.data<float>(), S.data<float>(), dX->mutable_data<float>());
    } else {
      hipLaunchKernel(AffineScaleGradientKernel,CAFFE_GET_BLOCKS(dY.size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
        dY.size(), size, dY.data<float>(), S.data<float>(), dX->mutable_data<float>());
    }
  }
  return true;
}


//namespace {

REGISTER_HIP_OPERATOR(AffineScale, AffineScaleOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(AffineScaleGradient, AffineScaleGradientOp<float, HIPContext>);

//}  // namespace

}  // namespace caffe2
