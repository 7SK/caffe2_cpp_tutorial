#include "caffe2/operator/diagonal_op.h"
#include "caffe2/core/context_hip.h"

namespace caffe2 {

int diagonal_op_step(const Tensor<HIPContext> &tensor) {
  auto step = 0;
  for (auto d: tensor.dims()) {
    step = step * d + 1;
  }
  return step;
}

int diagonal_op_size(const Tensor<HIPContext> &tensor) {
  auto size = tensor.dim(0);
  for (auto d: tensor.dims()) {
    if (size > d) size = d;
  }
  return size;
}

int diagonal_op_offset(const Tensor<HIPContext> &tensor, const std::vector<TIndex> &offset) {
  auto off = 0, i = 0;
  for (auto d: tensor.dims()) {
    off = off * d + offset[i++];
  }
  return off;
}

//namespace {

__global__ void DiagonalKernel(const int N, const int C, const int D, const float* X, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i * C + D];
  }
}

//}  // namespace

template <>
bool DiagonalOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  auto size = diagonal_op_size(X);
  Y->Resize(size);
  if (size > 0) {
    auto step = diagonal_op_step(X);
    auto offset = diagonal_op_offset(X, offset_);
    hipLaunchKernel(DiagonalKernel,CAFFE_GET_BLOCKS(Y->size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
      Y->size(), step, offset, X.data<float>(), Y->mutable_data<float>());
  }
  return true;
}

//namespace {

__global__ void DiagonalGradientKernel(const int N, const int C, const int D, const float* dY, float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = (i >= D && (i - D) % C == 0 ? dY[i] : 0);
  }
}

//}  // namespace

template <>
bool DiagonalGradientOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  auto size = diagonal_op_size(X);
  DCHECK_EQ(dY.size(), size);
  if (size > 0) {
    auto step = diagonal_op_step(X);
    auto offset = diagonal_op_offset(X, offset_);
    hipLaunchKernel(DiagonalGradientKernel, CAFFE_GET_BLOCKS(dX->size()), CAFFE_HIP_NUM_THREADS, 0, context_.hip_stream(),
      dX->size(), step, offset, dY.data<float>(), dX->mutable_data<float>());
  }
  return true;
}


//namespace {

REGISTER_HIP_OPERATOR(Diagonal, DiagonalOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(DiagonalGradient, DiagonalGradientOp<float, HIPContext>);

//}  // namespace

}  // namespace caffe2
