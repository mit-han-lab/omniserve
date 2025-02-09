// Inspired by vLLM-SmoothQuant: https://github.com/vllm-project/vllm/pull/1112 and TensorRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"
#include <cuda_fp16.h>
#include <cassert>

namespace vllm {
template <typename T, typename scale_type, bool use_per_token_dequant>
__global__ void dequant_add_residual_kernel(const int32_t *__restrict__ input,
                                            const T *__restrict__ residual,
                                            T *__restrict__ output,
                                            const scale_type scale, int num_tokens,
                                            int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    if constexpr (use_per_token_dequant) {
      output[token_idx * hidden_size + i] =
          (T)((((float)input[token_idx * hidden_size + i]) * to_float(scale[token_idx])) +
              (float)residual[token_idx * hidden_size + i]);
    } else {
      output[token_idx * hidden_size + i] =
          (T)((((float)input[token_idx * hidden_size + i]) * to_float(scale)) +
              (float)residual[token_idx * hidden_size + i]);
    }
  }
}

template <typename T>
__global__ void dequant_kernel(const int32_t *__restrict__ input,
                               T *__restrict__ output, half scale, int m,
                               int hidden_size, int input_stride, int out_stride) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    output[token_idx * out_stride + i] =
        (T)(((float)input[token_idx * input_stride + i]) * to_float(scale));
  }
}

template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(const T *__restrict__ input,
                             int8_t *__restrict__ output, scale_type scale,
                             int num_tokens, int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = from_float<T>(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / to_float(scale));
    }
  }
}


template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel_fuse_sum(const T *__restrict__ input,
                                      int8_t *__restrict__ output,
                                      scale_type input_sum,
                                      scale_type scale,
                                      int num_tokens,
                                      int hidden_size) {
  // TODO: get the sum here.
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  const int64_t token_idx_mul_hidden_size = token_idx * int64_t(hidden_size);

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    float sum_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx_mul_hidden_size + i];
      sum_val += val;
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    const float block_sum_val = blockReduceSum(sum_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = from_float<T>(block_amax_val / 127.0f);
      input_sum[token_idx] = from_float<T>(block_sum_val);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx_mul_hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx_mul_hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx_mul_hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx_mul_hidden_size + i]) / to_float(scale));
    }
  }
}
} // namespace vllm

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    at::Half scale) {
  int m = input.size(0);
  int n = input.size(1);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t, at::Half, false>
            <<<grid, block, 0, stream>>>(input.data_ptr<int32_t>(),
                                         residual.data_ptr<scalar_t>(),
                                         out.data_ptr<scalar_t>(), scale, m, n);
      });
}

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &scale) {  // [num_tokens]
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t, at::Half *, true>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), scale.data_ptr<at::Half>(), num_tokens, hidden_size);
      });
}

void invoke_dequant(torch::Tensor &out,   // [..., hidden_size]
                    torch::Tensor &input, // [..., hidden_size]
                    at::Half scale) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  int input_stride = input.stride(-2);
  int out_stride = out.stride(-2);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "dequant_kernel", [&] {
    vllm::dequant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), scale, num_tokens, hidden_size,
        input_stride, out_stride);
  });
}

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  at::Half scale) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    // It look like this function is never called. We hard code the template to half since input argument scale is at::Half.
    // using T = typename FloatTypeConverter<scalar_t>::Type;
    vllm::quant_kernel<half, half, false><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<scalar_t>()), out.data_ptr<int8_t>(), scale, num_tokens, hidden_size);
  });
}

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &scale) { // [num_tokens]
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    vllm::quant_kernel<T, T *, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), out.data_ptr<int8_t>(),
        reinterpret_cast<T*>(scale.data_ptr<scalar_t>()), num_tokens, hidden_size);
  });
}



// Get the sum of input tensor across the channel
void invoke_quant_fuse_sum(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  at::Half input_sum,
                  at::Half scale) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel_fuse_sum", [&] {
    vllm::quant_kernel_fuse_sum<scalar_t, at::Half, false><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(), input_sum, scale, num_tokens, hidden_size);
  });
}

void invoke_quant_fuse_sum(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &input_sum,
                  torch::Tensor &scale) { // [num_tokens]
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel_fuse_sum", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    vllm::quant_kernel_fuse_sum<T, T*, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), out.data_ptr<int8_t>(), reinterpret_cast<T*>(input_sum.data_ptr<scalar_t>()),
        reinterpret_cast<T*>(scale.data_ptr<scalar_t>()), num_tokens, hidden_size);
  });
}
