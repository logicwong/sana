#include "lcs.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>      // std::pair

template <typename scalar_t>
__global__ void faster_lcs_kernel(
        const scalar_t* __restrict__ output_tokens,
        const scalar_t* __restrict__ kwd_tokens,
        const scalar_t* __restrict__ output_lengths,
        const scalar_t* __restrict__ kwd_lengths,
        const size_t bos_idx,
        const size_t eos_idx,
        const size_t padding_idx,
        const size_t sep_idx,
        const size_t output_size,
        const size_t kwd_size,
        int* __restrict__ masks) {

    const int index = blockIdx.x;
    const int offset = index * (output_size);

    auto opt_idx = [offset](int k) { return offset + k; };

    const scalar_t* output_begin = output_tokens + index * output_size;
    const scalar_t* kwd_begin = kwd_tokens + index * kwd_size;

    int kwd_id = 1;
    int m = *(kwd_lengths + index * 1) - 2;
    int n = *(output_lengths + index * 1) - 2;
    while (true){
        const int kwd_token = *(kwd_begin + m);
        const int output_token = *(output_begin + n);

        if (kwd_token == bos_idx){
            break;
        } else if (kwd_token == sep_idx){
            kwd_id += 1;
            m -= 1;
        } else if (kwd_token == output_token){
            masks[opt_idx(n)] = kwd_id;
            m -= 1;
            n -= 1;
        } else {
            n -= 1;
        }
    }
}


torch::Tensor LcsCuda(
        torch::Tensor output_tokens,
        torch::Tensor kwd_tokens,
        torch::Tensor output_lengths,
        torch::Tensor kwd_lengths,
        const size_t bos_idx,
        const size_t eos_idx,
        const size_t padding_idx,
        const size_t sep_idx) {

    const auto batch_size = output_tokens.size(0);
    const auto shared_size = output_tokens.size(1) * sizeof(short);

    at::TensorOptions options(output_tokens.device());
    options = options.dtype(at::ScalarType::Int);
    auto masks = torch::zeros({batch_size, output_tokens.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(output_tokens.device().index());

    AT_DISPATCH_ALL_TYPES(output_tokens.scalar_type(), "faster_lcs", ([&] {
        faster_lcs_kernel<scalar_t><<<batch_size, 1, shared_size, stream>>>(
            output_tokens.data_ptr<scalar_t>(),
            kwd_tokens.data_ptr<scalar_t>(),
            output_lengths.data_ptr<scalar_t>(),
            kwd_lengths.data_ptr<scalar_t>(),
            bos_idx,
            eos_idx,
            padding_idx,
            sep_idx,
            output_tokens.size(1),
            kwd_tokens.size(1),
            masks.data_ptr<int>());
    }));

    return masks;
}
