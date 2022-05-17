#pragma once

#include <torch/extension.h>

torch::Tensor LcsCuda(
        torch::Tensor output_tokens,
        torch::Tensor kwd_tokens,
        torch::Tensor output_lengths,
        torch::Tensor kwd_lengths,
        const size_t bos_idx,
        const size_t eos_idx,
        const size_t padding_idx,
        const size_t sep_idx);