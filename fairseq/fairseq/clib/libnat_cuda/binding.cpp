/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 This code is partially adpoted from https://github.com/1ytic/pytorch-edit-distance
 */

#include "edit_dist.h"
#include "lcs.h"
#include <torch/types.h>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor LevenshteinDistance(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length) {

    CHECK_INPUT(source);
    CHECK_INPUT(target);
    CHECK_INPUT(source_length);
    CHECK_INPUT(target_length);
    return LevenshteinDistanceCuda(source, target, source_length, target_length);
}

torch::Tensor GenerateDeletionLabel(
        torch::Tensor source,
        torch::Tensor operations) {

    CHECK_INPUT(source);
    CHECK_INPUT(operations);
    return GenerateDeletionLabelCuda(source, operations);
}

std::pair<torch::Tensor, torch::Tensor> GenerateInsertionLabel(
        torch::Tensor target,
        torch::Tensor operations) {

    CHECK_INPUT(target);
    CHECK_INPUT(operations);
    return GenerateInsertionLabelCuda(target, operations);
}

torch::Tensor Lcs(
        torch::Tensor output_tokens,
        torch::Tensor kwd_tokens,
        torch::Tensor output_lengths,
        torch::Tensor kwd_lengths,
        const size_t bos_idx,
        const size_t eos_idx,
        const size_t padding_idx,
        const size_t sep_idx) {

    CHECK_INPUT(output_tokens);
    CHECK_INPUT(kwd_tokens);
    CHECK_INPUT(output_lengths);
    CHECK_INPUT(kwd_lengths);
    return LcsCuda(output_tokens, kwd_tokens, output_lengths, kwd_lengths, bos_idx, eos_idx, padding_idx, sep_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("levenshtein_distance", &LevenshteinDistance, "Levenshtein distance");
    m.def("generate_deletion_labels", &GenerateDeletionLabel, "Generate Deletion Label");
    m.def("generate_insertion_labels", &GenerateInsertionLabel, "Generate Insertion Label");
    m.def("lcs", &Lcs, "longestCommonSubsequence");
}
