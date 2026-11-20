// Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
//
// Flash-SoftDTW: shared operator contract for the CPU and CUDA backends.
//
// Problem definition (0-based cells, one problem per batch element b):
//
//   * Valid cells are (i, j) with 0 <= i < x_lens[b] and 0 <= j < y_lens[b].
//     If the end cell is not stored (x_lens[b] > T, or y_lens[b] > W in dense
//     mode) the problem is unreachable: value = +inf and the gradient is 0.
//   * Dense mode (no band): every valid cell is active and its cost is
//     cost[b, i, j].
//   * Banded mode: row i is restricted to columns begin[b, i] <= j < end[b, i]
//     (additionally clipped to the valid range). The cost tensor is stored
//     compactly, i.e. the cost of cell (i, j) is cost[b, i, j - begin[b, i]].
//     Columns beyond cost.size(2) are treated as outside the band.
//     Both begin[b, :] and end[b, :] must be non-decreasing in i (this holds
//     for Sakoe-Chiba and every monotone warping window).
//
// Recursion, with softmin_g(a, b, c) = -g * log(exp(-a/g) + exp(-b/g) + exp(-c/g)):
//
//   R(-1, -1) = 0,  R(-1, j) = R(i, -1) = +inf,  R(inactive cell) = +inf
//   R(i, j)   = cost(i, j) + softmin_g(R(i-1, j-1),
//                                      R(i-1, j) + warp,
//                                      R(i, j-1) + warp)
//   value[b]  = R(x_lens[b] - 1, y_lens[b] - 1)
//
// The gradient of value[b] w.r.t. cost(i, j) is the expected alignment
// E(i, j) under the Gibbs distribution over monotone paths. It is returned in
// the same (dense or compact) layout as `cost`, with zeros outside the band.

#pragma once

#include <torch/extension.h>

#include <vector>

namespace flash_softdtw {

// Returns {values [B] (accumulation dtype), R [B, T, W] (accumulation dtype,
// empty when save_r == false)}. R is an opaque buffer for backward_*: only its
// active cells are defined.
std::vector<at::Tensor> forward_cpu(const at::Tensor& cost,
                                    const at::Tensor& x_lens,
                                    const at::Tensor& y_lens,
                                    const c10::optional<at::Tensor>& begin,
                                    const c10::optional<at::Tensor>& end,
                                    double gamma, double warp, bool save_r);

// Returns grad_cost with the same shape and dtype as `cost`.
at::Tensor backward_cpu(const at::Tensor& cost, const at::Tensor& r,
                        const at::Tensor& x_lens, const at::Tensor& y_lens,
                        const c10::optional<at::Tensor>& begin,
                        const c10::optional<at::Tensor>& end,
                        const at::Tensor& grad_values, double gamma,
                        double warp);

#ifdef FLASH_SOFTDTW_WITH_CUDA
// `rows_per_block` <= 0 selects the launch configuration automatically.
std::vector<at::Tensor> forward_cuda(const at::Tensor& cost,
                                     const at::Tensor& x_lens,
                                     const at::Tensor& y_lens,
                                     const c10::optional<at::Tensor>& begin,
                                     const c10::optional<at::Tensor>& end,
                                     double gamma, double warp, bool save_r,
                                     int64_t rows_per_block);

at::Tensor backward_cuda(const at::Tensor& cost, const at::Tensor& r,
                         const at::Tensor& x_lens, const at::Tensor& y_lens,
                         const c10::optional<at::Tensor>& begin,
                         const c10::optional<at::Tensor>& end,
                         const at::Tensor& grad_values, double gamma,
                         double warp, int64_t rows_per_block);
#endif

}  // namespace flash_softdtw
