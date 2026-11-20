// Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "softdtw.h"

namespace flash_softdtw {
namespace {

struct Row {
  int64_t offset;
  int64_t lo;
  int64_t hi;
};

void check_index(const at::Tensor& tensor, at::IntArrayRef shape,
                 const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
  TORCH_CHECK(tensor.scalar_type() == at::kInt ||
                  tensor.scalar_type() == at::kLong,
              name, " must be int32 or int64");
}

void check_inputs(const at::Tensor& cost, const at::Tensor& x_lens,
                  const at::Tensor& y_lens,
                  const c10::optional<at::Tensor>& begin,
                  const c10::optional<at::Tensor>& end, double gamma,
                  double warp) {
  TORCH_CHECK(cost.device().is_cpu(), "cost must be on CPU");
  TORCH_CHECK(cost.dim() == 3, "cost must have shape [B, T, W]");
  TORCH_CHECK(cost.scalar_type() == at::kFloat ||
                  cost.scalar_type() == at::kDouble ||
                  cost.scalar_type() == at::kHalf ||
                  cost.scalar_type() == at::kBFloat16,
              "cost must be float32, float64, float16 or bfloat16");
  TORCH_CHECK(std::isfinite(gamma) && gamma > 0,
              "gamma must be finite and positive");
  TORCH_CHECK(std::isfinite(warp), "warp must be finite");
  check_index(x_lens, {cost.size(0)}, "x_lens");
  check_index(y_lens, {cost.size(0)}, "y_lens");
  TORCH_CHECK(begin.has_value() == end.has_value(),
              "begin and end must be provided together");
  if (begin.has_value()) {
    check_index(*begin, {cost.size(0), cost.size(1)}, "begin");
    check_index(*end, {cost.size(0), cost.size(1)}, "end");
  }
}

// Normalize metadata once, including non-contiguous int32/int64 views.
struct Metadata {
  at::Tensor x, y, begin, end;

  Metadata(const at::Tensor& x_lens, const at::Tensor& y_lens,
           const c10::optional<at::Tensor>& lo,
           const c10::optional<at::Tensor>& hi)
      : x(x_lens.to(at::kLong).contiguous()),
        y(y_lens.to(at::kLong).contiguous()) {
    if (lo.has_value()) {
      begin = lo->to(at::kLong).contiguous();
      end = hi->to(at::kLong).contiguous();
    }
  }

  std::vector<Row> rows(int64_t b, int64_t t, int64_t s, int64_t tmax,
                        int64_t width) const {
    std::vector<Row> result(t);
    const auto* lo = begin.defined() ? begin.const_data_ptr<int64_t>() : nullptr;
    const auto* hi = end.defined() ? end.const_data_ptr<int64_t>() : nullptr;
    for (int64_t i = 0; i < t; ++i) {
      const int64_t offset = lo ? lo[b * tmax + i] : 0;
      result[i] = {offset, std::max<int64_t>(offset, 0),
                   std::min({s, offset + width, hi ? hi[b * tmax + i] : s})};
    }
    return result;
  }
};

template <typename acc_t>
acc_t read_row(const acc_t* data, const Row& row, int64_t j) {
  return j >= row.lo && j < row.hi
      ? data[j - row.offset]
      : std::numeric_limits<acc_t>::infinity();
}

template <typename acc_t>
acc_t softmin(acc_t diagonal, acc_t up, acc_t left, acc_t gamma) {
  const acc_t minimum = std::min({diagonal, up, left});
  if (!std::isfinite(minimum)) return minimum;
  const acc_t sum = std::exp((minimum - diagonal) / gamma) +
                    std::exp((minimum - up) / gamma) +
                    std::exp((minimum - left) / gamma);
  return minimum - gamma * std::log(sum);
}

}  // namespace

std::vector<at::Tensor> forward_cpu(
    const at::Tensor& cost, const at::Tensor& x_lens,
    const at::Tensor& y_lens, const c10::optional<at::Tensor>& begin,
    const c10::optional<at::Tensor>& end, double gamma, double warp,
    bool save_r) {
  check_inputs(cost, x_lens, y_lens, begin, end, gamma, warp);
  const int64_t batch = cost.size(0), tmax = cost.size(1), width = cost.size(2);
  const auto acc_dtype = cost.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  auto values = at::full({batch}, std::numeric_limits<double>::infinity(),
                         cost.options().dtype(acc_dtype));
  auto r = save_r
      ? at::full(cost.sizes(), std::numeric_limits<double>::infinity(),
                 cost.options().dtype(acc_dtype))
      : at::empty({0}, cost.options().dtype(acc_dtype));
  const Metadata meta(x_lens, y_lens, begin, end);
  const auto* xs = meta.x.const_data_ptr<int64_t>();
  const auto* ys = meta.y.const_data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cost.scalar_type(),
                                "softdtw_forward_cpu", [&] {
    using acc_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    const auto* costs = cost.const_data_ptr<scalar_t>();
    auto* output = values.mutable_data_ptr<acc_t>();
    auto* saved = save_r ? r.mutable_data_ptr<acc_t>() : nullptr;
    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t g = static_cast<acc_t>(gamma), w = static_cast<acc_t>(warp);
    at::parallel_for(0, batch, 1, [&](int64_t first, int64_t last) {
      for (int64_t b = first; b < last; ++b) {
        const int64_t t = std::max<int64_t>(xs[b], 0);
        const int64_t s = std::max<int64_t>(ys[b], 0);
        if (t == 0 || s == 0) {
          output[b] = t == 0 && s == 0 ? acc_t(0) : inf;
          continue;
        }
        // The end cell lies outside the stored rows: unreachable.
        if (t > tmax) continue;
        const auto rows = meta.rows(b, t, s, tmax, width);
        std::vector<acc_t> previous(width, inf), current(width, inf);
        for (int64_t i = 0; i < t; ++i) {
          std::fill(current.begin(), current.end(), inf);
          const Row& row = rows[i];
          for (int64_t j = row.lo; j < row.hi; ++j) {
            const int64_t k = j - row.offset;
            const acc_t diagonal = i > 0
                ? read_row(previous.data(), rows[i - 1], j - 1)
                : (j == 0 ? acc_t(0) : inf);
            const acc_t up = i > 0
                ? read_row(previous.data(), rows[i - 1], j) : inf;
            const acc_t left = read_row(current.data(), row, j - 1);
            const acc_t prefix = softmin(diagonal, up + w, left + w, g);
            // Inactive predecessors remain unreachable even for non-finite costs.
            const acc_t value = prefix == inf ? inf : prefix + static_cast<acc_t>(
                costs[b * cost.stride(0) + i * cost.stride(1) + k * cost.stride(2)]);
            current[k] = value;
            if (save_r) saved[(b * tmax + i) * width + k] = value;
          }
          previous.swap(current);
        }
        output[b] = read_row(previous.data(), rows.back(), s - 1);
      }
    });
  });
  return {values, r};
}

at::Tensor backward_cpu(
    const at::Tensor& cost, const at::Tensor& r, const at::Tensor& x_lens,
    const at::Tensor& y_lens, const c10::optional<at::Tensor>& begin,
    const c10::optional<at::Tensor>& end, const at::Tensor& grad_values,
    double gamma, double warp) {
  check_inputs(cost, x_lens, y_lens, begin, end, gamma, warp);
  const int64_t batch = cost.size(0), tmax = cost.size(1), width = cost.size(2);
  const auto acc_dtype = cost.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  TORCH_CHECK(r.device().is_cpu() && r.sizes() == cost.sizes() &&
                  r.scalar_type() == acc_dtype,
              "R must have the cost shape and accumulation dtype on CPU");
  TORCH_CHECK(grad_values.device().is_cpu() && grad_values.dim() == 1 &&
                  grad_values.size(0) == batch,
              "grad_values must have shape [B] on CPU");
  const auto saved = r.contiguous();
  const auto grad_output = grad_values.to(acc_dtype).contiguous();
  auto grad = at::zeros(cost.sizes(), cost.options());
  const Metadata meta(x_lens, y_lens, begin, end);
  const auto* xs = meta.x.const_data_ptr<int64_t>();
  const auto* ys = meta.y.const_data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cost.scalar_type(),
                                "softdtw_backward_cpu", [&] {
    using acc_t = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    const auto* costs = cost.const_data_ptr<scalar_t>();
    const auto* rdata = saved.const_data_ptr<acc_t>();
    const auto* upstream = grad_output.const_data_ptr<acc_t>();
    auto* gradient = grad.mutable_data_ptr<scalar_t>();
    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t g = static_cast<acc_t>(gamma), w = static_cast<acc_t>(warp);
    at::parallel_for(0, batch, 1, [&](int64_t first, int64_t last) {
      for (int64_t b = first; b < last; ++b) {
        const int64_t t = std::max<int64_t>(xs[b], 0);
        const int64_t s = std::max<int64_t>(ys[b], 0);
        if (t == 0 || s == 0 || width == 0 || t > tmax) continue;
        const auto rows = meta.rows(b, t, s, tmax, width);
        const auto* rb = rdata + b * tmax * width;
        if (!std::isfinite(read_row(rb + (t - 1) * width, rows.back(), s - 1))) {
          continue;
        }
        std::vector<acc_t> current(width, 0), previous(width, 0);
        current[s - 1 - rows.back().offset] = acc_t(1);
        for (int64_t i = t; i-- > 0;) {
          std::fill(previous.begin(), previous.end(), acc_t(0));
          const Row& row = rows[i];
          for (int64_t j = row.hi; j-- > row.lo;) {
            const int64_t k = j - row.offset;
            const acc_t expectation = current[k];
            if (expectation == 0 || !std::isfinite(rb[i * width + k])) continue;
            const acc_t c = static_cast<acc_t>(costs[
                b * cost.stride(0) + i * cost.stride(1) + k * cost.stride(2)]);
            if (!std::isfinite(c)) continue;
            gradient[(b * tmax + i) * width + k] =
                static_cast<scalar_t>(expectation * upstream[b]);
            const acc_t diagonal = i > 0
                ? read_row(rb + (i - 1) * width, rows[i - 1], j - 1)
                : (j == 0 ? acc_t(0) : inf);
            const acc_t up = i > 0
                ? read_row(rb + (i - 1) * width, rows[i - 1], j) + w : inf;
            const acc_t left = read_row(rb + i * width, row, j - 1) + w;
            const acc_t minimum = std::min({diagonal, up, left});
            if (!std::isfinite(minimum)) continue;
            // Normalize predecessor weights directly to avoid cancellation in R - cost.
            const acc_t pd = std::exp((minimum - diagonal) / g);
            const acc_t pu = std::exp((minimum - up) / g);
            const acc_t pl = std::exp((minimum - left) / g);
            const acc_t scale = expectation / (pd + pu + pl);
            if (i > 0) {
              const Row& prior = rows[i - 1];
              if (j - 1 >= prior.lo && j - 1 < prior.hi)
                previous[j - 1 - prior.offset] += scale * pd;
              if (j >= prior.lo && j < prior.hi)
                previous[j - prior.offset] += scale * pu;
            }
            if (j - 1 >= row.lo) current[k - 1] += scale * pl;
          }
          current.swap(previous);
        }
      }
    });
  });
  return grad;
}

}  // namespace flash_softdtw
