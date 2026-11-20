// Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.

#include <torch/extension.h>

#include "softdtw.h"

namespace flash_softdtw {
namespace {

std::vector<at::Tensor> forward(const at::Tensor& cost, const at::Tensor& x_lens,
                                const at::Tensor& y_lens,
                                const c10::optional<at::Tensor>& begin,
                                const c10::optional<at::Tensor>& end,
                                double gamma, double warp, bool save_r,
                                int64_t rows_per_block) {
  if (cost.is_cuda()) {
#ifdef FLASH_SOFTDTW_WITH_CUDA
    return forward_cuda(cost, x_lens, y_lens, begin, end, gamma, warp, save_r,
                        rows_per_block);
#else
    TORCH_CHECK(false, "flash_softdtw was built without CUDA support");
#endif
  }
  return forward_cpu(cost, x_lens, y_lens, begin, end, gamma, warp, save_r);
}

at::Tensor backward(const at::Tensor& cost, const at::Tensor& r,
                    const at::Tensor& x_lens, const at::Tensor& y_lens,
                    const c10::optional<at::Tensor>& begin,
                    const c10::optional<at::Tensor>& end,
                    const at::Tensor& grad_values, double gamma, double warp,
                    int64_t rows_per_block) {
  if (cost.is_cuda()) {
#ifdef FLASH_SOFTDTW_WITH_CUDA
    return backward_cuda(cost, r, x_lens, y_lens, begin, end, grad_values,
                         gamma, warp, rows_per_block);
#else
    TORCH_CHECK(false, "flash_softdtw was built without CUDA support");
#endif
  }
  return backward_cpu(cost, r, x_lens, y_lens, begin, end, grad_values, gamma,
                      warp);
}

}  // namespace
}  // namespace flash_softdtw

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Flash-SoftDTW native kernels";
  m.def("forward", &flash_softdtw::forward, py::arg("cost"), py::arg("x_lens"),
        py::arg("y_lens"), py::arg("begin"), py::arg("end"), py::arg("gamma"),
        py::arg("warp"), py::arg("save_r"), py::arg("rows_per_block") = 0);
  m.def("backward", &flash_softdtw::backward, py::arg("cost"), py::arg("r"),
        py::arg("x_lens"), py::arg("y_lens"), py::arg("begin"), py::arg("end"),
        py::arg("grad_values"), py::arg("gamma"), py::arg("warp"),
        py::arg("rows_per_block") = 0);
#ifdef FLASH_SOFTDTW_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}
