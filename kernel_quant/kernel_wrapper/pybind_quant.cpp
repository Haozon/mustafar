#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "mustafar_wrapper_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch extension for Mustafar quantized batched spmv CUDA kernel";
    m.def("mustafar_key_formulation_quant", &mustafar_key_formulation_quant, 
          "Quantized key formulation kernel");
    m.def("mustafar_value_formulation_quant", &mustafar_value_formulation_quant, 
          "Quantized value formulation kernel");
}
