#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "mustafar_wrapper_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.doc() = "PyTorch extension for Mustafar quantized batched spmv CUDA kernel";
    
    // 原始函数名
    m.def(
        "mustafar_key_formulation_quant",
        &mustafar_key_formulation_quant,
        py::arg("bmp"),
        py::arg("NZ_quant"),
        py::arg("tile_offsets"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("B"),
        py::arg("M_Global"),
        py::arg("K_Global"),
        py::arg("Batch_Size"),
        py::arg("num_key_value_groups"),
        py::arg("bit"),
        py::arg("capacity"),
        py::arg("dequant_mode") = 0,
        "Quantized key formulation kernel (dequant_mode: 0=speed, 1=memory)"
    );
    m.def(
        "mustafar_value_formulation_quant",
        &mustafar_value_formulation_quant,
        py::arg("bmp"),
        py::arg("NZ_quant"),
        py::arg("tile_offsets"),
        py::arg("tile_counts"),
        py::arg("tile_units"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("B"),
        py::arg("Reduction_Workspace"),
        py::arg("M_Global"),
        py::arg("K_Global"),
        py::arg("Batch_Size"),
        py::arg("num_key_value_groups"),
        py::arg("bit"),
        py::arg("capacity"),
        py::arg("dequant_mode") = 0,
        py::arg("split_k") = 1,
        py::arg("value_tile_config") = 0,
        "Quantized value formulation kernel (dequant_mode: 0=speed, 1=memory)"
    );
    
    // 添加别名以兼容模型代码
    m.def(
        "mustafar_quant_sparse_forward",
        &mustafar_key_formulation_quant,
        py::arg("bmp"),
        py::arg("NZ_quant"),
        py::arg("tile_offsets"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("B"),
        py::arg("M_Global"),
        py::arg("K_Global"),
        py::arg("Batch_Size"),
        py::arg("num_key_value_groups"),
        py::arg("bit"),
        py::arg("capacity"),
        py::arg("dequant_mode") = 0,
        "Quantized sparse forward (Key) - alias for mustafar_key_formulation_quant"
    );
    m.def(
        "mustafar_quant_sparse_value_forward",
        &mustafar_value_formulation_quant,
        py::arg("bmp"),
        py::arg("NZ_quant"),
        py::arg("tile_offsets"),
        py::arg("tile_counts"),
        py::arg("tile_units"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("B"),
        py::arg("Reduction_Workspace"),
        py::arg("M_Global"),
        py::arg("K_Global"),
        py::arg("Batch_Size"),
        py::arg("num_key_value_groups"),
        py::arg("bit"),
        py::arg("capacity"),
        py::arg("dequant_mode") = 0,
        py::arg("split_k") = 1,
        py::arg("value_tile_config") = 0,
        "Quantized sparse value forward - alias for mustafar_value_formulation_quant"
    );
}
