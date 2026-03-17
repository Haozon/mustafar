from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(os.path.dirname(current_dir), 'csrc')

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
else:
    major, minor = (8, 6)

sm_arch = f"sm_{major}{minor}"
compute_arch = f"compute_{major}{minor}"

setup(
    name='mustafar_package_quant',
    ext_modules=[
        CUDAExtension(
            name='mustafar_package_quant',
            sources=[
                'pybind_quant.cpp',
                'mustafar_wrapper_quant.cu',
                os.path.join(csrc_dir, 'SpMM_API_Quant.cu'),
            ],
            include_dirs=[csrc_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'-arch={sm_arch}',
                    f'-gencode=arch={compute_arch},code={sm_arch}',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
