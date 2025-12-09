from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(os.path.dirname(current_dir), 'csrc')

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
                    '-arch=sm_80',  # 根据你的 GPU 架构调整
                    '-gencode=arch=compute_80,code=sm_80',
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
