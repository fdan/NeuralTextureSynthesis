from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='histogram',
    ext_modules=[
        CUDAExtension('histogram', [
            'histogram.cpp',
            'histogram_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
