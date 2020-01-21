# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='visibility',
    version='0.1',
    author="Daniele Cattaneo",
    author_email="cattaneo@informatik.uni-freiburg.de",
    url="https://github.com/catta202000/CMRNet",
    ext_modules=[
        CUDAExtension('visibility', [
            'src/visibility.cpp',
            'src/visibility_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })