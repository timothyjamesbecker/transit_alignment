from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import sys

ext_modules=[ Extension('transit_utils',
              ['transit_utils.pyx'],
              libraries=['m'],
              extra_compile_args = ['-O3','-ffast-math','-fno-associative-math'])]

setup(
    name = 'transit_utils',
    cmdclass = {"build_ext": build_ext},
    # ext_modules = ext_modules,
    ext_modules = cythonize(ext_modules,compiler_directives={'language_level':'3'}),
    include_dirs=[np.get_include()]
)