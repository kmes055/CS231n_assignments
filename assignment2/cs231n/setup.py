from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            #C:\Users\JunSoo\Google 드라이브\
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
