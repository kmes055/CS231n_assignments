from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cython', ['../gdrive/My Drive/CS231n/assignment2/cs231n/im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
