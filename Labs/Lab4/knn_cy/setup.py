from distutils.core import setup
import numpy
from distutils.extension import Extension


from Cython.Build import cythonize


setup(
    ext_modules=cythonize("knn.pyx"),
    include_dirs=[numpy.get_include()]
) 