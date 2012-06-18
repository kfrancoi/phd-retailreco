from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

#Not lib to include right now
if sys.platform == 'darwin' or sys.platform == 'linux2':
	include_math_dir = "/usr/include"
	lib_math_dir = "/usr/lib"
elif sys.platform == 'win32':
	include_math_dir = ""
	lib_math_dir = ""

ext = Extension("distance", ["distance.pyx"],
	include_dirs=[numpy.get_include(), include_math_dir],
	library_dirs=[lib_math_dir])

setup(ext_modules=[ext],
	cmdclass={'build_ext':build_ext})