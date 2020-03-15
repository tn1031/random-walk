import os
import platform
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

if platform.system() == "Darwin":
    os.environ["CC"] = "/usr/local/bin/gcc-9"
    os.environ["CXX"] = "/usr/local/bin/g++-9"
else:
    os.environ["CC"] = "/usr/bin/gcc"
    os.environ["CXX"] = "/usr/bin/g++"
    os.environ["CFLAGS"] = "-std=c++11"

setup(
    ext_modules=cythonize(
        ["randomwalk/c_sample_neighbor.pyx", "randomwalk/c_randomwalk.pyx"]
    )
)
