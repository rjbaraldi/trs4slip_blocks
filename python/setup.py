import numpy as np
from setuptools import Extension
from setuptools import setup, find_packages


extensions = [
    Extension("trs4slip",
              ["extension/trs4slip.pyx", "src/trs4slip.cpp"],
              extra_compile_args=["-std=c++2a"],
              include_dirs=[np.get_include(), "src/"])]


setup(
    name="trs4slip",
    version="0.1.0",
    ext_modules=extensions,
    packages=find_packages(),
)
