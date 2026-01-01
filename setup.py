from setuptools import setup
from Cython.Build import cythonize

setup(
    name="cplayer",
    ext_modules=cythonize("cplayer.pyx"),
    py_modules=[],
    packages=[],
)
