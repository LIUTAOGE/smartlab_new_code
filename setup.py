from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["extract_keyframe_service.py", "evaluator.py"]))