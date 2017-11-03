from setuptools import setup, Extension

import numpy


module1 = Extension('sortednp', language='c++',
                 extra_compile_args=['-std=c99'],
                 include_dirs=[numpy.get_include()],
                    sources = ['sortednpmodule.c'])

setup (name = 'sortednp',
       version = '0.0.0',
       install_requires=['numpy'],
       description = 'Merge and intersect sorted numpy arrays.',
       ext_modules = [module1])
