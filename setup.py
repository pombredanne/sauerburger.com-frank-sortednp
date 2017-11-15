from setuptools import setup, Extension

import numpy


snp_backend = Extension('_sortednp', language='c++',
    extra_compile_args=['-g'],
    include_dirs=[numpy.get_include()],
    sources=['sortednpmodule.cpp'],
	headers=['sortednpmodule.h'])

setup (name='sortednp',
    version='0.0.0',
    packages=["sortednp"],
    install_requires=['numpy'],
    test_suite='tests',
    description='Merge and intersect sorted numpy arrays.',
    ext_modules=[snp_backend])
