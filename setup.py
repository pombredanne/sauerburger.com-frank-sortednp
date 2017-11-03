from setuptools import setup, Extension

module1 = Extension('sortednp', language='c++',
                 extra_compile_args=['-std=c99'],
                    sources = ['sortednpmodule.c'])

setup (name = 'sortednp',
       version = '0.0.0',
       description = 'Merge and intersect sorted numpy arrays.',
       ext_modules = [module1])
