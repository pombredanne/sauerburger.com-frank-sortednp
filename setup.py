"""
Sortednp install script. Run

 pip3 install .

to install the sortednp python package.
"""
from setuptools import setup, Extension

class NumpyApiExtension(Extension):
    """
    Specialized setuptools Extension, which tries to solve the numpy
    dependence by postponing its import until pip took care of its install.
    """

    def __getattribute__(self, name):
        """
        Delegate the call to the base class, unless "include_dirs" is
        requested. In that case, try to import numpy and return its include
        path.
        """

        if name == "include_dirs":
            try:
                # Delayed import. Numpy should be installed by now.
                import numpy
            except ImportError:
                # The above import fails, if setuptools is used to install the
                # package. Setuptools does not install the dependencies
                # beforehand.
                raise Exception("Please use 'pip3 install .' or install numpy "
                                "manually before running this script.")

            return [numpy.get_include()]

        return super().__getattribute__(name)


def load_intro(path="README.md"):
    """
    Load and return the first paragraph from the README file.
    """
    with open(path) as readme_file:
        readme = readme_file.read()
    end = readme.index("#", 1)

    return readme[:end].strip()


BACKEND = NumpyApiExtension('_sortednp', language='c++',
                            extra_compile_args=['-g'],
                            sources=['sortednpmodule.cpp'],
                            depends=['numpy'])

setup(name='sortednp',
      version='0.0.0',
      packages=["sortednp"],
      install_requires=['numpy>=1.7'],
      test_suite='tests',
      description='Merge and intersect sorted numpy arrays.',
      long_description=load_intro(),
      url="https://gitlab.sauerburger.com/frank/sortednp",
      author="Frank Sauerburger",
      author_email="frank@sauerburger.com",
      keywords="merge intersect sorted numpy",
      license="MIT",
      python_requires='>=3',
      classifiers=[],

      ext_modules=[BACKEND])
