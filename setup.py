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
                raise Exception("Rerun the previous pip command or install "
                                "numpy manually before running this script.")

            return [numpy.get_include()]

        return super().__getattribute__(name)


def load_long_description(*filenames, paragraphs=None):
    """
    Try to load all paragraph from any of the given file. If none of the
    files could be opened, return None.
    """
    for filename in filenames:
        try:
            with open(filename) as readme_file:
                content = readme_file.read()

            paragraph = "\n\n".join(content.split("\n\n")[0:paragraphs])
            return paragraph

        except FileNotFoundError as _:
            pass

    return None

BACKEND = NumpyApiExtension('sortednp._internal', language='c++',
                            extra_compile_args=['-g'],
                            sources=['src/sortednpmodule.cpp'],
                            depends=['numpy'])

setup(name='sortednp',
      version='0.2.0-rc',
      packages=["sortednp", "sortednp.tests"],
      package_dir={"": "src"},
      install_requires=['numpy>=1.14'],
      test_suite='sortednp.tests',
      description='Merge and intersect sorted numpy arrays.',
      long_description=load_long_description("README.md"),
      long_description_content_type='text/markdown',
      url="https://gitlab.sauerburger.com/frank/sortednp",
      author="Frank Sauerburger",
      author_email="frank@sauerburger.com",
      keywords="merge intersect sorted numpy",
      license="MIT",
      python_requires='>=3',
      platforms=["Linux", "Unix"],
      classifiers=["Intended Audience :: Developers",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: C"],
      ext_modules=[BACKEND])
