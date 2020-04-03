#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

setup(name='ThemisPy',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Postprocessing tools for Themis',
      author='Themis Development Team',
      author_email='Themis@perimeterinstitute.ca',
      url='https://github.com/aeb/ThemisPy',
      packages=find_packages(),
      install_requires=['numpy','matplotlib']
     )



