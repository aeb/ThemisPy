#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer
import glob

script_dir = glob.glob('scripts/themispy_*')
script_files = []
for sf in script_dir :
    if ( sf[-1]!='~' ) :
        script_files.append(sf)

print("ThemisPy script files:",script_files)

        
setup(name='ThemisPy',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Postprocessing tools for Themis',
      author='Themis Development Team',
      author_email='Themis@perimeterinstitute.ca',
      url='https://github.com/aeb/ThemisPy',
      packages=find_packages(),
      install_requires=['numpy','scipy','matplotlib'],
      scripts=script_files
     )

