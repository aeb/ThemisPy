# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./themispy/'))
sys.path.insert(0, os.path.abspath('./scripts/'))
print("Documentation generation searching",sys.path)

# -- Project information -----------------------------------------------------

project = 'ThemisPy'
copyright = '2020, Themis Development Team'
author = 'Themis Development Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ['sphinx.ext.autodoc']
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinxarg.ext']

#source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext'} #, '.md': 'markdown'}
#source_suffix = {'.rst': 'restructuredtext'} #, '.txt': 'restructuredtext'} #, '.md': 'markdown'}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['docs/_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

import versioneer
version = versioneer.get_version()
release = version
show_authors = True

master_doc = 'docs/src/index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "rtcat_sphinx_theme"
html_theme_path = ["./docs/_themes"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docs/_static']




# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/': None}
intersphinx_mapping = {'python': ('https://docs.python.org', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'ehtim': ('https://achael.github.io/eht-imaging/', None),
                       'Themis': ('https://perimeterinstitute.github.io/Themis/html/', None)
#                       'Themis': ('/Users/abroderick/Research/Themis/Themis/docs/html',None)
                   }
