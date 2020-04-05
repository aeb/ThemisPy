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
              'sphinx.ext.napoleon']

#source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext'} #, '.md': 'markdown'}


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
html_theme_default = 'bizstyle'
html_theme_path_default = []


try:
    
    import rtcat_sphinx_theme
    html_theme = "rtcat_sphinx_theme"
    html_theme_path = [rtcat_sphinx_theme.get_html_theme_path()]

    #html_theme = 'solar_theme'
    #import solar_theme
    #html_theme_path = [solar_theme.theme_path]
    
except ModuleNotFoundError :
    html_theme = html_theme_default
    html_theme_path = []
    print("WARNING: Could not import rtcat_sphix_theme.  Will used default.")


#html_theme = html_theme_default
#html_theme_path = []
#print("WARNING: Could not import rtcat_sphix_theme.  Will used default.")
    

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docs/_static']




# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/': None}
intersphinx_mapping = {'python': ('http://docs.python.org', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('http://matplotlib.sourceforge.net/', None)}

