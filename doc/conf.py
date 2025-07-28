# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#sys.path.insert(0, os.path.abspath('../src/python/'))
#sys.path.insert(0, os.path.abspath('../src/python/integrate_python/'))
#sys.path.insert(0, os.path.abspath('../../integrate_module'))
sys.path.insert(0, os.path.abspath('../../integrate_module/integrate/'))
sys.path.insert(0, os.path.abspath('../integrate_module/integrate/'))

project = 'INTEGRATE'
copyright = '2023,2024,2025 INTEGRATE WORKING GROUP'
author = 'INTEGRATE WORKING GROUP'

# The short X.Y version
version = '0.26'    
# The full version, including alpha/beta/rc tags
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = ['myst_parser']
#extensions = ['nbsphinx']
#extensions = ['myst_nb']
extensions = [
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',

]
autosummary_generate = True

# Napoleon settings (for allowing using Google and NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
