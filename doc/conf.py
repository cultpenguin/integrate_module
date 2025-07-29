# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the parent directory (where the integrate package is located)
sys.path.insert(0, os.path.abspath('..'))

# Try to import integrate to ensure it's available
try:
    import integrate
    print(f"Successfully imported integrate from {integrate.__file__}")
except ImportError as e:
    print(f"Failed to import integrate: {e}")

# Set up the environment for Sphinx autodoc
autodoc_mock_imports = []

# Mock problematic imports if needed
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

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

# GitHub Pages compatibility
html_baseurl = 'https://cultpenguin.github.io/integrate_mockup/'
html_copy_source = False
html_show_sourcelink = False

# Ensure proper handling of notebooks and static files
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'  # Don't execute notebooks during build

def setup(app):
    """Custom setup function to add .nojekyll file to output."""
    import os
    
    def add_nojekyll_file(app, exception):
        if exception is None:  # Build was successful
            nojekyll_path = os.path.join(app.outdir, '.nojekyll')
            with open(nojekyll_path, 'w') as f:
                f.write('')
    
    app.connect('build-finished', add_nojekyll_file)
