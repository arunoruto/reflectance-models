import os
import sys
import datetime

# Make the refmod package importable
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'refmod'
author = 'Mirza Arnaut' # From pyproject.toml
copyright = f"{datetime.datetime.now().year}, {author}"
version = '0.1.0' # From pyproject.toml
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx_rtd_theme',        # Read the Docs theme
    'sphinxcontrib.bibtex',    # For BibTeX citations
    'myst_parser',             # For Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for MyST Parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # Enables ::: fences for directives
    "dollarmath",   # Enables $...$ and $$...$$ for math
    "linkify",      # Automatically finds and links URLs
    "smartquotes",
    "replacements",
]
myst_heading_anchors = 3 # Auto-generate header anchors up to level 3

# -- Options for Napoleon ----------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False # Typically False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True # For [CiteKey] in docstrings

# -- Options for Autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': False, # Usually False, set to True if you want to show members without docstrings
    'private-members': False,
    'special-members': '__init__', # Comma-separated list of special members to document (e.g., '__init__')
    'show-inheritance': True,
    'member-order': 'bysource', # 'alphabetical', 'groupwise', or 'bysource'
}
autodoc_typehints = "signature" # 'signature', 'description', 'none'
# To prevent type hint resolution issues with some complex types or forward refs:
# autodoc_typehints_format = 'short' # Use short names for types (e.g. ndarray instead of numpy.ndarray)
# import typing
# typing.TYPE_CHECKING = True # Helps with forward references if used in your code

# -- Options for Intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for BibTeX ------------------------------------------------------
bibtex_bibfiles = ['library.bib']
bibtex_default_style = 'unsrt' # Matches lumafit
bibtex_reference_style = 'author_year' # Matches lumafit, for :cite:t: style

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] # Directory for static files (custom CSS, images)
# html_logo = "_static/logo.png" # Example: if you add a logo
# html_favicon = "_static/favicon.ico" # Example: if you add a favicon

# If you have custom CSS in _static:
# html_css_files = [
#     'css/custom.css',
# ]

# -- Options for sphinx_rtd_theme ------------------------------------------
# html_theme_options = {
#     'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your GA UI
#     'analytics_anonymize_ip': False,
#     'logo_only': False,
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     'vcs_pageview_mode': '',
#     'style_nav_header_background': 'white',
#     # Toc options
#     'collapse_navigation': True,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }
