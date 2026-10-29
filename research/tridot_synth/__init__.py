"""TriDot-Pose synthetic video & annotation generator.

Bytecode-only package: the original sources were lost.  ``research.pyc_import``
serves the ``.pyc`` modules from ``__pycache__`` so the generator stays usable
(e.g. ``python -m research.tridot_synth.generate --help``).
"""

from research.pyc_import import install as _install

_install()

__version__ = "1.1.0"
