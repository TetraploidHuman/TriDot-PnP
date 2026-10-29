"""B0-Classic baseline: HSV blob detection + geometry + OpenCV PnP.

Bytecode-only package: the original sources were lost.  ``research.pyc_import``
serves the ``.pyc`` modules from ``__pycache__`` so that ``from
research.baseline_b0.detector import B0Detector`` keeps working.
"""

from research.pyc_import import install as _install

_install()

__version__ = "1.0.0"
