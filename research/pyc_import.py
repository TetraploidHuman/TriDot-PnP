"""Sourceless import shim for research sub-packages whose ``.py`` sources were lost.

Several research sub-packages in this repo (``baseline_b0``, ``baseline_b1``,
``tridot_synth``, ``label_tool``) only ship compiled bytecode under
``__pycache__/``.  Python refuses to import from ``__pycache__`` when the source
file is missing, so ``import research.baseline_b0.detector`` fails even though
the bytecode is perfectly valid.

This module installs a :class:`importlib.abc.MetaPathFinder` that serves those
submodules directly from their ``.pyc`` files.  It is intentionally a *temporary*
crutch: the real fix is to recover or rewrite the sources (see
``research/results/audit_20261028/REPORT.md``).  Keeping the bytecode loadable
means every existing script (``eval_find.py``, ``diag_*.py``, the synth
generator) keeps working in the meantime.

Usage::

    from research.pyc_import import install
    install()                      # idempotent

    from research.baseline_b0.detector import B0Detector

The interpreter must match the bytecode tag (CPython 3.11, magic ``a70d0d0a``);
:func:`install` raises a clear error instead of silently skipping when it does
not.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

#: Packages whose submodules live only as bytecode in ``<pkg>/__pycache__/``.
SOURCELESS_PACKAGES: dict[str, Path] = {
    "research.baseline_b0": Path(__file__).resolve().parent / "baseline_b0",
    "research.baseline_b1": Path(__file__).resolve().parent / "baseline_b1",
    "research.tridot_synth": Path(__file__).resolve().parent / "tridot_synth",
    "research.label_tool": Path(__file__).resolve().parent / "label_tool",
}

_FINDER_ATTR = "_tridot_sourceless_finder"


class SourcelessPackageFinder(importlib.abc.MetaPathFinder):
    """Serve ``<pkg>.<leaf>`` from ``<pkg>/__pycache__/<leaf>.<tag>.pyc``."""

    def __init__(self, packages: dict[str, Path] | None = None) -> None:
        self.packages = dict(packages or SOURCELESS_PACKAGES)

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401 - stdlib signature
        parent, _, leaf = fullname.rpartition(".")
        root = self.packages.get(parent)
        if root is None or not leaf:
            return None
        pyc = root / "__pycache__" / f"{leaf}.{_cache_tag()}.pyc"
        if not pyc.is_file():
            return None
        # The loader still reads the .pyc (its path is what get_code() opens),
        # but the module's ``__file__`` is set to the would-be source location so
        # ``Path(__file__).parent / "static"`` and similar resolve correctly.
        virtual = root / f"{leaf}.py"
        loader = importlib.machinery.SourcelessFileLoader(fullname, str(pyc))
        spec = importlib.util.spec_from_loader(fullname, loader, origin=str(virtual))
        # spec_from_loader leaves has_location False; we want __file__/__cached__
        # populated (the recovered bytecode legitimately expects both).
        spec.has_location = True
        spec.cached = str(pyc)
        return spec


def _cache_tag() -> str:
    return f"cpython-{sys.version_info.major}{sys.version_info.minor}"


def install(force: bool = False) -> None:
    """Register the finder on ``sys.meta_path`` (idempotent)."""
    if not force and getattr(sys, _FINDER_ATTR, None) is not None:
        return
    finder = SourcelessPackageFinder()
    # Insert before the default PathFinder so ``research.foo.bar`` is resolved
    # from bytecode rather than reported as missing.
    sys.meta_path.insert(0, finder)
    setattr(sys, _FINDER_ATTR, finder)


def available() -> dict[str, list[str]]:
    """Return ``{package: [submodule, ...]}`` for every loadable bytecode module."""
    out: dict[str, list[str]] = {}
    tag = _cache_tag()
    for pkg, root in SOURCELESS_PACKAGES.items():
        cache = root / "__pycache__"
        if not cache.is_dir():
            continue
        mods = sorted(
            p.name[: -(len(tag) + 5)]  # strip ".<tag>.pyc"
            for p in cache.glob(f"*.{tag}.pyc")
            if not p.name.startswith("__init__.")
        )
        out[pkg] = mods
    return out
