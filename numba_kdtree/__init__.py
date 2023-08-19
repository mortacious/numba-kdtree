try:
    from importlib.metadata import version

    __version__ = version("numba_kdtree")
except Exception:  # pragma: no cover # pylint: disable=broad-exception-caught
    try:
        from ._version import __version__
    except ImportError:
        __version__ = '0.0.0'

from .kd_tree import *
