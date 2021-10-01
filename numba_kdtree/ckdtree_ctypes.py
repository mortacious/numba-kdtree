from ctypes import c_float, c_double, c_ssize_t, c_int, c_bool, c_void_p, POINTER, cdll
from numba import types
import pathlib


# ctypes interface to the c implementation for use in numba
library_path = pathlib.Path(__file__).parent


class _CKDTree(object):
    pass


ckdtree = _CKDTree

c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)
c_ssize_t_p = POINTER(c_ssize_t)

try:
    _ckdtreelib = cdll.LoadLibrary(str(library_path / "_ckdtree.so"))
except Exception:
    raise ImportError('Cannot load dynamic library.')

_supported_types = {types.float32: 'float',
                    types.float64: 'double'
                    }


def _define_functions(fmt_str):
    for numba_type, c_type in _supported_types.items():
        eval(compile(fmt_str.format(c_type=c_type, numba_type=numba_type.name), '<string>', 'exec'))


ckdtree.init = {}
_define_functions("""
_ckdtreelib.ckdtree_init_{c_type}.restype = c_void_p
_ckdtreelib.ckdtree_init_{c_type}.argtypes = (c_{c_type}_p,
                                           c_ssize_t_p,
                                           c_ssize_t, 
                                           c_ssize_t, 
                                           c_ssize_t, 
                                           c_{c_type}_p, 
                                           c_{c_type}_p)
ckdtree.init[types.{numba_type}] = _ckdtreelib.ckdtree_init_{c_type}
""")

ckdtree.free = {}
_define_functions("""
_ckdtreelib.ckdtree_free_{c_type}.restype  = None
_ckdtreelib.ckdtree_free_{c_type}.argtypes = (c_void_p,)
ckdtree.free[types.{numba_type}] = _ckdtreelib.ckdtree_free_{c_type}
""")

ckdtree.build = {}
_define_functions("""
_ckdtreelib.ckdtree_build_{c_type}.restype  = c_int
_ckdtreelib.ckdtree_build_{c_type}.argtypes = (c_void_p,
                                c_ssize_t, c_ssize_t,
                                c_{c_type}_p, c_{c_type}_p, c_int, c_int)
ckdtree.build[types.{numba_type}] = _ckdtreelib.ckdtree_build_{c_type}
""")

ckdtree.size = {}
_define_functions("""
_ckdtreelib.ckdtree_size_{c_type}.restype  = c_ssize_t
_ckdtreelib.ckdtree_size_{c_type}.argtypes = (c_void_p,)
ckdtree.size[types.{numba_type}] = _ckdtreelib.ckdtree_size_{c_type}
""")

ckdtree.query_knn = {}
_define_functions("""
_ckdtreelib.ckdtree_query_knn_{c_type}.restype  = c_int
_ckdtreelib.ckdtree_query_knn_{c_type}.argtypes = (c_void_p,
                                                   c_{c_type}_p,
                                                   c_ssize_t_p,
                                                   c_{c_type}_p,
                                                   c_ssize_t,
                                                   c_ssize_t,
                                                   c_double,
                                                   c_double,
                                                   c_{c_type}
                                                   )
ckdtree.query_knn[types.{numba_type}] = _ckdtreelib.ckdtree_query_knn_{c_type}
""")


ckdtree.query_radius = {}
_define_functions("""
_ckdtreelib.ckdtree_query_radius_{c_type}.restype  = c_void_p
_ckdtreelib.ckdtree_query_radius_{c_type}.argtypes = (c_void_p,
                                                   c_{c_type}_p,
                                                   c_ssize_t,
                                                   c_{c_type},
                                                   c_double,
                                                   c_double,
                                                   c_bool,
                                                   c_bool
                                                   )
ckdtree.query_radius[types.{numba_type}] = _ckdtreelib.ckdtree_query_radius_{c_type}
""")


_ckdtreelib.radius_result_set_get_size.restype = c_ssize_t
_ckdtreelib.radius_result_set_get_size.argtypes = (c_void_p,)
ckdtree.radius_result_set_get_size = _ckdtreelib.radius_result_set_get_size

_ckdtreelib.radius_result_set_copy_and_free.restype = None
_ckdtreelib.radius_result_set_copy_and_free.argtypes = (c_void_p, c_ssize_t_p)
ckdtree.radius_result_set_copy_and_free = _ckdtreelib.radius_result_set_copy_and_free
