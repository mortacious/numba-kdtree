from llvmlite import binding
from numba import types
import pathlib
import platform

# interface to the c implementation for use in numba

library_path = pathlib.Path(__file__).parent.absolute()

if platform.system() == "Windows":
    library_name = "_ckdtree.lib"
else:
    library_name = "_ckdtree.so"


# a simple class holding all the function pointers required from the c library
class CKDTreeFunctions(object):
    pass


ckdtree = CKDTreeFunctions

# define the pointer types to be used
float_ptr = types.CPointer(types.float32)
double_ptr = types.CPointer(types.double)
ssize_t_ptr = types.CPointer(types.ssize_t)
ckdtree_handle = types.size_t

try:
    binding.load_library_permanently(str(library_path / library_name))
except Exception:
    raise ImportError('Unable to load dynamic library.')


# the data types supported by the c implementation
_supported_types = {types.float32: ("float", float_ptr),
                    types.float64: ("double", double_ptr)
                    }

ckdtree.init = {}
ckdtree.free = {}
ckdtree.build = {}
ckdtree.size = {}
ckdtree.query_knn = {}
ckdtree.query_radius = {}
ckdtree.leafsize = {}
ckdtree.nodesize = {}
ckdtree.copy_tree = {}


for numba_type, (c_type, ptr_type) in _supported_types.items():
    ckdtree.init[numba_type] = types.ExternalFunction(f"ckdtree_init_{c_type}", ckdtree_handle(
        types.voidptr,
        types.ssize_t,
        ptr_type,
        ssize_t_ptr,
        types.ssize_t,
        types.ssize_t,
        types.ssize_t,
        ptr_type,
        ptr_type
    ))

    ckdtree.free[numba_type] = types.ExternalFunction(f"ckdtree_free_{c_type}", types.void(
        ckdtree_handle
    ))

    ckdtree.build[numba_type] = types.ExternalFunction(f"ckdtree_build_{c_type}", types.int32(
        ckdtree_handle,
        types.ssize_t,
        types.ssize_t,
        ptr_type,
        ptr_type,
        types.int32,
        types.int32
    ))

    ckdtree.size[numba_type] = types.ExternalFunction(f"ckdtree_size_{c_type}", types.ssize_t(
        ckdtree_handle
    ))

    ckdtree.query_knn[numba_type] = types.ExternalFunction(f"ckdtree_query_knn_{c_type}", types.int32(
        ckdtree_handle,
        ptr_type,
        ssize_t_ptr,
        ssize_t_ptr,
        ptr_type,
        types.ssize_t,
        types.ssize_t,
        types.double,
        types.double,
        numba_type
    ))

    ckdtree.query_radius[numba_type] = types.ExternalFunction(f"ckdtree_query_radius_{c_type}", types.voidptr(
        ckdtree_handle,
        ptr_type,
        types.ssize_t,
        numba_type,
        types.double,
        types.double,
        types.bool_,
        types.bool_
    ))

    ckdtree.leafsize[numba_type] = types.ExternalFunction(f"leafsize_{c_type}", types.ssize_t(
        ckdtree_handle
    ))

    ckdtree.nodesize[numba_type] = types.ExternalFunction(f"nodesize_{c_type}", types.ssize_t(
        ckdtree_handle
    ))

    ckdtree.copy_tree[numba_type] = types.ExternalFunction(f"copy_tree_{c_type}", types.ssize_t(
        ckdtree_handle,
        types.voidptr
    ))


ckdtree.radius_result_set_get_size = types.ExternalFunction("radius_result_set_get_size", types.ssize_t(
    types.voidptr
))

ckdtree.radius_result_set_copy_and_free = types.ExternalFunction("radius_result_set_copy_and_free", types.void(
    types.voidptr,
    ssize_t_ptr
))       

