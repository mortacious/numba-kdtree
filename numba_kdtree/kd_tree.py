import numba as nb
import numpy as np
import time

from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method
from .ckdtree_ctypes import ckdtree as ckdtree_ct


__all__ = ["KDTree"]

FASTMATH = True
DEBUG = False

INT_TYPE = np.int64
INT_TYPE_T = nb.int64

DIM_TYPE = np.uint32
DIM_TYPE_T = nb.uint32

FLOAT_TYPE = np.float32
FLOAT_TYPE_T = nb.float32

BOOL_TYPE = np.uint8
BOOL_TYPE_T = nb.uint8

IntArray = np.ndarray
DimIndexArray = np.ndarray
DataArray = np.ndarray
BoolArray = np.ndarray

NUMBA_THREADS = nb.config.NUMBA_NUM_THREADS


@nb.njit(nogil=True, inline="always", debug=DEBUG)
def arange(length, dtype=INT_TYPE):
    """Simple `np.arange` implementation without start/step."""
    out = np.empty((length,), dtype=dtype)
    for i in range(length):  # pylint: disable=not-an-iterable
        out[i] = i
    return out


@structref.register
class KDTreeType(types.StructRef):
    def preprocess_fields(self, fields):
        # We don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class _KDTree(structref.StructRefProxy):
    def __new__(cls, ckdtree, root_bbox, data, idx):
        ckdtree = nb.types.voidptr()
        return structref.StructRefProxy.__new__(cls,
                                                ckdtree,
                                                root_bbox,
                                                data,
                                                idx)

    @property
    def root_bbox(self):
        return _KDTree_get_root_bbox(self)

    @property
    def data(self):
        return _KDTree_get_data(self)

    @property
    def idx(self):
        return _KDTree_get_idx(self)

    @property
    def size(self):
        return _KDTree_get_size(self)

    def built(self):
        return _KDTree_built(self)

    def __del__(self):
        _KDTree_free(self)

    def free_index(self):
        _KDTree_free(self)

    def query(self, X, k=1, p=2.0, eps=0.0, distance_upper_bound=np.inf, n_jobs=1):
        return _KDTree_query(self, X, k, p, eps, distance_upper_bound, n_jobs)

    def query_radius(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False, n_jobs=-1):
        return _KDTree_query_radius(self, X, r, p, eps, return_sorted, return_length, n_jobs)


structref.define_proxy(_KDTree, KDTreeType,
                       ["ckdtree", "root_bbox", "data", "idx"])


@nb.njit()
def _KDTree_get_root_bbox(self):
    return self.root_bbox


@nb.njit()
def _KDTree_get_data(self):
    return self.data


@nb.njit()
def _KDTree_get_idx(self):
    return self.idx


@nb.njit()
def _KDTree_get_size(self):
    return self._size()


@nb.njit()
def _KDTree_built(self):
    return self.ckdtree != 0


@nb.njit()
def _KDTree_free(self):
    self.free_index()


@nb.njit()
def _KDTree_query(self, X, k=1, p=2, eps=0.0, distance_upper_bound=np.inf, n_jobs=1):
    if n_jobs == 1:
        return self.query(X, k=k, p=p, eps=eps, distance_upper_bound=distance_upper_bound)
    else:
        return self.query_parallel(X, k=k, p=p, eps=eps, distance_upper_bound=distance_upper_bound, n_jobs=n_jobs)


@nb.njit()
def _KDTree_query_radius(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False, n_jobs=1):
    if n_jobs == 1:
        return self.query_radius(X, r, p, eps, return_sorted, return_length)
    else:
        return self.query_radius_parallel(X, r, p, eps, return_sorted, return_length, n_jobs=n_jobs)


@overload_method(KDTreeType, "_size")
def _ol_size(self):
    dtype = self.field_dict['data'].dtype
    if dtype != nb.types.float32:
        dtype = nb.types.float64

    func_size = ckdtree_ct.size[dtype]

    def _size_impl(self):
        return func_size(self.ckdtree)

    return _size_impl


@overload_method(KDTreeType, "free_index")
def _ol_free_index(self):
    dtype = self.field_dict['data'].dtype
    if dtype != nb.types.float32:
        dtype = nb.types.float64

    func_free = ckdtree_ct.free[dtype]

    def _free_index_impl(self):
        func_free(self.ckdtree)

    return _free_index_impl


@overload_method(KDTreeType, "build_index", jit_options={"nogil": True, "debug": DEBUG, "fastmath": FASTMATH})
def _ol_build_index(self, leafsize, balanced=False, compact=False):
    # choose the appropriate methods based on the data type
    dtype = self.field_dict['data'].dtype
    if dtype != nb.types.float32:
        dtype = nb.types.float64

    func_init = ckdtree_ct.init[dtype]
    func_build = ckdtree_ct.build[dtype]
    func_free = ckdtree_ct.free[dtype]

    def _build_index_impl(self, leafsize, balanced=False, compact=False):
        n_data, n_features = self.data.shape
        if self.ckdtree != 0:
            func_free(self.ckdtree)
        self.ckdtree = func_init(self.data.ctypes, self.idx.ctypes, n_data, n_features, leafsize, self.root_bbox[0].ctypes, self.root_bbox[1].ctypes)
        compact_ = 1 if compact else 0
        balanced_ = 1 if balanced else 0
        func_build(self.ckdtree, 0, n_data, self.root_bbox[0].ctypes, self.root_bbox[1].ctypes, balanced_, compact_)

    return _build_index_impl


@overload_method(KDTreeType, "query", jit_options={"nogil": True, "debug": DEBUG, "fastmath": FASTMATH, 'parallel': False})
def _ol_query(self, X, k=1, p=2, eps=0.0, distance_upper_bound=np.inf):
    # choose the appropriate methods based on the data type
    dtype = self.field_dict['data'].dtype
    if dtype == nb.types.float32:
        dtype_npy = np.float32
    else:
        dtype = nb.types.float64
        dtype_npy = np.float64

    func_query_knn = ckdtree_ct.query_knn[dtype]

    def _query_impl(self, X, k=1, p=2, eps=0.0, distance_upper_bound=np.inf):
        n_features = self.data.shape[1]
        xx = np.ascontiguousarray(X.reshape(-1, n_features)).astype(dtype_npy)
        n_queries = xx.shape[0]
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")

        dd = np.empty((n_queries, k), dtype=dtype_npy)
        ii = np.full((n_queries, k), fill_value=-1, dtype=INT_TYPE)
        for i in range(n_queries):
            func_query_knn(self.ckdtree, dd[i].ctypes, ii[i].ctypes,
                      xx[i].ctypes, 1, k, eps, p, distance_upper_bound)
        return dd, ii

    return _query_impl


@overload_method(KDTreeType, "query_parallel", jit_options={"nogil": True, "debug": DEBUG, "fastmath": FASTMATH, 'parallel': True})
def _ol_query_parallel(self, X, k=1, p=2, eps=0.0, distance_upper_bound=np.inf, n_jobs=-1):
    dtype = self.field_dict['data'].dtype
    if dtype == nb.types.float32:
        dtype_npy = np.float32
    else:
        dtype = nb.types.float64
        dtype_npy = np.float64

    func_query_knn = ckdtree_ct.query_knn[dtype]

    def _query_parallel_impl(self, X, k=1, p=2, eps=0.0, distance_upper_bound=np.inf, n_jobs=-1):
        numba_threads_prev = nb.get_num_threads()
        if n_jobs < 0:
            nb.set_num_threads(NUMBA_THREADS)
        else:
            nb.set_num_threads(n_jobs)

        n_features = self.data.shape[1]
        xx = np.ascontiguousarray(X.reshape(-1, n_features))
        n_queries = xx.shape[0]

        dd = np.empty((n_queries, k), dtype=dtype_npy)
        ii = np.full((n_queries, k), fill_value=-1, dtype=INT_TYPE)

        for i in nb.prange(n_queries):
            func_query_knn(self.ckdtree, dd[i].ctypes, ii[i].ctypes,
                      xx[i].ctypes, 1, k, eps, p, distance_upper_bound)
        nb.set_num_threads(numba_threads_prev)
        return dd, ii

    return _query_parallel_impl


@overload_method(KDTreeType, "query_radius", jit_options={"nogil": True, "debug": DEBUG, "fastmath": FASTMATH, 'parallel': False})
def _ol_query_radius(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False):
    # choose the appropriate methods based on the data type
    dtype = self.field_dict['data'].dtype
    if dtype == nb.types.float32:
        dtype_npy = np.float32
    else:
        dtype = nb.types.float64
        dtype_npy = np.float64

    func_query_knn = ckdtree_ct.query_radius[dtype]
    radius_result_set_get_size = ckdtree_ct.radius_result_set_get_size
    radius_result_set_copy_and_free = ckdtree_ct.radius_result_set_copy_and_free

    result_array_type = types.int64[:]

    # noinspection PyShadowingNames
    def _query_radius_impl(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False):
        n_features = self.data.shape[1]
        xx = np.ascontiguousarray(X.reshape(-1, n_features)).astype(dtype_npy)  # only one query for now!
        n_queries = xx.shape[0]

        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")

        # prepare result list
        results_list = nb.typed.List.empty_list(item_type=result_array_type, allocated=n_queries)
        results_list.extend([np.empty(0, dtype=np.int64) for i in range(n_queries)])

        for i in range(n_queries):
            result_set = func_query_knn(self.ckdtree, xx[i].ctypes, 1, r, eps, p, return_length, return_sorted)
            # copy the result set into a separate buffer owned by python
            num_results = radius_result_set_get_size(result_set)
            results = np.empty(num_results, dtype=np.int64)
            radius_result_set_copy_and_free(result_set, results.ctypes)
            results_list[i] = results

        return results_list

    return _query_radius_impl


@overload_method(KDTreeType, "query_radius_parallel", jit_options={"nogil": True, "debug": DEBUG, "fastmath": FASTMATH, 'parallel': True})
def _ol_query_radius_parallel(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False, n_jobs=-1):
    # choose the appropriate methods based on the data type
    dtype = self.field_dict['data'].dtype
    if dtype == nb.types.float32:
        dtype_npy = np.float32
    else:
        dtype = nb.types.float64
        dtype_npy = np.float64

    func_query_knn = ckdtree_ct.query_radius[dtype]
    radius_result_set_get_size = ckdtree_ct.radius_result_set_get_size
    radius_result_set_copy_and_free = ckdtree_ct.radius_result_set_copy_and_free

    result_array_type = types.int64[:]

    # noinspection PyShadowingNames
    def _query_radius_parallel_impl(self, X, r, p=2.0, eps=0.0, return_sorted=False, return_length=False, n_jobs=-1):
        numba_threads_prev = nb.get_num_threads()
        if n_jobs < 0:
            nb.set_num_threads(NUMBA_THREADS)
        else:
            nb.set_num_threads(n_jobs)

        n_features = self.data.shape[1]
        xx = np.ascontiguousarray(X.reshape(-1, n_features)).astype(dtype_npy)  # only one query for now!
        n_queries = xx.shape[0]

        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")

        # prepare result list
        results_list = nb.typed.List.empty_list(item_type=result_array_type, allocated=n_queries)
        results_list.extend([np.empty(0, dtype=np.int64) for i in range(n_queries)])

        for i in nb.prange(n_queries):
            result_set = func_query_knn(self.ckdtree, xx[i].ctypes, 1, r, eps, p, return_length, return_sorted)
            # copy the result set into a separate buffer owned by python
            num_results = radius_result_set_get_size(result_set)
            results = np.empty(num_results, dtype=np.int64)
            radius_result_set_copy_and_free(result_set, results.ctypes)
            results_list[i] = results
        nb.set_num_threads(numba_threads_prev)

        return results_list

    return _query_radius_parallel_impl


@nb.njit(nogil=True)
def _make_kdtree(data, root_bbox, idx, leafsize=10, balanced=False, compact=False):
    # create the transparent underlying c object by calling the function appropriate to the data dtype
    ckdtree = np.uint64(0) # leave the c object empty for now
    kdtree = _KDTree(ckdtree, root_bbox, data, idx)
    kdtree.build_index(leafsize, balanced, compact)
    return kdtree


# constructor method
def KDTree(data: DataArray, leafsize: int = 10, compact: bool = False, balanced: bool = False, root_bbox=None):
    if data.dtype == np.float32:
        conv_dtype = np.float32
    else:
        conv_dtype = np.float64

    data = np.ascontiguousarray(data).astype(conv_dtype)
    n_data, n_features = data.shape

    if root_bbox is None:
        # compute the bounding box
        mins = np.amin(data, axis=0) if n_data > 0 else np.zeros(n_features, dtype=conv_dtype)
        maxes = np.amax(data, axis=0) if n_data > 0 else np.zeros(n_features, dtype=conv_dtype)
        root_bbox = np.vstack((mins, maxes))
    root_bbox = np.ascontiguousarray(root_bbox, dtype=conv_dtype)

    idx = np.arange(n_data, dtype=INT_TYPE)

    tic = time.time()
    kdtree = _make_kdtree(data, root_bbox, idx, leafsize, compact, balanced)
    return kdtree

