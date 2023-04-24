import pytest
from numba_kdtree import KDTree
import numpy as np
from scipy.spatial import cKDTree
import numba as nb
from timeit import timeit


@pytest.fixture(scope='module')
def data():
    from .util import load_standford_bunny
    return load_standford_bunny(dtype=np.float32)


@pytest.fixture(scope='module')
def kdtree(data):
    kd_tree = KDTree(data, leafsize=10, balanced=False, compact=False)
    return kd_tree


@pytest.fixture(scope='module')
def scipy_kdtree(data):
    scipy_tree = cKDTree(data, leafsize=10, balanced_tree=False, compact_nodes=False)
    return scipy_tree


def _test_kdtree_build(data, **kwargs):
    kd_tree = KDTree(data, **kwargs)
    assert kd_tree.built()
    return kd_tree


def test_kdtree_build(data):
    kdtree_f = _test_kdtree_build(data, leafsize=10, balanced=False, compact=False)

    assert kdtree_f.size == 13143
    data_double = data.astype(np.float64)
    kdtree_d = _test_kdtree_build(data_double, leafsize=10, balanced=False, compact=False)

    scipy_tree = cKDTree(data, leafsize=10, balanced_tree=False, compact_nodes=False)
    assert kdtree_f.size == kdtree_d.size == scipy_tree.size, "invalid_sizes kdtree"

    num_executions = 5
    runtime_kdtree_f = timeit(lambda: _test_kdtree_build(data,
                                                         leafsize=10, balanced=False, compact=False),
                              number=num_executions) / num_executions

    runtime_kdtree_d = timeit(lambda: _test_kdtree_build(data_double,
                                                         leafsize=10, balanced=False, compact=False),
                              number=num_executions) / num_executions
    runtime_scipy = timeit(lambda: cKDTree(data,
                                           leafsize=10, balanced_tree=False, compact_nodes=False),
                           number=num_executions) / num_executions

    print("\nbuild time:\nscipy:", runtime_scipy, "\nkdtree(float):",
          runtime_kdtree_f, "\nkdtree(double):", runtime_kdtree_d)


def test_kdtree_query(data, kdtree, scipy_kdtree):
    k = 10
    dd, ii, nn = kdtree.query(data[:1], k=k)  # pre-compile

    # query the nearest neighbors of each input point in a single thread
    dd, ii, nn = kdtree.query(data, k=k)
    dd_scipy, ii_scipy = scipy_kdtree.query(data, k=k, workers=1)

    assert np.all(ii == ii_scipy)
    assert np.allclose(dd, dd_scipy)

    num_executions = 5
    k_benchmark = 100
    runtime_kdtree_query = timeit(lambda: kdtree.query(data, k=k_benchmark),
                              number=num_executions) / num_executions
    runtime_scipy_query = timeit(lambda: scipy_kdtree.query(data, k=k_benchmark, workers=1),
                              number=num_executions) / num_executions
    print("\nquery time(single thread):\nscipy:", runtime_scipy_query, "\nkdtree(float):",
          runtime_kdtree_query)


def test_kdtree_query_parallel(data, kdtree, scipy_kdtree):
    k = 10
    dd, ii, nn = kdtree.query(data[:1], k=k, workers=-1)  # pre-compile

    # query using all available cpu cores
    dd, ii, nn = kdtree.query(data, k=k, workers=-1)  # pre-compile
    dd_scipy, ii_scipy = scipy_kdtree.query(data, k=k, workers=-1)

    assert np.all(ii == ii_scipy)
    assert np.allclose(dd, dd_scipy)

    num_executions = 5
    k_benchmark = 100
    runtime_kdtree_query = timeit(lambda: kdtree.query(data, k=k_benchmark, workers=-1),
                              number=num_executions) / num_executions
    runtime_scipy_query = timeit(lambda: scipy_kdtree.query(data, k=k_benchmark, workers=-1),
                              number=num_executions) / num_executions
    print("\nquery time(multi thread):\nscipy:", runtime_scipy_query, "\nkdtree(float):",
          runtime_kdtree_query)


def test_kdtree_query_max_radius(data, kdtree, scipy_kdtree):
    k = 100
    upper_bound = 0.0005

    dd, ii, nn = kdtree.query(data[:1], k=k, distance_upper_bound=upper_bound)  # pre-compile

    # query the nearest neighbors of each input point in a single thread
    dd, ii, nn = kdtree.query(data, k=k, distance_upper_bound=upper_bound)
    dd_scipy, ii_scipy = scipy_kdtree.query(data, k=k, distance_upper_bound=upper_bound, workers=1)
    assert np.any(nn != k)

    invalid = ii < 0
    assert np.all(ii_scipy[invalid] == data.shape[0])
    assert np.all(ii[~invalid] == ii_scipy[~invalid])

    num_executions = 5
    k_benchmark = 100
    runtime_kdtree_query = timeit(lambda: kdtree.query(data, k=k_benchmark, distance_upper_bound=upper_bound),
                              number=num_executions) / num_executions
    runtime_scipy_query = timeit(lambda: scipy_kdtree.query(data, k=k_benchmark, workers=1,
                                                            distance_upper_bound=upper_bound),
                              number=num_executions) / num_executions
    print("\nquery time(single threadm, max radius):\nscipy:", runtime_scipy_query, "\nkdtree(float):",
          runtime_kdtree_query)


def test_kdtree_query_radius(data, kdtree, scipy_kdtree):
    # pre compile
    ii = kdtree.query_radius(data[:1], r=0.01, return_sorted=True)

    ii = kdtree.query_radius(data[:100], r=0.01, return_sorted=True)
    ii_scipy = scipy_kdtree.query_ball_point(data[:100], r=0.01, return_sorted=True, workers=1)

    for i in range(len(ii)):
        assert len(ii[i]) == len(ii_scipy[i])
        assert np.all(ii[i] == ii_scipy[i]), "Not equal for i={}".format(i)

    num_executions = 5
    r_benchmark = 0.1
    runtime_kdtree_query = timeit(lambda: kdtree.query_radius(data[:500], r=r_benchmark, return_sorted=True),
                              number=num_executions) / num_executions
    runtime_scipy_query = timeit(lambda: scipy_kdtree.query_ball_point(data[:500], r=r_benchmark,
                                                                       return_sorted=True, workers=1),
                              number=num_executions) / num_executions
    print("\nradius query time(single thread):\nscipy:", runtime_scipy_query, "\nkdtree(float):",
          runtime_kdtree_query)


def test_kdtree_query_radius_parallel(data, kdtree, scipy_kdtree):
    ii = kdtree.query_radius(data[:1], r=0.01, return_sorted=True, workers=-1)

    ii = kdtree.query_radius(data[:100], r=0.01, return_sorted=True, workers=-1)
    ii_scipy = scipy_kdtree.query_ball_point(data[:100], r=0.01, return_sorted=True, workers=-1)

    for i in range(len(ii)):
        assert len(ii[i]) == len(ii_scipy[i])
        assert np.all(ii[i] == ii_scipy[i]), "Not equal for i={}".format(i)

    num_executions = 5
    r_benchmark = 0.1
    runtime_kdtree_query = timeit(lambda: kdtree.query_radius(data[:500], r=r_benchmark, return_sorted=True, workers=-1),
                              number=num_executions) / num_executions
    runtime_scipy_query = timeit(lambda: scipy_kdtree.query_ball_point(data[:500], r=r_benchmark,
                                                                       return_sorted=True, workers=-1),
                              number=num_executions) / num_executions
    print("\nradius query time(multi thread):\nscipy:", runtime_scipy_query, "\nkdtree(float):",
          runtime_kdtree_query)


def test_argument_conversion(data, kdtree):
    # conversion of scalar arguments
    _, ii_test, _ = kdtree.query(data[0], k=10)

    # conversion of lists
    _, ii, _ = kdtree.query([data[0]], k=10)
    assert np.all(ii == ii_test)

    # conversion of numba lists
    _, ii, _ = kdtree.query(nb.typed.List([data[0]]), k=10)
    assert np.all(ii == ii_test)

    # conversion of manually specified arrays
    _, ii_test, _ = kdtree.query(np.array([0, 0, 0]), k=10)
    _, ii2, _ = kdtree.query([0, 0, 0], k=10)
    assert np.all(ii2 == ii_test)


def test_use_in_numba_function(data, kdtree):
    @nb.njit(nogil=True, fastmath=True)
    def query_numba(data, kdtree, k):
        ii = np.empty((data.shape[0], k), dtype=np.int64)
        for i in range(data.shape[0]):
            d, ii[i], n = kdtree.query(data[i], k=k)

        return ii

    # pre compile
    ii = query_numba(data, kdtree, 10)
    dd, ii, nn = kdtree.query(data[:1], k=10)  # pre-compile

    num_executions = 5
    k_benchmark = 100

    # run times should be very similar here
    runtime_kdtree_query = timeit(lambda: kdtree.query(data, k=k_benchmark),
                              number=num_executions) / num_executions
    runtime_numba_kdtree_query = timeit(lambda: query_numba(data, kdtree, k_benchmark),
                              number=num_executions) / num_executions
    print("\nnumba query time(single thread):\nkdtree(float):",
          runtime_kdtree_query, "\nkdtree (numba):", runtime_numba_kdtree_query)


def test_use_in_numba_function_parallel(data, kdtree):
    @nb.njit(nogil=True, fastmath=True, parallel=True)
    def query_numba_parallel(data, kdtree, k):
        ii = np.empty((data.shape[0], k), dtype=np.int64)
        for i in nb.prange(data.shape[0]):
            d, ii[i], n = kdtree.query(data[i], k=k)

        return ii

    # pre compile
    ii = query_numba_parallel(data, kdtree, 10)
    dd, ii, nn = kdtree.query(data[:1], k=10, workers=-1)  # pre-compile

    num_executions = 5
    k_benchmark = 100

    # run times should be very similar here
    runtime_kdtree_query = timeit(lambda: kdtree.query(data, k=k_benchmark, workers=-1),
                              number=num_executions) / num_executions
    runtime_numba_kdtree_query = timeit(lambda: query_numba_parallel(data, kdtree, k_benchmark),
                              number=num_executions) / num_executions
    print("\nnumba query time(multi thread):\nkdtree(float):",
          runtime_kdtree_query, "\nkdtree (numba):", runtime_numba_kdtree_query)


def test_construct_in_numba_function(data):
    @nb.njit(nogil=True, fastmath=True)
    def construct_kdtree_in_numba(data):
        kdtree = KDTree(data, leafsize=10)
        return kdtree

    num_executions = 10
    runtime_kdtree_numba = timeit(lambda: construct_kdtree_in_numba(data),
                              number=num_executions) / num_executions

    runtime_kdtree_python = timeit(lambda: KDTree(data), number=num_executions) / num_executions

    print("\nbuild time:\nnumba:",
          runtime_kdtree_numba, "\npython:", runtime_kdtree_python)
