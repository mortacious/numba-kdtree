import pytest
from timeit import default_timer as timer
from numba_kdtree import KDTree
import numpy as np
from scipy.spatial import cKDTree


@pytest.fixture(scope='module')
def data():
    randomstate = np.random.RandomState(seed=0)
    data = randomstate.random(3000000).reshape(-1, 3).astype(np.float64)
    return data


@pytest.fixture(scope='module')
def kd_tree(data):
    leafsize = 10
    kdtree = KDTree(data, leafsize=10, balanced=False, compact=False)
    return kdtree


@pytest.fixture(scope='module')
def scipy_tree(data):
    scipy_tree = cKDTree(data, leafsize=10, balanced_tree=False, compact_nodes=False)
    return scipy_tree


def test_kdtree_build(data):
    print()  # better output
    leafsize=10
    tic = timer()
    kd_tree = KDTree(data, leafsize=leafsize)
    print("kdtree build time with compilation", timer()-tic, "s")
    tic = timer()
    kd_tree = KDTree(data, leafsize=leafsize)
    print("kdtree build time without compilation", timer()-tic, "s")
    assert kd_tree.built()
    tic = timer()
    scipy_tree = cKDTree(data, leafsize=leafsize, balanced_tree=False, compact_nodes=False)
    print("scipy tree build time", timer()-tic)
    assert kd_tree.size == scipy_tree.size, "invalid_sizes kdtree: {}, scipy: {}".format(kd_tree.size, scipy_tree.size)


def test_kdtree_query(data, kd_tree, scipy_tree):
    print()  # better output
    dd, ii = kd_tree.query(data[:1], k=30)  # pre-compile
    tic = timer()
    dd, ii = kd_tree.query(data, k=30)
    print("kdtree time", timer() - tic)

    tic_scipy = timer()
    dd, ii_scipy = scipy_tree.query(data, k=30, n_jobs=1)
    print("scipy time", timer() - tic_scipy)
    assert np.all(ii == ii_scipy)


def test_kdtree_query_parallel(data, kd_tree, scipy_tree):
    print()  # better output
    dd, ii = kd_tree.query(data[:1], k=30, n_jobs=-1)  # pre-compile
    tic = timer()
    dd, ii = kd_tree.query(data, k=30, n_jobs=-1)
    print("time without compilation", timer() - tic)

    tic_scipy = timer()
    dd, ii_scipy = scipy_tree.query(data, k=30, n_jobs=-1)
    print("scipy time", timer() - tic_scipy)
    assert np.all(ii == ii_scipy)


def test_kdtree_query_radius(data, kd_tree, scipy_tree):
    print()  # better output
    ii = kd_tree.query_radius(data[0], r=0.01, return_sorted=True)
    tic = timer()
    ii = kd_tree.query_radius(data[:100], r=0.5, return_sorted=True)
    print("kdtree time", timer() - tic)

    tic_scipy = timer()
    ii_scipy = scipy_tree.query_ball_point(data[:100], r=0.5, return_sorted=True, n_jobs=1)
    toc_scipy = timer() - tic_scipy
    print("scipy time",  toc_scipy)
    for i in range(len(ii)):
        assert np.all(ii[i] == ii_scipy[i]), "Not equal for i={}".format(i)


def test_kdtree_query_radius_parallel(data, kd_tree, scipy_tree):
    print()  # better output
    ii = kd_tree.query_radius(data[0], r=0.01, return_sorted=True, n_jobs=-1)
    tic = timer()
    ii = kd_tree.query_radius(data[:100], r=0.5, return_sorted=True, n_jobs=-1)
    print("kdtree time", timer() - tic)

    tic_scipy = timer()
    ii_scipy = scipy_tree.query_ball_point(data[:100], r=0.5, return_sorted=True, n_jobs=-1)
    toc_scipy = timer() - tic_scipy
    print("scipy time",  toc_scipy)
    for i in range(len(ii)):
        assert np.all(ii[i] == ii_scipy[i]), "Not equal for i={}".format(i)


def test_argument_conversion(data, kd_tree):
    _, ii_test = kd_tree.query(data[0], k=10)

    _, ii = kd_tree.query([data[0]], k=10)
    assert np.all(ii == ii_test)

    _, ii_test = kd_tree.query(np.array([0,0,0]), k=10)
    _, ii2 = kd_tree.query([0, 0, 0], k=10)
    assert np.all(ii2 == ii_test)