import numpy as np
from numba_kdtree import KDTree


data = np.random.random(3_000_000).reshape(-1, 3).astype(np.float32)
kdtree = KDTree(data, leafsize=10)

# query the nearest neighbors of the first 100 points
distances, indices, num_neighbors = kdtree.query(data[:100], k=30)

# query all points in a radius around the first 100 points
indices = kdtree.query_radius(data[:100], r=0.5, return_sorted=True)