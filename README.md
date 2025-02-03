# Numba-Kdtree - a fast KDTree implementation for numba 

`numba-kdtree` provides a fast KDTree implementation for `numba` CPU functions based on [scipys `ckdtree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html). 

The provided `KDTree` class is usable from both python and numba nopython functions calling directly into C code for performance critical sections utilizing a ctypes-like interface.

Once the `KDTree` class is compiled by `numba`, it is just as fast as the original scipy version.

Note: Currently only a basic subset of the original `ckdtree` interface is implemented.

## Installation

### Using pip
```
pip install numba-kdtree
```

### From source
```
git clone https://github.com/mortacious/numba-kdtree.git
cd numba-kdtree
python -m pip install
```

## Usage

```python
import numpy as np
from numba_kdtree import KDTree
data = np.random.random(3_000_000).reshape(-1, 3)
kdtree = KDTree(data, leafsize=10)

# compute the nearest neighbors of the first 100 points
distances, indices = kdtree.query(data[:100], k=30)

# compute all points in a 0.5 radius around the first 100 points and return it's sorted indices
indices = kdtree.query_radius(data[:100], r=0.5, return_sorted=True)
```

The `KDTree` can be freely passed into a `numba` nopython function or constructed directly within it:

```python
import numpy as np
from numba import njit
from numba_kdtree import KDTree

def numba_function_with_kdtree(kdtree, data):
    for i in range(data.shape[0]):
        distances, indices = kdtree.query(data[i], k=30)
        #<Use the computed neighbors
        
data = np.random.random(3_000_000).reshape(-1, 3)
kdtree = KDTree(data, leafsize=10)

numba_function_with_kdtree(kdtree)
```

```python
import numpy as np
from numba import njit
from numba_kdtree import KDTree

def numba_function_constructing_kdtree(kdtree, data):
    kdtree = KDTree(data, leafsize=10)
    for i in range(data.shape[0]):
        distances, indices = kdtree.query(data[i], k=30)
        #<Use the computed neighbors
        
data = np.random.random(10_000_000).reshape(-1, 10)

numba_function_constructing_kdtree(data)
```

Additionally, the `KDTree` object can also be serialized via pickle:

```python
import pickle

data = np.random.random(3_000_000).reshape(-1, 3)
kdtree = KDTree(data, leafsize=10) 

# pass the tree through pickle
# Note: This also copies the data array preserving it's integrity
serialized_tree = pickle.dumps(kdtree)

# The copied data array is now owned by the restored tree
restored_kdtree = pickle.loads(serialized_tree)

k = 10
# query the old tree
dd, ii, nn = kdtree.query_parallel(data[:100], k=k)

# query the new tree
dd_r, ii_r, nn_r = restored_kdtree.query_parallel(data[:100], k=k)
```

## TODOs

- Implement the full scipy `ckdtree` interface