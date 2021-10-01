# Numba-kdtree

A simple KD-Tree for numba using a ctypes wrapper around the scipy `ckdtree` implementation. 
The KD-Tree is usable in both python and numba nopython functions.

Once the query functions are compiled by numba, the implementation is just as fast as the original scipy version.

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
python setup.py install
```

## Usage

```python
import numpy as np
from numba_kdtree import KDTree
data = np.random.random(3_000_000).reshape(-1, 3)
kdtree = KDTree(data, leafsize=10)

# query the nearest neighbors of the first 100 points
distances, indices = kdtree.query(data[:100], k=30)

# query all points in a radius around the first 100 points
indices = kdtree.query_radius(data[:100], r=0.5, return_sorted=True)
```

The `KDTree` can also be used from within numba functions


```python
import numpy as np
from numba import njit
from numba_kdtree import KDTree

def numba_function_with_kdtree(kdtree, data):
    for i in range(data.shape[0]):
        distances, indices = kdtree.query(data[0], k=30)
        #<Use the computed neighbors
        
data = np.random.random(3_000_000).reshape(-1, 3)
kdtree = KDTree(data, leafsize=10)

numba_function_with_kdtree(kdtree, data[:10000])
```

## TODOs

- Implement all scipy `ckdtree` functions
- Fix the parallel query functions