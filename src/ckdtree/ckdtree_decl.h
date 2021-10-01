//
// Created by mortacious on 1/6/21.
//

#pragma once
#ifndef HYPERSPACE_NEARESTNEIGHBOR_CKDTREE_DECL_H
#define HYPERSPACE_NEARESTNEIGHBOR_CKDTREE_DECL_H
#include <numpy/npy_common.h>
#include <vector>
#include <cmath>

 // some useful macros
#define CKDTREE_LIKELY(x) NPY_LIKELY(x)
#define CKDTREE_UNLIKELY(x)  NPY_UNLIKELY(x)
#define CKDTREE_PREFETCH(x, rw, loc)  NPY_PREFETCH(x, rw, loc)

#define ckdtree_intp_t ssize_t

template<typename T>
struct ckdtreenode {
    ckdtreenode(ckdtree_intp_t start_idx, ckdtree_intp_t end_idx)
        : split_dim(-1), children(end_idx-start_idx), split(-1.0), start_idx(start_idx), end_idx(end_idx), _less(-1), _greater(-1) {}
    ckdtree_intp_t      split_dim;
    ckdtree_intp_t      children;
    T   split;
    ckdtree_intp_t      start_idx;
    ckdtree_intp_t      end_idx;
    ckdtree_intp_t      _less;
    ckdtree_intp_t      _greater;
};

template<typename T>
struct ckdtree {
    ckdtree()
     : tree_buffer(), raw_data(nullptr), n(0), m(0), leafsize(0), raw_maxes(nullptr), raw_mins(nullptr), raw_indices(nullptr), size(-1) {}

     ~ckdtree() = default;

    // tree structure
    std::vector<ckdtreenode<T>> tree_buffer;
    // meta data
    T   *raw_data; // pointer to the raw data
    ckdtree_intp_t      n; // number of data points
    ckdtree_intp_t      m; // number of features
    ckdtree_intp_t      leafsize; // maximum leaf size in the tree
    T   *raw_maxes; // maximum of the bounding box
    T   *raw_mins; // minimum of the bounding box
    ckdtree_intp_t      *raw_indices; // index array into the data values
    ckdtree_intp_t size; // size of the tree
};

#endif //HYPERSPACE_NEARESTNEIGHBOR_KDTREE_DECL_H
