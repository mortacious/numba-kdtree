//
// Created by mortacious on 1/6/21.
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "ckdtree_decl.h"
#include "ckdtree.h"
#include "partial_sort.h"


// actual build method in C++
template<typename T>
ckdtree_intp_t
build(ckdtree<T>* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
      T *mins, T *maxes,
      const int _median, const int _compact)
{
    using DataType = T;
    using ckdtreenode = ckdtreenode<DataType>;

    const ckdtree_intp_t m = self->m;


    const DataType *data = self->raw_data;

    auto *indices = (ckdtree_intp_t *)(self->raw_indices);
    ckdtreenode *n;//, *root;
    ckdtree_intp_t node_index, _less, _greater;
    ckdtree_intp_t i, j, p, q, d;
    T size, split, minval, maxval;

    /* put a new node into the node stack */
    self->tree_buffer.emplace_back(start_idx, end_idx);
    //self->tree_buffer.push_back(new_node);
    node_index = self->tree_buffer.size() - 1;
    //root = tree_buffer_root(self->tree_buffer);
    n = &self->tree_buffer[node_index];//root + node_index;
    //memset(n, 0, sizeof(n[0]));

    //n->start_idx = start_idx;
    //n->end_idx = end_idx;
    //n->children = end_idx - start_idx;
    if (n->children <= self->leafsize) {
        /* below brute force limit, return leafnode */
        //std::cout << "leaf node with " << n->children << std::endl;
        n->split_dim = -1;
        return node_index;
    }
    else {
        if (CKDTREE_LIKELY(_compact)) {
            /* Recompute hyperrectangle bounds. This should lead to a more
             * compact kd-tree but comes at the expense of larger construction
             * time. However, construction time is usually dwarfed by the
             * query time by orders of magnitude.
             */
            const DataType *tmp_data_point;
            tmp_data_point = data + indices[start_idx] * m;
            for(i=0; i<m; ++i) {
                maxes[i] = tmp_data_point[i];
                mins[i] = tmp_data_point[i];
            }
            for (j = start_idx + 1; j < end_idx; ++j) {
                tmp_data_point = data + indices[j] * m;
                for(i=0; i<m; ++i) {
                    T tmp = tmp_data_point[i];
                    maxes[i] = maxes[i] > tmp ? maxes[i] : tmp;
                    mins[i] = mins[i] < tmp ? mins[i] : tmp;
                }
            }
        }

        /* split on the dimension with largest spread */
        d = 0;
        size = 0;
        for (i=0; i<m; ++i) {
            if (maxes[i] - mins[i] > size) {
                d = i;
                size = maxes[i] - mins[i];
            }
        }
        maxval = maxes[d];
        minval = mins[d];
        if (maxval == minval) {
            /* all points are identical; warn user?
             * return leafnode
             */
            n->split_dim = -1;
            return node_index;
        }

        /* construct new inner node */

        if (CKDTREE_LIKELY(_median)) {
            /* split on median to create a balanced tree
             * adopted from scikit-learn
             */
            i = (end_idx - start_idx) / 2;
            partition_node_indices<DataType>(data, indices + start_idx, d, i, m,
                                   end_idx - start_idx);
            p = start_idx + i;
            split = data[indices[p]*m+d];
        }
        else {
            /* split with the sliding midpoint rule */
            split = (maxval + minval) / 2;
        }

        p = start_idx;
        q = end_idx - 1;
        while (p <= q) {
            if (data[indices[p] * m + d] < split)
                ++p;
            else if (data[indices[q] * m + d] >= split)
                --q;
            else {
                ckdtree_intp_t t = indices[p];
                indices[p] = indices[q];
                indices[q] = t;
                ++p;
                --q;
            }
        }
        /* slide midpoint if necessary */
        if (p == start_idx) {
            /* no points less than split */
            j = start_idx;
            split = data[indices[j] * m + d];
            for (i = start_idx+1; i < end_idx; ++i) {
                if (data[indices[i] * m + d] < split) {
                    j = i;
                    split = data[indices[j] * m + d];
                }
            }
            ckdtree_intp_t t = indices[start_idx];
            indices[start_idx] = indices[j];
            indices[j] = t;
            p = start_idx + 1;
            q = start_idx;
        }
        else if (p == end_idx) {
            /* no points greater than split */
            j = end_idx - 1;
            split = data[indices[j] * m + d];
            for (i = start_idx; i < end_idx-1; ++i) {
                if (data[indices[i] * m + d] > split) {
                    j = i;
                    split = data[indices[j] * m + d];
                }
            }
            ckdtree_intp_t t = indices[end_idx-1];
            indices[end_idx-1] = indices[j];
            indices[j] = t;
            p = end_idx - 1;
            q = end_idx - 2;
        }

        if (CKDTREE_LIKELY(_compact)) {
            _less = build(self, start_idx, p, maxes, mins, _median, _compact);
            _greater = build(self, p, end_idx, maxes, mins, _median, _compact);
        }
        else
        {
            std::vector<DataType> tmp(m);
            DataType *mids = &tmp[0];

            for (i=0; i<m; ++i) mids[i] = maxes[i];
            mids[d] = split;
            _less = build(self, start_idx, p, mins, mids, _median, _compact);

            for (i=0; i<m; ++i) mids[i] = mins[i];
            mids[d] = split;
            _greater = build(self, p, end_idx, mids, maxes,  _median, _compact);
        }

        /* recompute n because std::vector can
         * reallocate its internal buffer
         */
        n = &self->tree_buffer[node_index];
        /* fill in entries */
        n->_less = _less;
        n->_greater = _greater;
        n->split_dim = d;
        n->split = split;

        return node_index;
    }
}


// c functions to call the c++ code
int ckdtree_build_float(ckdtree_float* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                        float *mins, float *maxes, int _balanced, int _compact) {
    if (!self) {
        return 1;
    }
    //auto self = (ckdtree<float>*) self_; // cast the void pointer
    build<float>(self, start_idx, end_idx, mins, maxes, _balanced, _compact);
    self->size = self->tree_buffer.size();
    return 0;
}

int ckdtree_build_double(ckdtree_double* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                         double *mins, double *maxes, int _balanced, int _compact) {
    if (!self) {
        return 1;
    }
    //auto self = (ckdtree<double>*) self_; // cast the void pointer
    build<double>(self, start_idx, end_idx, mins, maxes, _balanced, _compact);
    self->size = self->tree_buffer.size();

    return 0;
}