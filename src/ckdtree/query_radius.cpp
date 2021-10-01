//
// Created by mortacious on 1/11/21.
//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <new>

#include "ckdtree_decl.h"
#include "rectangle.h"
#include "ckdtree.h"

template<typename T>
void
traverse_no_checking(const ckdtree<T> *self,
                     const int return_length,
                     std::vector<ckdtree_intp_t> &results,
                     const ckdtree_intp_t node_index)
{
    const ckdtree_intp_t *indices = self->raw_indices;
    const ckdtreenode<T>* node = &self->tree_buffer[node_index];

    if (node->split_dim == -1) {  /* leaf node */
        const ckdtree_intp_t start = node->start_idx;
        const ckdtree_intp_t end = node->end_idx;
        for (ckdtree_intp_t i = start; i < end; ++i) {
            if (return_length) {
                results[0] ++;
            } else {
                results.push_back(indices[i]);
            }
        }
    }
    else {
        traverse_no_checking(self, return_length, results, node->_less);
        traverse_no_checking(self, return_length, results, node->_greater);
    }
}


template <typename T, typename MinMaxDist> static void
traverse_checking(const ckdtree<T> *self,
                  const int return_length,
                  std::vector<ckdtree_intp_t> &results,
                  const ckdtree_intp_t node_index,
                  RectRectDistanceTracker<T, MinMaxDist> *tracker
)
{
    //const ckdtreenode<T> *lnode;
    const ckdtreenode<T>* node = &self->tree_buffer[node_index];

    T d;
    ckdtree_intp_t i;

    if (tracker->min_distance > tracker->upper_bound * tracker->epsfac) {
        return;
    }
    else if (tracker->max_distance < tracker->upper_bound / tracker->epsfac) {
        // push all the points into the vector
        traverse_no_checking<T>(self, return_length, results, node_index);
    }
    else if (node->split_dim == -1)  { /* leaf node */
        /* brute-force */
        //lnode = node;
        const double p = tracker->p;
        const T tub = tracker->upper_bound;
        const T *tpt = tracker->rect1.mins();
        const T *data = self->raw_data;
        const ckdtree_intp_t *indices = self->raw_indices;
        const ckdtree_intp_t m = self->m;
        const ckdtree_intp_t start = node->start_idx;
        const ckdtree_intp_t end = node->end_idx;

        CKDTREE_PREFETCH(data + indices[start] * m, 0, m);
        if (start < end - 1)
                CKDTREE_PREFETCH(data + indices[start+1] * m, 0, m);

        for (i = start; i < end; ++i) {

            if (i < end -2 )
                    CKDTREE_PREFETCH(data + indices[i+2] * m, 0, m);

            d = MinMaxDist::point_point_p(self, data + indices[i] * m, tpt, p, m, tub);

            if (d <= tub) {
                if(return_length) {
                    results[0] ++;
                } else {
                    results.push_back((ckdtree_intp_t) indices[i]);
                }
            }
        }
    }
    else {
        tracker->push_less_of(2, node);
        traverse_checking(self, return_length, results, node->_less, tracker);
        tracker->pop();

        tracker->push_greater_of(2, node);
        traverse_checking(self, return_length, results, node->_greater, tracker);
        tracker->pop();
    }
}

template<typename T>
int
query_radius(const ckdtree<T> *self, const T *x,
             const float r, const double p, const double eps,
             const ckdtree_intp_t n_queries,
             std::vector<ckdtree_intp_t> *results,
             const bool return_length,
             const bool sort_output)
{
#define HANDLE(cond, kls) \
    if(cond) { \
        if(return_length) results[i].push_back(0); \
        RectRectDistanceTracker<T, kls> tracker(self, point, rect, p, eps, r); \
        traverse_checking<T>(self, return_length, results[i], 0, &tracker); \
    } else

    for (ckdtree_intp_t i=0; i < n_queries; ++i) {
        const ckdtree_intp_t m = self->m;
        Rectangle<T> rect(m, self->raw_mins, self->raw_maxes);
        Rectangle<T> point(m, x + i * m, x + i * m);
        HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2<T>)
        HANDLE(p == 1, MinkowskiDistP1<T>)
        HANDLE(std::isinf(p), MinkowskiDistPinf<T>)
        HANDLE(1, MinkowskiDistPp<T>)
        {}

        if (!return_length && sort_output) {
            std::sort(results[i].begin(), results[i].end());
        }
    }
    return 0;
}

/* C Interface functions */
radius_result_set*
ckdtree_query_radius_float(ckdtree_float* self, float *x, ckdtree_intp_t n_queries, float r,
                        double eps, double p, bool return_length, bool sort_output) {
   // auto self = (ckdtree<float>*) self_; // cast the void pointer
    auto result_vector = new std::vector<ckdtree_intp_t>(); // allocate the result vector on the heap
    result_vector->reserve(16); // make some initial space
    query_radius<float>(self, x, r, p, eps, n_queries, result_vector, return_length, sort_output);
    return result_vector;
}

radius_result_set*
ckdtree_query_radius_double(ckdtree_double* self, double *x, ckdtree_intp_t n_queries, double r,
                            double eps, double p, bool return_length, bool sort_output) {
    //auto self = (ckdtree<double>*) self_; // cast the void pointer
    auto result_vector = new std::vector<ckdtree_intp_t>(); // allocate the result vector on the heap
    result_vector->reserve(16); // make some initial space
    query_radius<double>(self, x, r, p, eps, n_queries, result_vector, return_length, sort_output);
    return result_vector;
}

ckdtree_intp_t radius_result_set_get_size(radius_result_set* result_set) {
    //auto result_set = (std::vector<ckdtree_intp_t>*) result_set_;
    return result_set->size();
}

void radius_result_set_copy_and_free(radius_result_set* result_set, ckdtree_intp_t* result) {
    //auto result_set = (std::vector<ckdtree_intp_t>*) result_set_;
//    for(int i=0; i<result_set->size(); ++i) {
//        std::cout << (*result_set)[i] << " ";
//    }
//    std::cout << std::endl;

    memcpy(result, result_set->data(), result_set->size()*sizeof(ckdtree_intp_t));
    delete result_set; // free the result set
}

