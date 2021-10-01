//
// Created by mortacious on 1/7/21.
//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>
#include "ckdtree_decl.h"
#include "ckdtree.h"
#include "distance.h"

/*
 * Priority queue
 * ==============
 */

union heapcontents {
    ckdtree_intp_t intdata;
    void     *ptrdata;
};

struct heapitem {
    double priority;
    heapcontents contents;
};

struct heap {

    std::vector<heapitem> _heap;
    ckdtree_intp_t n;
    ckdtree_intp_t space;

    heap (ckdtree_intp_t initial_size) : _heap(initial_size) {
        space = initial_size;
        n = 0;
    }

    inline void push(heapitem &item) {
        ckdtree_intp_t i;
        heapitem t;
        n++;

        if (n > space) _heap.resize(2*space+1);
        space = _heap.size();

        i = n-1;
        _heap[i] = item;
        while ((i > 0) && (_heap[i].priority < _heap[(i-1)/2].priority)) {
            t = _heap[(i-1)/2];
            _heap[(i-1)/2] = _heap[i];
            _heap[i] = t;
            i = (i-1)/2;
        }
    }

    inline heapitem peek() {return _heap[0];}

    inline void remove() {
        heapitem t;
        ckdtree_intp_t i, j, k, l, nn;
        _heap[0] = _heap[n-1];
        n--;
        /*
         * No point in freeing up space as the heap empties.
         * The whole heap gets deallocated at the end of any
         * query below. Just keep the space to avoid unnecessary
         * reallocs.
         */
        nn = n;
        i=0;
        j=1;
        k=2;
        while (((j<nn) && (_heap[i].priority > _heap[j].priority)) ||
               ((k<nn) && (_heap[i].priority > _heap[k].priority))) {
            if ((k<nn) && (_heap[j].priority >_heap[k].priority))
                l = k;
            else
                l = j;
            t = _heap[l];
            _heap[l] = _heap[i];
            _heap[i] = t;
            i = l;
            j = 2*i+1;
            k = 2*i+2;
        }
    }

    inline heapitem pop() {
        heapitem it = _heap[0];
        remove();
        return it;
    }
};

/*
 * nodeinfo
 * ========
 */

template<typename T>
struct nodeinfo {
    const ckdtreenode<T>  *node;
    ckdtree_intp_t     m;
    T  min_distance; /* full min distance */
    T        buf[1]; // the good old struct hack
    /* accessors to 'packed' attributes */
    inline T  * const side_distances() {
        /* min distance to the query per side; we
         * update this as the query is proceeded */
        return buf;
    }
    inline T        * const maxes() {
        return buf + m;
    }
    inline T        * const mins() {
        return buf + 2 * m;
    }

    inline void init_plain(const struct nodeinfo * from) {
        /* skip copying min and max, because we only need side_distance array in this case. */
        std::memcpy(buf, from->buf, sizeof(T) * m);
        min_distance = from->min_distance;
    }

    inline void update_side_distance(const int d, const T new_side_distance, const T p) {
        if (CKDTREE_UNLIKELY(std::isinf(p))) {
            min_distance = std::fmax(min_distance, new_side_distance);
        } else {
            min_distance += new_side_distance - side_distances()[d];
        }
        side_distances()[d] = new_side_distance;
    }
};

/*
 * Memory pool for nodeinfo structs
 * ================================
 */

template<typename T>
struct nodeinfo_pool {

    std::vector<char*> pool;

    ckdtree_intp_t alloc_size;
    ckdtree_intp_t arena_size;
    ckdtree_intp_t m;
    char *arena;
    char *arena_ptr;

    nodeinfo_pool(ckdtree_intp_t m) {
        alloc_size = sizeof(nodeinfo<T>) + (3 * m -1)*sizeof(T);
        alloc_size = 64*(alloc_size/64)+64;
        arena_size = 4096*((64*alloc_size)/4096)+4096;
        arena = new char[arena_size];
        arena_ptr = arena;
        pool.push_back(arena);
        this->m = m;
    }

    ~nodeinfo_pool() {
        for (ckdtree_intp_t i = pool.size()-1; i >= 0; --i)
            delete [] pool[i];
    }

    inline nodeinfo<T> *allocate() {
        nodeinfo<T> *ni1;
        auto m1 = (ckdtree_intp_t)arena_ptr;
        auto m0 = (ckdtree_intp_t)arena;
        if ((arena_size-(ckdtree_intp_t)(m1-m0))<alloc_size) {
            arena = new char[arena_size];
            arena_ptr = arena;
            pool.push_back(arena);
        }
        ni1 = (nodeinfo<T>*)arena_ptr;
        ni1->m = m;
        arena_ptr += alloc_size;
        return ni1;
    }
};

/* k-nearest neighbor search for a single point x */
template <typename T, typename MinMaxDist>
static void
query_single_point(const ckdtree<T> *self,
                   T   *result_distances,
                   ckdtree_intp_t      *result_indices,
                   const T  *x,
                   const ckdtree_intp_t     k,
                   const double  eps,
                   const double  p,
                   T  distance_upper_bound)
{

    //static double inf = strtod("INF", NULL);

    /* memory pool to allocate and automatically reclaim nodeinfo structs */
    nodeinfo_pool<T> nipool(self->m);

    /*
     * priority queue for chasing nodes
     * entries are:
     *  - minimum distance between the cell and the target
     *  - distances between the nearest side of the cell and the target
     *    the head node of the cell
     */
    heap q(12);

    /*
     *  priority queue for chasing nodes
     *  entries are:
     *   - minimum distance between the cell and the target
     *   - distances between the nearest side of the cell and the target
     *     the head node of the cell
     */
    heap neighbors(k);

    ckdtree_intp_t      i;
    const ckdtree_intp_t m = self->m;
    nodeinfo<T>      *ni1;
    nodeinfo<T>       *ni2;
    T   d;
    double   epsfac;
    heapitem      it, it2, neighbor;
    const ckdtreenode<T>   *node;
    const ckdtreenode<T>   *inode;

    /* set up first nodeifo */
    ni1 = nipool.allocate();
    ni1->node = &self->tree_buffer[0];

    /* initialize first node, update min_distance */
    ni1->min_distance = 0;

    for (i=0; i<m; ++i) {
        //auto min = self->raw_mins[i];
        //auto max = self->raw_maxes[i];

        ni1->mins()[i] = self->raw_mins[i];
        ni1->maxes()[i] = self->raw_maxes[i];

        T side_distance;

        side_distance = PlainDist1D<T>::side_distance_from_min_max(
                self, x[i], self->raw_mins[i], self->raw_maxes[i], i);

        side_distance = MinMaxDist::distance_p(side_distance, p);

        ni1->side_distances()[i] = 0;
        ni1->update_side_distance(i, side_distance, p);
    }

    /* fiddle approximation factor */
    if (CKDTREE_LIKELY(p == 2.0)) {
        double tmp = 1. + eps;
        epsfac = 1. / (tmp*tmp);
    }
    else if (eps == 0.)
        epsfac = 1.;
    else if (std::isinf(p))
        epsfac = 1. / (1. + eps);
    else
        epsfac = 1. / std::pow((1. + eps), p);

    /* internally we represent all distances as distance**p */
    if (CKDTREE_LIKELY(p == 2.0)) {
        T tmp = distance_upper_bound;
        distance_upper_bound = tmp*tmp;
    }
    else if ((!std::isinf(p)) && (!std::isinf(distance_upper_bound)))
        distance_upper_bound = std::pow(distance_upper_bound,p);

    for(;;) {
        if (ni1->node->split_dim == -1) {
            node = ni1->node;

            /* brute-force */
            {
                const ckdtree_intp_t start_idx = node->start_idx;
                const ckdtree_intp_t end_idx = node->end_idx;
                const T *data = self->raw_data;
                const ckdtree_intp_t *indices = self->raw_indices;

                CKDTREE_PREFETCH(data+indices[start_idx]*m, 0, m);
                if (start_idx < end_idx - 1)
                        CKDTREE_PREFETCH(data+indices[start_idx+1]*m, 0, m);

                for (i=start_idx; i<end_idx; ++i) {

                    if (i < end_idx - 2)
                            CKDTREE_PREFETCH(data+indices[i+2]*m, 0, m);
                    d = MinMaxDist::point_point_p(self, data+indices[i]*m, x, p, m, distance_upper_bound);
                    if (d < distance_upper_bound) {
                        /* replace furthest neighbor */
                        if (neighbors.n == k)
                            neighbors.remove();
                        neighbor.priority = -d;
                        neighbor.contents.intdata = indices[i];
                        neighbors.push(neighbor);

                        /* adjust upper bound for efficiency */
                        if (neighbors.n == k)
                            distance_upper_bound = -neighbors.peek().priority;
                    }
                }
            }
            /* done with this node, get another */
            if (q.n == 0) {
                /* no more nodes to visit */
                break;
            }
            else {
                it = q.pop();
                ni1 = (nodeinfo<T>*)(it.contents.ptrdata);
            }

        }
        else {
            inode = ni1->node;
            const ckdtree_intp_t split_dim = inode->split_dim;
            const T split = inode->split;

            /*
             * we don't push cells that are too far onto the queue at all,
             * but since the distance_upper_bound decreases, we might get
             * here even if the cell's too far
             */
            if (ni1->min_distance > distance_upper_bound*epsfac) {
                /* since this is the nearest cell, we're done, bail out */
                break;
            }
            // set up children for searching
            // ni2 will be pushed to the queue

            ni2 = nipool.allocate();

            /*
             * non periodic : the 'near' node is know from the
             * relative distance to the split, and
             * has the same distance as the parent node.
             *
             * we set ni1 to 'near', and set ni2 to 'far'.
             * we only recalculate the distance of 'far' later.
             *
             * This code branch doesn't use min and max.
             */
            ni2->init_plain(ni1);

            T side_distance;

            if (x[split_dim] < split) {
                ni1->node = &self->tree_buffer[inode->_less]; //+ inode->_less;
                ni2->node = &self->tree_buffer[inode->_greater]; // + inode->_greater;
                side_distance = split - x[split_dim];
            } else {
                ni1->node = &self->tree_buffer[inode->_greater];//self->ctree + inode->_greater;
                ni2->node = &self->tree_buffer[inode->_less];//self->ctree + inode->_less;
                side_distance = x[split_dim] - split;
            }

            side_distance = MinMaxDist::distance_p(side_distance, p);

            ni2->update_side_distance(split_dim, side_distance, p);

            /* Ensure ni1 is closer than ni2 */
            if (ni1->min_distance > ni2->min_distance) {
                {
                    std::swap(ni1, ni2);
                }
            }

            /*
             * near child is at the same or closer than the distance as the current node
             * we're going here next, so no point pushing it on the queue
             * no need to recompute the distance or the side_distances
             */

            /*
             * far child can be further
             * push it on the queue if it's near enough
             */

            if (ni2->min_distance<=distance_upper_bound*epsfac) {
                it2.priority = ni2->min_distance;
                it2.contents.ptrdata = (void*) ni2;
                q.push(it2);
            }
        }
    }
    /* heapsort */
    std::vector<heapitem> sorted_neighbors(k);
    for(i = neighbors.n - 1; i >=0; --i) {
        auto neighbor = neighbors.pop();
        result_indices[i] = neighbor.contents.intdata;
        if (CKDTREE_LIKELY(p == 2.0))
            result_distances[i] = std::sqrt(-neighbor.priority);
        else if ((p == 1.) || (std::isinf(p)))
            result_distances[i] = -neighbor.priority;
        else
            result_distances[i] = std::pow((-neighbor.priority),(1./p));
    }
}

/* Query n points for their k nearest neighbors */
template<typename T>
int
query_knn(const ckdtree<T>      *self,
          T        *dd,
          ckdtree_intp_t           *ii,
          const T  *xx,
          const ckdtree_intp_t     n,
          const ckdtree_intp_t     k,
          const double  eps,
          const double  p,
          const T  distance_upper_bound)
{
#define HANDLE(cond, t, kls) \
    if(cond) { \
        query_single_point<t, kls>(self, dd_row, ii_row, xx_row, k, eps, p, distance_upper_bound); \
    } else

    ckdtree_intp_t m = self->m;
    ckdtree_intp_t i;
    for (i=0; i<n; ++i) {
        T *dd_row = dd + (i*k);
        ckdtree_intp_t *ii_row = ii + (i*k);
        const T *xx_row = xx + (i*m);
        HANDLE(CKDTREE_LIKELY(p == 2), T, MinkowskiDistP2<T>)
        HANDLE(p == 1, T, MinkowskiDistP1<T>)
        HANDLE(std::isinf(p), T, MinkowskiDistPinf<T>)
        HANDLE(1, T, MinkowskiDistPp<T>)
        {}
    }
    return 0;
}

/* C Interface functions */
int
ckdtree_query_knn_float(ckdtree_float* self, float *dd, ckdtree_intp_t *ii, float *xx, ckdtree_intp_t n, ckdtree_intp_t k,
                        double eps, double p, float distance_upper_bound) {
    //auto self = (ckdtree<float>*) self_; // cast the void pointer
    return query_knn<float>(self, dd, ii, xx, n, k, eps, p, distance_upper_bound);
}

int
ckdtree_query_knn_double(ckdtree_double* self, double *dd, ckdtree_intp_t *ii, double *xx, ckdtree_intp_t n,
                         ckdtree_intp_t k, double eps, double p, double distance_upper_bound) {
    //auto self = (ckdtree<double>*) self_; // cast the void pointer
    return query_knn<double>(self, dd, ii, xx, n, k, eps, p, distance_upper_bound);

}
