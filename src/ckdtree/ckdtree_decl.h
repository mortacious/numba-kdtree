//
// Created by mortacious on 1/6/21.
//

#pragma once
//#include <numpy/npy_common.h>
#include <vector>
#include <cmath>
#include <sys/types.h>
 // some useful macros
#ifdef HAVE___BUILTIN_EXPECT
    #define CKDTREE_LIKELY(x) __builtin_expect(!!(x), 1)
    #define CKDTREE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define CKDTREE_LIKELY(x) (x)
    #define CKDTREE_UNLIKELY(x) (x)
#endif

#ifdef HAVE___BUILTIN_PREFETCH
    /* unlike _mm_prefetch also works on non-x86 */
    #define CKDTREE_PREFETCH(x, rw, loc) __builtin_prefetch((x), (rw), (loc))
#else
    #ifdef HAVE__MM_PREFETCH
        /* _MM_HINT_ET[01] (rw = 1) unsupported, only available in gcc >= 4.9 */
        #define CKDTREE_PREFETCH(x, rw, loc) _mm_prefetch((x), loc == 0 ? _MM_HINT_NTA : \
                                                     (loc == 1 ? _MM_HINT_T2 : \
                                                      (loc == 2 ? _MM_HINT_T1 : \
                                                       (loc == 3 ? _MM_HINT_T0 : -1))))
    #else
        #define CKDTREE_PREFETCH(x, rw,loc)
    #endif
#endif

// define the ssize_t type for windows systems
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif


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