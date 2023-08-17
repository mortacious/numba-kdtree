//
// Created by mortacious on 1/6/21.
//

#pragma once

#if defined(_MSC_VER)
    // Microsoft
    #define _EXPORT __declspec(dllexport)
    #define _IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    #define _EXPORT __attribute__((visibility("default")))
    #define _IMPORT
#else
    // do nothing?
    #define _EXPORT
    #define _IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#if KDTREE_COMPILING
#   define KDTREE_PUBLIC _EXPORT
#else
#   define KDTREE_PUBLIC _IMPORT
#endif


#ifdef __cplusplus
extern "C"
{
#endif

#include "ckdtree_decl.h"

typedef struct ckdtree<float> ckdtree_float;
typedef struct ckdtree<double> ckdtree_double;

/* Init and Free methods in C */
KDTREE_PUBLIC ckdtree_float* ckdtree_init_float(char* tree_buffer, ckdtree_intp_t buffer_size, 
                                  float* data, ckdtree_intp_t* indices, 
                                  ckdtree_intp_t n, ckdtree_intp_t m,
                                  ckdtree_intp_t leafsize, float *mins, float *maxes);

KDTREE_PUBLIC ckdtree_double* ckdtree_init_double(char* tree_buffer, ckdtree_intp_t buffer_size, 
                                    double* data, ckdtree_intp_t* indices, 
                                    ckdtree_intp_t n, ckdtree_intp_t m,
                                    ckdtree_intp_t leafsize, double *mins, double *maxes);

KDTREE_PUBLIC void ckdtree_free_float(ckdtree_float* self);

KDTREE_PUBLIC void ckdtree_free_double(ckdtree_double* self);

/* Build methods in C */
KDTREE_PUBLIC int
ckdtree_build_float(ckdtree_float* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                    float *mins, float *maxes, int _balanced, int _compact);

KDTREE_PUBLIC int
ckdtree_build_double(ckdtree_double* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                     double *mins, double *maxes, int _balanced, int _compact);

KDTREE_PUBLIC ssize_t ckdtree_size_float(ckdtree_float* self);

KDTREE_PUBLIC ssize_t ckdtree_size_double(ckdtree_double* self);

KDTREE_PUBLIC int
ckdtree_query_knn_float(ckdtree_float* self, float *dd, ckdtree_intp_t *ii, ckdtree_intp_t *nn, float *xx, ckdtree_intp_t n, ckdtree_intp_t k,
                        double eps, double p, float distance_upper_bound);

KDTREE_PUBLIC int
ckdtree_query_knn_double(ckdtree_double* self, double *dd, ckdtree_intp_t *ii, ckdtree_intp_t *nn, double *xx, ckdtree_intp_t n,
                         ckdtree_intp_t k, double eps, double p, double distance_upper_bound);

typedef struct std::vector<ckdtree_intp_t> radius_result_set;

KDTREE_PUBLIC radius_result_set*
ckdtree_query_radius_float(ckdtree_float* self, float *x, ckdtree_intp_t n_queries, float r,
                           double eps, double p, bool return_length, bool sort_output);

KDTREE_PUBLIC radius_result_set*
ckdtree_query_radius_double(ckdtree_double* self, double *x, ckdtree_intp_t n_queries, double r,
                            double eps, double p, bool return_length, bool sort_output);

KDTREE_PUBLIC ckdtree_intp_t radius_result_set_get_size(radius_result_set* result_set);

KDTREE_PUBLIC void radius_result_set_copy_and_free(radius_result_set* result_set, ckdtree_intp_t* result);


KDTREE_PUBLIC ckdtree_intp_t leafsize_float(ckdtree_float* self);
KDTREE_PUBLIC ckdtree_intp_t leafsize_double(ckdtree_double* self);

KDTREE_PUBLIC ckdtree_intp_t nodesize_float(ckdtree_float* self);
KDTREE_PUBLIC ckdtree_intp_t nodesize_double(ckdtree_double* self);

KDTREE_PUBLIC ckdtree_intp_t copy_tree_float(ckdtree_float* self, char* buffer);
KDTREE_PUBLIC ckdtree_intp_t copy_tree_double(ckdtree_double* self, char* buffer);


#ifdef __cplusplus
}
#endif
