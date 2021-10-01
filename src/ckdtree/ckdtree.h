//
// Created by mortacious on 1/6/21.
//

#pragma once
#ifndef HYPERSPACE_NEARESTNEIGHBOR_CKDTREE_H
#define HYPERSPACE_NEARESTNEIGHBOR_CKDTREE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "ckdtree_decl.h"

typedef struct ckdtree<float> ckdtree_float;
typedef struct ckdtree<double> ckdtree_double;

/* Init and Free methods in C */
ckdtree_float* ckdtree_init_float(float* data, ckdtree_intp_t* indices, ckdtree_intp_t n, ckdtree_intp_t m,
                             ckdtree_intp_t leafsize, float *maxes, float *mins);

ckdtree_double* ckdtree_init_double(double* data, ckdtree_intp_t* indices, ckdtree_intp_t n, ckdtree_intp_t m,
                              ckdtree_intp_t leafsize, double *maxes, double *mins);

void ckdtree_free_float(ckdtree_float* self);

void ckdtree_free_double(ckdtree_double* self);

/* Build methods in C */
int
ckdtree_build_float(ckdtree_float* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                    float *mins, float *maxes, int _balanced, int _compact);

int
ckdtree_build_double(ckdtree_double* self, ckdtree_intp_t start_idx, ckdtree_intp_t end_idx,
                     double *mins, double *maxes, int _balanced, int _compact);

ssize_t ckdtree_size_float(ckdtree_float* self);

ssize_t ckdtree_size_double(ckdtree_double* self);

int
ckdtree_query_knn_float(ckdtree_float* self, float *dd, ckdtree_intp_t *ii, float *xx, ckdtree_intp_t n, ckdtree_intp_t k,
                        double eps, double p, float distance_upper_bound);

int
ckdtree_query_knn_double(ckdtree_double* self, double *dd, ckdtree_intp_t *ii, double *xx, ckdtree_intp_t n,
                         ckdtree_intp_t k, double eps, double p, double distance_upper_bound);

typedef struct std::vector<ckdtree_intp_t> radius_result_set;

radius_result_set*
ckdtree_query_radius_float(ckdtree_float* self, float *x, ckdtree_intp_t n_queries, float r,
                           double eps, double p, bool return_length, bool sort_output);

radius_result_set*
ckdtree_query_radius_double(ckdtree_double* self, double *x, ckdtree_intp_t n_queries, double r,
                            double eps, double p, bool return_length, bool sort_output);

ckdtree_intp_t radius_result_set_get_size(radius_result_set* result_set);

void radius_result_set_copy_and_free(radius_result_set* result_set, ckdtree_intp_t* result);

#ifdef __cplusplus
}
#endif
#endif //HYPERSPACE_NEARESTNEIGHBOR_CKDTREE_H
