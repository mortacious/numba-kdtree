//
// Created by mortacious on 1/6/21.
//

#include "ckdtree_decl.h"
#include "ckdtree.h"
#include <cstring>
#include "numba/core/runtime/nrt_external.h"

namespace detail {
    template<typename T>
    void free_ckdtree(void* self) {
        auto self_t = reinterpret_cast<ckdtree<T>*>(self);
        delete self_t;
    }
}

template<typename T>
NRT_MemInfo* init_ckdtree(NRT_api_functions* nrt,
                          char* tree_buffer, 
                          ckdtree_intp_t buffer_size, 
                          T* data, 
                          ckdtree_intp_t* indices, 
                          ckdtree_intp_t n, 
                          ckdtree_intp_t m,
                          ckdtree_intp_t leafsize, 
                          T *mins, 
                          T *maxes) {
    ckdtree<T>* self = new ckdtree<T>(); 
    NRT_MemInfo *mi = nrt->manage_memory(self, detail::free_ckdtree<T>);
    if(tree_buffer != nullptr) {
        // initialize existing tree
        auto buffer_ptr = reinterpret_cast<ckdtreenode<T>*>(tree_buffer);
        buffer_size /= sizeof(ckdtreenode<T>);
        self->tree_buffer.assign(buffer_ptr, buffer_ptr + buffer_size);
    }
    self->raw_data = data;
    self->n = n;
    self->m = m;
    self->leafsize = leafsize;
    self->raw_maxes = maxes;
    self->raw_mins = mins;
    self->raw_indices = indices;

    return mi;
}

// c interface methods
NRT_MemInfo* ckdtree_init_float(void* nrt, char* tree_buffer, ckdtree_intp_t buffer_size, 
                                  float* data, ckdtree_intp_t* indices, 
                                  ckdtree_intp_t n, ckdtree_intp_t m,
                                  ckdtree_intp_t leafsize, float *mins, float *maxes) {
    return init_ckdtree<float>(reinterpret_cast<NRT_api_functions*>(nrt), tree_buffer, buffer_size, data, indices, n, m, leafsize, mins, maxes);
}

NRT_MemInfo* ckdtree_init_double(void* nrt, char* tree_buffer, ckdtree_intp_t buffer_size, 
                                    double* data, ckdtree_intp_t* indices, 
                                    ckdtree_intp_t n, ckdtree_intp_t m,
                                    ckdtree_intp_t leafsize, double *mins, double *maxes) {
    return init_ckdtree<double>(reinterpret_cast<NRT_api_functions*>(nrt), tree_buffer, buffer_size, data, indices, n, m, leafsize, mins, maxes);
}

ssize_t ckdtree_size_float(ckdtree_float* self) {
    return self->size;
}

ssize_t ckdtree_size_double(ckdtree_double* self) {
    return self->size;
}

ckdtree_intp_t leafsize_float(ckdtree_float* self) {
    return self->leafsize;
}

ckdtree_intp_t leafsize_double(ckdtree_double* self) {
    return self->leafsize;
}

ckdtree_intp_t nodesize_float(ckdtree_float* self) {
    return sizeof(ckdtreenode<float>);
}

ckdtree_intp_t nodesize_double(ckdtree_double* self) {
    return sizeof(ckdtreenode<double>);
}

ckdtree_intp_t copy_tree_float(ckdtree_float* self, char* buffer) {
    size_t size = self->tree_buffer.size() * sizeof(ckdtreenode<float>);
    memcpy(buffer, self->tree_buffer.data(), size);
    return size;
}

ckdtree_intp_t copy_tree_double(ckdtree_double* self, char* buffer) {
    size_t size = self->tree_buffer.size() * sizeof(ckdtreenode<double>);
    memcpy(buffer, self->tree_buffer.data(), size);
    return size;
}