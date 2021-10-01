//
// Created by mortacious on 1/6/21.
//

#include "ckdtree_decl.h"
#include "ckdtree.h"

template<typename T>
ckdtree<T>* init_ckdtree(T* data, ckdtree_intp_t* indices, ckdtree_intp_t n, ckdtree_intp_t m,
                                  ckdtree_intp_t leafsize, T *mins, T *maxes) {
    ckdtree<T>* self = new ckdtree<T>;
    //self->tree_buffer = new std::vector<ckdtreenode<T>>;
    self->raw_data = data;
    self->n = n;
    self->m = m;
    self->leafsize = leafsize;
    self->raw_maxes = maxes;
    self->raw_mins = mins;
    self->raw_indices = indices;
    return self;
}

// c interface methods
ckdtree_float* ckdtree_init_float(float* data, ckdtree_intp_t* indices, ckdtree_intp_t n, ckdtree_intp_t m,
                             ckdtree_intp_t leafsize, float *mins, float *maxes) {
    return init_ckdtree<float>(data, indices, n, m, leafsize, mins, maxes);
}

ckdtree_double* ckdtree_init_double(double* data, ckdtree_intp_t* indices, ckdtree_intp_t n, ckdtree_intp_t m,
                              ckdtree_intp_t leafsize, double *mins, double *maxes) {
    return init_ckdtree<double>(data, indices, n, m, leafsize, mins, maxes);
}

void ckdtree_free_float(ckdtree_float* self) {
    //printf("free c data\n");
    //auto self_t = (ckdtree<float>*) self;
    delete self;
    self = nullptr;
}

void ckdtree_free_double(ckdtree_double* self) {
    //printf("free c data\n");
    //auto self_t = (ckdtree<double>*) self;
    //delete self_t->tree_buffer;
    delete self;
    self = nullptr;
}

ssize_t ckdtree_size_float(ckdtree_float* self) {
    if(!self) {
        return -1;
    }
    //auto self_t = (ckdtree<float>*) self;
    return self->size;
}

ssize_t ckdtree_size_double(ckdtree_double* self) {
    if(!self) {
        return -1;
    }
    //auto self_t = (ckdtree<double>*) self;
    return self->size;
}
