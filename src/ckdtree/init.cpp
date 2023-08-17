//
// Created by mortacious on 1/6/21.
//

#include "ckdtree_decl.h"
#include "ckdtree.h"
#include <cstring>

template<typename T>
ckdtree<T>* init_ckdtree(char* tree_buffer, ckdtree_intp_t buffer_size, 
                         T* data, ckdtree_intp_t* indices, 
                         ckdtree_intp_t n, ckdtree_intp_t m,
                         ckdtree_intp_t leafsize, T *mins, T *maxes) {
    ckdtree<T>* self = new ckdtree<T>;
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
    return self;
}

// c interface methods
ckdtree_float* ckdtree_init_float(char* tree_buffer, ckdtree_intp_t buffer_size, 
                                  float* data, ckdtree_intp_t* indices, 
                                  ckdtree_intp_t n, ckdtree_intp_t m,
                                  ckdtree_intp_t leafsize, float *mins, float *maxes) {
    return init_ckdtree<float>(tree_buffer, buffer_size, data, indices, n, m, leafsize, mins, maxes);
}

ckdtree_double* ckdtree_init_double(char* tree_buffer, ckdtree_intp_t buffer_size, 
                                    double* data, ckdtree_intp_t* indices, 
                                    ckdtree_intp_t n, ckdtree_intp_t m,
                                    ckdtree_intp_t leafsize, double *mins, double *maxes) {
    return init_ckdtree<double>(tree_buffer, buffer_size, data, indices, n, m, leafsize, mins, maxes);
}

void ckdtree_free_float(ckdtree_float* self) {
    //printf("free c data float\n");
    //auto self_t = (ckdtree<float>*) self;
    if(self != nullptr) {
        delete self;
    }
    self = nullptr;
}

void ckdtree_free_double(ckdtree_double* self) {
    //printf("free c data double\n");
    //auto self_t = (ckdtree<double>*) self;
    //delete self_t->tree_buffer;
    if(self != nullptr) {
        delete self;
    }
    self = nullptr;
}

#define RETURN_UNINITIALIZED(self) \
    if(!self){ \
        return -1; \
    }

ssize_t ckdtree_size_float(ckdtree_float* self) {
    RETURN_UNINITIALIZED(self)
    //auto self_t = (ckdtree<float>*) self;
    return self->size;
}

ssize_t ckdtree_size_double(ckdtree_double* self) {
    RETURN_UNINITIALIZED(self)
    //auto self_t = (ckdtree<double>*) self;
    return self->size;
}

ckdtree_intp_t leafsize_float(ckdtree_float* self) {
    RETURN_UNINITIALIZED(self)
    return self->leafsize;
}

ckdtree_intp_t leafsize_double(ckdtree_double* self) {
    RETURN_UNINITIALIZED(self)
    return self->leafsize;
}

ckdtree_intp_t nodesize_float(ckdtree_float* self) {
    RETURN_UNINITIALIZED(self)
    return sizeof(ckdtreenode<float>);
}

ckdtree_intp_t nodesize_double(ckdtree_double* self) {
    RETURN_UNINITIALIZED(self)
    return sizeof(ckdtreenode<double>);
}

ckdtree_intp_t copy_tree_float(ckdtree_float* self, char* buffer) {
    RETURN_UNINITIALIZED(self)
    size_t size = self->tree_buffer.size() * sizeof(ckdtreenode<float>);
    memcpy(buffer, self->tree_buffer.data(), size);
    return size;
}

ckdtree_intp_t copy_tree_double(ckdtree_double* self, char* buffer) {
    RETURN_UNINITIALIZED(self)
    size_t size = self->tree_buffer.size() * sizeof(ckdtreenode<double>);
    memcpy(buffer, self->tree_buffer.data(), size);
    return size;
}