//
// Created by mortacious on 1/7/21.
//

#pragma once
#ifndef HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_BASE_H
#define HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_BASE_H
#include "ckdtree_decl.h"
#include "rectangle.h"


template <typename T, typename Dist1D>
struct BaseMinkowskiDistPp {
    /* 1-d pieces
     * These should only be used if p != infinity
     */

    static inline void
    interval_interval_p(const ckdtree<T> * tree,
                        const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                        const ckdtree_intp_t k, const double p,
                        T *min, T *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        *min = std::pow(*min, p);
        *max = std::pow(*max, p);
    }

    static inline void
    rect_rect_p(const ckdtree<T> * tree,
                const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                const double p,
                T *min, T *max)
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            T min_, max_;

            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);

            *min += std::pow(min_, p);
            *max += std::pow(max_, p);
        }
    }

    static inline T
    point_point_p(const ckdtree<T> * tree,
                  const T *x, const T *y,
                  const double p, const ckdtree_intp_t k,
                  const T upperbound)
    {
        /*
         * Compute the distance between x and y
         *
         * Computes the Minkowski p-distance to the power p between two points.
         * If the distance**p is larger than upperbound, then any number larger
         * than upperbound may be returned (the calculation is truncated).
         */

        ckdtree_intp_t i;
        T r, r1;
        r = 0;
        for (i=0; i<k; ++i) {
            r1 = Dist1D::point_point(tree, x, y, i);
            r += std::pow(r1, p);
            if (r>upperbound)
                return r;
        }
        return r;
    }

    static inline T
    distance_p(const T s, const T p)
    {
        return std::pow(s,p);
    }
};

template <typename T, typename Dist1D>
struct BaseMinkowskiDistP1 : public BaseMinkowskiDistPp<T, Dist1D> {

    static inline void
    interval_interval_p(const ckdtree<T> * tree,
                        const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                        const ckdtree_intp_t k, const double p,
                        T *min, T *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
    }

    static inline void
    rect_rect_p(const ckdtree<T> * tree,
                const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                const double p,
                T *min, T *max)
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            T min_, max_;

            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);

            *min += min_;
            *max += max_;
        }
    }

    static inline T
    point_point_p(const ckdtree<T> * tree,
                  const T *x, const T *y,
                  const double p, const ckdtree_intp_t k,
                  const T upperbound)
    {
        ckdtree_intp_t i;
        T r;
        r = 0;
        for (i=0; i<k; ++i) {
            r += Dist1D::point_point(tree, x, y, i);
            if (r>upperbound)
                return r;
        }
        return r;
    }

    static inline T
    distance_p(const T s, const T p)
    {
        return s;
    }
};

template <typename T, typename Dist1D>
struct BaseMinkowskiDistPinf : public BaseMinkowskiDistPp<T, Dist1D> {

    static inline void
    interval_interval_p(const ckdtree<T> * tree,
                        const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                        const ckdtree_intp_t k, double p,
                        T *min, T *max)
    {
        return rect_rect_p(tree, rect1, rect2, p, min, max);
    }

    static inline void
    rect_rect_p(const ckdtree<T>* tree,
                const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                const double p,
                T *min, T *max)
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            T min_, max_;

            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);

            *min = std::fmax(*min, min_);
            *max = std::fmax(*max, max_);
        }
    }

    static inline T
    point_point_p(const ckdtree<T> * tree,
                  const T *x, const T *y,
                  const double p, const ckdtree_intp_t k,
                  const T upperbound)
    {
        ckdtree_intp_t i;
        T r;
        r = 0;
        for (i=0; i<k; ++i) {
            r = std::fmax(r,Dist1D::point_point(tree, x, y, i));
            if (r>upperbound)
                return r;
        }
        return r;
    }
    static inline T
    distance_p(const T s, const T p)
    {
        return s;
    }
};

template <typename T, typename Dist1D>
struct BaseMinkowskiDistP2 : public BaseMinkowskiDistPp<T, Dist1D> {
    static inline void
    interval_interval_p(const ckdtree<T> * tree,
                        const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                        const ckdtree_intp_t k, const double p,
                        T *min, T *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        Dist1D::interval_interval(tree, rect1, rect2, k, min, max);
        *min *= *min;
        *max *= *max;
    }

    static inline void
    rect_rect_p(const ckdtree<T> * tree,
                const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                const double p,
                T *min, T *max)
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            T min_, max_;

            Dist1D::interval_interval(tree, rect1, rect2, i, &min_, &max_);
            min_ *= min_;
            max_ *= max_;

            *min += min_;
            *max += max_;
        }
    }

    static inline T
    point_point_p(const ckdtree<T> * tree,
                  const T *x, const T *y,
                  const double p, const ckdtree_intp_t k,
                  const T upperbound)
    {
        ckdtree_intp_t i;
        T r;
        r = 0;
        for (i=0; i<k; ++i) {
            T r1 = Dist1D::point_point(tree, x, y, i);
            r += r1 * r1;
            if (r>upperbound)
                return r;
        }
        return r;
    }

    static inline T
    distance_p(const T s, const T p)
    {
        return s * s;
    }
};

#endif //HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_BASE_H
