//
// Created by mortacious on 1/7/21.
//

#pragma once
#ifndef HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_H
#define HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_H
#include "distance_base.h"

template<typename T>
struct PlainDist1D {
    static inline const T side_distance_from_min_max(
            const ckdtree<T> * tree, const T x,
            const T min,
            const T max,
            const ckdtree_intp_t k
    )
    {
        T s, t;
        s = 0;
        t = x - max;
        if (t > s) {
            s = t;
        } else {
            t = min - x;
            if (t > s) s = t;
        }
        return s;
    }

    static inline void
    interval_interval(const ckdtree<T> * tree,
                      const Rectangle<T>& rect1, const Rectangle<T>& rect2,
                      const ckdtree_intp_t k,
                      T *min, T *max)
    {
        /* Compute the minimum/maximum distance along dimension k between points in
         * two hyperrectangles.
         */
        *min = std::fmax(0., fmax(rect1.mins()[k] - rect2.maxes()[k],
                                     rect2.mins()[k] - rect1.maxes()[k]));
        *max = std::fmax(rect1.maxes()[k] - rect2.mins()[k],
                            rect2.maxes()[k] - rect1.mins()[k]);
    }

    static inline T
    point_point(const ckdtree<T> * tree,
                const T *x, const T *y,
                const ckdtree_intp_t k) {
        return std::fabs(x[k] - y[k]);
    }
};

template<typename T>
using MinkowskiDistPp = BaseMinkowskiDistPp<T, PlainDist1D<T>>;

template<typename T>
using MinkowskiDistPinf = BaseMinkowskiDistPinf<T, PlainDist1D<T>>;

template<typename T>
using MinkowskiDistP1 = BaseMinkowskiDistP1<T, PlainDist1D<T>>;

template<typename T>
using NonOptimizedMinkowskiDistP2 = BaseMinkowskiDistP2<T, PlainDist1D<T>> ;

/*
 * Measuring distances
 * ===================
 */
template<typename T>
inline T
sqeuclidean_distance_double(const T *u, const T *v, ckdtree_intp_t n)
{
    T s;
    ckdtree_intp_t i = 0;
    // manually unrolled loop, might be vectorized
    T acc[4] = {0., 0., 0., 0.};
    for (; i + 4 <= n; i += 4) {
        T _u[4] = {u[i], u[i + 1], u[i + 2], u[i + 3]};
        T _v[4] = {v[i], v[i + 1], v[i + 2], v[i + 3]};
        T diff[4] = {_u[0] - _v[0],
                          _u[1] - _v[1],
                          _u[2] - _v[2],
                          _u[3] - _v[3]};
        acc[0] += diff[0] * diff[0];
        acc[1] += diff[1] * diff[1];
        acc[2] += diff[2] * diff[2];
        acc[3] += diff[3] * diff[3];
    }
    s = acc[0] + acc[1] + acc[2] + acc[3];
    if (i < n) {
        for(; i<n; ++i) {
            T d = u[i] - v[i];
            s += d * d;
        }
    }
    return s;
}


template<typename T>
struct MinkowskiDistP2: NonOptimizedMinkowskiDistP2<T> {
    static inline T
    point_point_p(const ckdtree<T> * tree,
                  const T *x, const T *y,
                  const double p, const ckdtree_intp_t k,
                  const T upperbound)
    {
        return sqeuclidean_distance_double(x, y, k);
    }
};

//template<typename T>
//struct BoxDist1D {
//    static inline void _interval_interval_1d (
//            T min, T max,
//            T *realmin, T *realmax,
//            const T full, const T half
//    )
//    {
//        /* Minimum and maximum distance of two intervals in a periodic box
//         *
//         * min and max is the nonperiodic distance between the near
//         * and far edges.
//         *
//         * full and half are the box size and 0.5 * box size.
//         *
//         * value is returned in realmin and realmax;
//         *
//         * This function is copied from kdcount, and the convention
//         * of is that
//         *
//         * min = rect1.min - rect2.max
//         * max = rect1.max - rect2.min = - (rect2.min - rect1.max)
//         *
//         * We will fix the convention later.
//         * */
//        if (CKDTREE_UNLIKELY(full <= 0)) {
//            /* A non-periodic dimension */
//            /* \/     */
//            if(max <= 0 || min >= 0) {
//                /* do not pass though 0 */
//                min = ckdtree_fabs(min);
//                max = ckdtree_fabs(max);
//                if(min < max) {
//                    *realmin = min;
//                    *realmax = max;
//                } else {
//                    *realmin = max;
//                    *realmax = min;
//                }
//            } else {
//                min = ckdtree_fabs(min);
//                max = ckdtree_fabs(max);
//                *realmax = ckdtree_fmax(max, min);
//                *realmin = 0;
//            }
//            /* done with non-periodic dimension */
//            return;
//        }
//        if(max <= 0 || min >= 0) {
//            /* do not pass through 0 */
//            min = ckdtree_fabs(min);
//            max = ckdtree_fabs(max);
//            if(min > max) {
//                double t = min;
//                min = max;
//                max = t;
//            }
//            if(max < half) {
//                /* all below half*/
//                *realmin = min;
//                *realmax = max;
//            } else if(min > half) {
//                /* all above half */
//                *realmax = full - min;
//                *realmin = full - max;
//            } else {
//                /* min below, max above */
//                *realmax = half;
//                *realmin = ckdtree_fmin(min, full - max);
//            }
//        } else {
//            /* pass though 0 */
//            min = -min;
//            if(min > max) max = min;
//            if(max > half) max = half;
//            *realmax = max;
//            *realmin = 0;
//        }
//    }
//    static inline void
//    interval_interval(const ckdtree<T> * tree,
//                      const Rectangle<T>& rect1, const Rectangle<T>& rect2,
//                      const ckdtree_intp_t k,
//                      T *min, T *max)
//    {
//        /* Compute the minimum/maximum distance along dimension k between points in
//         * two hyperrectangles.
//         */
//        _interval_interval_1d(rect1.mins()[k] - rect2.maxes()[k],
//                              rect1.maxes()[k] - rect2.mins()[k], min, max,
//                              tree->raw_boxsize_data[k], tree->raw_boxsize_data[k + rect1.m]);
//    }
//
//    static inline T
//    point_point(const ckdtree<T> * tree,
//                const T *x, const T *y,
//                const ckdtree_intp_t k)
//    {
//        double r1;
//        r1 = wrap_distance(x[k] - y[k], tree->raw_boxsize_data[k + tree->m], tree->raw_boxsize_data[k]);
//        r1 = ckdtree_fabs(r1);
//        return r1;
//    }
//
//    static inline const T
//    wrap_position(const T x, const T boxsize)
//    {
//        if (boxsize <= 0) return x;
//        const T r = std::floor(x / boxsize);
//        T x1 = x - r * boxsize;
//        /* ensure result is within the box. */
//        while(x1 >= boxsize) x1 -= boxsize;
//        while(x1 < 0) x1 += boxsize;
//        return x1;
//    }

//    static inline const T side_distance_from_min_max(
//            const ckdtree<T> * tree, const T x,
//            const T min,
//            const T max,
//            const ckdtree_intp_t k
//    )
//    {
//        T s, t, tmin, tmax;
//        T fb = tree->raw_boxsize_data[k];
//        T hb = tree->raw_boxsize_data[k + tree->m];
//
//        if (fb <= 0) {
//            /* non-periodic dimension */
//            s = PlainDist1D::side_distance_from_min_max(tree, x, min, max, k);
//            return s;
//        }
//
//        /* periodic */
//        s = 0;
//        tmax = x - max;
//        tmin = x - min;
//        /* is the test point in this range */
//        if(CKDTREE_LIKELY(tmax < 0 && tmin > 0)) {
//            /* yes. min distance is 0 */
//            return 0;
//        }
//
//        /* no */
//        tmax = ckdtree_fabs(tmax);
//        tmin = ckdtree_fabs(tmin);
//
//        /* make tmin the closer edge */
//        if(tmin > tmax) { t = tmin; tmin = tmax; tmax = t; }
//
//        /* both edges are less than half a box. */
//        /* no wrapping, use the closer edge */
//        if(tmax < hb) return tmin;
//
//        /* both edge are more than half a box. */
//        /* wrapping on both edge, use the
//         * wrapped further edge */
//        if(tmin > hb) return fb - tmax;
//
//        /* the further side is wrapped */
//        tmax = fb - tmax;
//        if(tmin > tmax) return tmax;
//        return tmin;
//    }
//
//private:
//    static inline double
//    wrap_distance(const double x, const double hb, const double fb)
//    {
//        double x1;
//        if (CKDTREE_UNLIKELY(x < -hb)) x1 = fb + x;
//        else if (CKDTREE_UNLIKELY(x > hb)) x1 = x - fb;
//        else x1 = x;
//#if 0
//        printf("ckdtree_fabs_b x : %g x1 %g\n", x, x1);
//#endif
//        return x1;
//    }
//
//
//};
//
//
//typedef BaseMinkowskiDistPp<BoxDist1D> BoxMinkowskiDistPp;
//typedef BaseMinkowskiDistPinf<BoxDist1D> BoxMinkowskiDistPinf;
//typedef BaseMinkowskiDistP1<BoxDist1D> BoxMinkowskiDistP1;
//typedef BaseMinkowskiDistP2<BoxDist1D> BoxMinkowskiDistP2;

#endif //HYPERSPACE_NEARESTNEIGHBOR_DISTANCE_H
