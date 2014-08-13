////////////////////////////////////////////////////////////////////////////////
//                  Shewchuk Predicates ported to CUDA
////////////////////////////////////////////////////////////////////////////////

#pragma once

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

/* On some machines, the exact arithmetic routines might be defeated by the  */
/*   use of internal extended precision floating-point registers.  Sometimes */
/*   this problem can be fixed by defining certain values to be volatile,    */
/*   thus forcing them to be stored to memory and rounded off.  This isn't   */
/*   a great solution, though, as it slows the arithmetic down.              */
/*                                                                           */
/* To try this out, write "#define volatile" below.  Normally,       */
/*   however, should be defined to be nothing.  ("#define INEXACT".) */

// float or double
#ifdef REAL_TYPE_FP32
#define Absolute(a) fabsf(a) 
#define MUL(a,b)    __fmul_rn(a,b)
#else
#define Absolute(a) fabs(a) 
#define MUL(a,b)    __dmul_rn(a,b)
#endif

/* Which of the following two methods of finding the absolute values is      */
/*   fastest is compiler-dependent.  A few compilers can inline and optimize */
/*   the fabs() call; but most will incur the overhead of a function call,   */
/*   which is disastrously slow.  A faster way on IEEE machines might be to  */
/*   mask the appropriate bit, but that's difficult to do in C.              */

//#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))


/* Many of the operations are broken up into two pieces, a main part that    */
/*   performs an approximate operation, and a "tail" that computes the       */
/*   roundoff error of that operation.                                       */
/*                                                                           */
/* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    */
/*   Split(), and Two_Product() are all implemented as described in the      */
/*   reference.  Each of these macros requires certain variables to be       */
/*   defined in the calling routine.  The variables `bvirt', `c', `abig',    */
/*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   */
/*   they store the result of an operation that may incur roundoff error.    */
/*   The input parameter `x' (or the highest numbered `x_' parameter) must   */
/*   also be declared `INEXACT'.                                             */

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a; \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (RealType) (a + b); \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (RealType) (x - a); \
    avirt = x - bvirt; \
    bround = b - bvirt; \
    around = a - avirt; \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (RealType) (a + b); \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (RealType) (a - x); \
    avirt = x + bvirt; \
    bround = bvirt - b; \
    around = a - avirt; \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (RealType) (a - b); \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
    c = MUL(predConsts[ Splitter ], a); \
    abig = (RealType) (c - a); \
    ahi = c - abig; \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo); \
    Split(b, bhi, blo); \
    err1 = x - MUL(ahi, bhi); \
    err2 = err1 - MUL(alo, bhi); \
    err3 = err2 - MUL(ahi, blo); \
    y = MUL(alo, blo) - err3

#define Two_Product(a, b, x, y) \
    x = MUL(a, b); \
    Two_Product_Tail(a, b, x, y)

/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = MUL(a, b); \
    Split(a, ahi, alo); \
    err1 = x - MUL(ahi, bhi); \
    err2 = err1 - MUL(alo, bhi); \
    err3 = err2 - MUL(ahi, blo); \
    y = MUL(alo, blo) - err3

/* Macros for summing expansions of various fixed lengths.  These are all    */
/*   unrolled versions of Expansion_Sum().                                   */

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b , _i, x0); \
    Two_Sum( a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0); \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
    Split(b, bhi, blo); \
    Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
    Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
    Two_Sum(_i, _0, _k, x1); \
    Fast_Two_Sum(_j, _k, x3, x2)

/*****************************************************************************/
/*                                                                           */
/*  exactinit()   Initialize the variables used for exact arithmetic.        */
/*                                                                           */
/*  `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in   */
/*  floating-point arithmetic.  `epsilon' bounds the relative roundoff       */
/*  error.  It is used for floating-point error analysis.                    */
/*                                                                           */
/*  `splitter' is used to split floating-point numbers into two half-        */
/*  length significands for exact multiplication.                            */
/*                                                                           */
/*  I imagine that a highly optimizing compiler might be too smart for its   */
/*  own good, and somehow cause this routine to fail, if it pretends that    */
/*  floating-point arithmetic is too much like real arithmetic.              */
/*                                                                           */
/*  Don't change this routine unless you fully understand it.                */
/*                                                                           */
/*****************************************************************************/

__global__ void kerInitPredicate( RealType* predConsts )
{
    RealType half;
    RealType epsilon, splitter; 
    RealType check, lastcheck;
    int every_other;

    every_other = 1;
    half        = 0.5;
    epsilon     = 1.0;
    splitter    = 1.0;
    check       = 1.0;

    /* Repeatedly divide `epsilon' by two until it is too small to add to    */
    /*   one without causing roundoff.  (Also check if the sum is equal to   */
    /*   the previous sum, for machines that round up instead of using exact */
    /*   rounding.  Not that this library will work on such machines anyway. */
    do
    {
        lastcheck   = check;
        epsilon     *= half;

        if (every_other)
        {
            splitter *= 2.0;
        }

        every_other = !every_other;
        check       = 1.0 + epsilon;
    } while ((check != 1.0) && (check != lastcheck));

    /* Error bounds for orientation and incircle tests. */
    predConsts[ Epsilon ]           = epsilon; 
    predConsts[ Splitter ]          = splitter + 1.0;
    predConsts[ Resulterrbound ]    = (3.0 + 8.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundA ]      = (3.0 + 16.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundB ]      = (2.0 + 12.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundC ]      = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
    predConsts[ O3derrboundA ]      = (7.0 + 56.0 * epsilon) * epsilon;
    predConsts[ O3derrboundB ]      = (3.0 + 28.0 * epsilon) * epsilon;
    predConsts[ O3derrboundC ]      = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
    predConsts[ IccerrboundA ]      = (10.0 + 96.0 * epsilon) * epsilon;
    predConsts[ IccerrboundB ]      = (4.0 + 48.0 * epsilon) * epsilon;
    predConsts[ IccerrboundC ]      = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
    predConsts[ IsperrboundA ]      = (16.0 + 224.0 * epsilon) * epsilon;
    predConsts[ IsperrboundB ]      = (5.0 + 72.0 * epsilon) * epsilon;
    predConsts[ IsperrboundC ]      = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;
    predConsts[ O3derrboundAlifted ]= (11.0 + 112.0 * epsilon) * epsilon;
        //(10.0 + 112.0 * epsilon) * epsilon;
    predConsts[ O2derrboundAlifted ]= (6.0 + 48.0 * epsilon) * epsilon;
    predConsts[ O1derrboundAlifted ]= (3.0 + 16.0 * epsilon) * epsilon;

    return;
}

/*****************************************************************************/
/*                                                                           */
/*  d_scale_expansion_zeroelim()   Multiply an expansion by a scalar,          */
/*                               eliminating zero components from the        */
/*                               output expansion.                           */
/*                                                                           */
/*  Sets h = be.  See either version of my paper for details.                */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
/*  properties as well.  (That is, if e has one of these properties, so      */
/*  will h.)                                                                 */
/*                                                                           */
/*****************************************************************************/

/* e and h cannot be the same. */
__device__ int d_scale_expansion_zeroelim
(
const RealType* predConsts,
int             elen,
RealType*       e,
RealType        b,
RealType*       h
)
{
    RealType Q, sum;
    RealType hh;
    RealType product1;
    RealType product0;
    int eindex, hindex;
    RealType enow;
    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     */
/*                                  components from the output expansion.    */
/*                                                                           */
/*  Sets h = e + f.  See the long version of my paper for details.           */
/*                                                                           */
/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
/*  properties.                                                              */
/*                                                                           */
/*****************************************************************************/

/* h cannot be e or f. */
__device__ int d_fast_expansion_sum_zeroelim
(
int      elen,
RealType *e,
int      flen,
RealType *f,
RealType *h
)
{
    RealType Q;
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex, hindex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

__device__ RealType d_fast_expansion_sum_sign
(
int      elen,
RealType *e,
int      flen,
RealType *f
)
{
    RealType Q;
    RealType lastTerm; 
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    lastTerm = 0.0; 
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                lastTerm = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    if ((Q != 0.0) || (lastTerm == 0.0)) {
        lastTerm = Q;
    }
    return lastTerm;
}

__device__ RealType d_fast_expansion_sum_estimate
(
int      elen,
RealType *e,
int      flen,
RealType *f
)
{
    RealType Q;
    RealType est; 
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    est = 0.0; 
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        est += hh; 
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            est += hh; 
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        est += hh; 
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        est += hh; 
    }

    est += Q; 
    
    return est;
}

__device__ int d_squared_scale_expansion_zeroelim
(
const RealType* predConsts,
int             elen,
RealType        *e,
RealType        b,
RealType        *h
)
{
    RealType Q, sum, Q2, sum2;
    RealType hh;
    RealType product1, product2;
    RealType product0;
    int eindex, hindex;
    RealType enow;
    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;

    hindex = 0;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    Two_Product_Presplit(hh, b, bhi, blo, Q2, hh); 

    if (hh != 0) {
        h[hindex++] = hh; 
    }

    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);

        Two_Product_Presplit(hh, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product1, sum, Q, hh);

        Two_Product_Presplit(hh, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }
    }

    if (Q != 0) {
        Two_Product_Presplit(Q, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh);

        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }
    }

    if ((Q2 != 0) || (hindex == 0)) {
        h[hindex++] = Q2; 
    }

    return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  orient2dfast()   Approximate 2D orientation test.  Nonrobust.            */
/*  orient2dexact()   Exact 2D orientation test.  Robust.                    */
/*  orient2dslow()   Another exact 2D orientation test.  Robust.             */
/*  orient2d()   Adaptive exact 2D orientation test.  Robust.                */
/*                                                                           */
/*               Return a positive value if the points pa, pb, and pc occur  */
/*               in counterclockwise order; a negative value if they occur   */
/*               in clockwise order; and zero if they are collinear.  The    */
/*               result is also a rough approximation of twice the signed    */
/*               area of the triangle defined by the three points.           */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In orient2d() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, orient2d() is usually quite */
/*  fast, but will run more slowly when the input points are collinear or    */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/
 
__forceinline__ __device__ RealType orient2dExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc
)
{
    RealType axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
    RealType axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
    RealType aterms[4], bterms[4], cterms[4];
    RealType v[8];
    int vlength;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pa[0], pc[1], axcy1, axcy0);
    Two_Two_Diff(axby1, axby0, axcy1, axcy0,
        aterms[3], aterms[2], aterms[1], aterms[0]);

    Two_Product(pb[0], pc[1], bxcy1, bxcy0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0,
        bterms[3], bterms[2], bterms[1], bterms[0]);

    Two_Product(pc[0], pa[1], cxay1, cxay0);
    Two_Product(pc[0], pb[1], cxby1, cxby0);
    Two_Two_Diff(cxay1, cxay0, cxby1, cxby0,
        cterms[3], cterms[2], cterms[1], cterms[0]);

    vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);

    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

__device__ RealType orient2dFastExact
(
const RealType* predConsts,
const RealType *pa,
const RealType *pb,
const RealType *pc
)
{
    RealType detleft, detright, det;
    RealType detsum, errbound;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        } else {
            detsum = detleft + detright;
        }
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        } else {
            detsum = -detleft - detright;
        }
    } else {
        return det;
    }

    errbound = predConsts[ CcwerrboundA ] * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return orient2dExact( predConsts, pa, pb, pc ); 
}

/*****************************************************************************/
/*                                                                           */
/*  orient3dfast()   Approximate 3D orientation test.  Nonrobust.            */
/*  orient3dexact()   Exact 3D orientation test.  Robust.                    */
/*  orient3dslow()   Another exact 3D orientation test.  Robust.             */
/*  orient3d()   Adaptive exact 3D orientation test.  Robust.                */
/*                                                                           */
/*               Return a positive value if the point pd lies below the      */
/*               plane passing through pa, pb, and pc; "below" is defined so */
/*               that pa, pb, and pc appear in counterclockwise order when   */
/*               viewed from above the plane.  Returns a negative value if   */
/*               pd lies above the plane.  Returns zero if the points are    */
/*               coplanar.  The result is also a rough approximation of six  */
/*               times the signed volume of the tetrahedron defined by the   */
/*               four points.                                                */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In orient3d() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, orient3d() is usually quite */
/*  fast, but will run more slowly when the input points are coplanar or     */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

__forceinline__ __device__ RealType orient3dFast
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType det;
    RealType permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2];
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy) 
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
        + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
        + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);
    errbound = predConsts[ O3derrboundA ] * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return 0.0;
}

__forceinline__ __device__ float orient3dDet
(
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2];
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    return adz * (bdxcdy - cdxbdy) 
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);
}

__device__ RealType orient3dExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
    RealType bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
    RealType axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
    RealType bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
    RealType ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    RealType temp8[8];
    int templen;
    RealType abc[12], bcd[12], cda[12], dab[12];
    int abclen, bcdlen, cdalen, dablen;
    RealType adet[24], bdet[24];
    int alen, blen, clen, dlen;
    RealType abdet[48], cddet[48];
    int ablen, cdlen;
    int i;

    RealType* cdet = adet; 
    RealType* ddet = bdet; 

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

    Two_Product(pb[0], pc[1], bxcy1, bxcy0);
    Two_Product(pc[0], pb[1], cxby1, cxby0);
    Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

    Two_Product(pc[0], pd[1], cxdy1, cxdy0);
    Two_Product(pd[0], pc[1], dxcy1, dxcy0);
    Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

    Two_Product(pd[0], pa[1], dxay1, dxay0);
    Two_Product(pa[0], pd[1], axdy1, axdy0);
    Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

    Two_Product(pa[0], pc[1], axcy1, axcy0);
    Two_Product(pc[0], pa[1], cxay1, cxay0);
    Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

    Two_Product(pb[0], pd[1], bxdy1, bxdy0);
    Two_Product(pd[0], pb[1], dxby1, dxby0);
    Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

    templen = d_fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);
    cdalen = d_fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda);
    templen = d_fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);
    dablen = d_fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab);
    for (i = 0; i < 4; i++) {
        bd[i] = -bd[i];
        ac[i] = -ac[i];
    }
    templen = d_fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);
    abclen = d_fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc);
    templen = d_fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);
    bcdlen = d_fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd);

    alen = d_scale_expansion_zeroelim(predConsts, bcdlen, bcd, pa[2], adet);
    blen = d_scale_expansion_zeroelim(predConsts, cdalen, cda, -pb[2], bdet);
    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

    clen = d_scale_expansion_zeroelim(predConsts, dablen, dab, pc[2], cdet);
    dlen = d_scale_expansion_zeroelim(predConsts, abclen, abc, -pd[2], ddet);
    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

    return d_fast_expansion_sum_sign(ablen, abdet, cdlen, cddet);
}

__device__ RealType orient3dAdaptExact
(
const RealType *predConsts, 
const RealType *pa,
const RealType *pb,
const RealType *pc,
const RealType *pd,
RealType permanent
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    RealType det, errbound;

    RealType bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
    RealType bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
    RealType bc[4], ca[4], ab[4];
    RealType adet[8], bdet[8], cdet[8];
    int alen, blen, clen;
    RealType abdet[16];
    int ablen;

    RealType adxtail, bdxtail, cdxtail;
    RealType adytail, bdytail, cdytail;
    RealType adztail, bdztail, cdztail;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    adx = (RealType) (pa[0] - pd[0]);
    bdx = (RealType) (pb[0] - pd[0]);
    cdx = (RealType) (pc[0] - pd[0]);
    ady = (RealType) (pa[1] - pd[1]);
    bdy = (RealType) (pb[1] - pd[1]);
    cdy = (RealType) (pc[1] - pd[1]);
    adz = (RealType) (pa[2] - pd[2]);
    bdz = (RealType) (pb[2] - pd[2]);
    cdz = (RealType) (pc[2] - pd[2]);

    Two_Product(bdx, cdy, bdxcdy1, bdxcdy0);
    Two_Product(cdx, bdy, cdxbdy1, cdxbdy0);
    Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc[3], bc[2], bc[1], bc[0]);
    alen = d_scale_expansion_zeroelim( predConsts, 4, bc, adz, adet );

    Two_Product(cdx, ady, cdxady1, cdxady0);
    Two_Product(adx, cdy, adxcdy1, adxcdy0);
    Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca[3], ca[2], ca[1], ca[0]);
    blen = d_scale_expansion_zeroelim( predConsts, 4, ca, bdz, bdet );

    Two_Product(adx, bdy, adxbdy1, adxbdy0);
    Two_Product(bdx, ady, bdxady1, bdxady0);
    Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab[3], ab[2], ab[1], ab[0]);
    clen = d_scale_expansion_zeroelim( predConsts, 4, ab, cdz, cdet );

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
    det = d_fast_expansion_sum_estimate(ablen, abdet, clen, cdet);

    errbound = predConsts[ O3derrboundB ] * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pd[0], adx, adxtail);
    Two_Diff_Tail(pb[0], pd[0], bdx, bdxtail);
    Two_Diff_Tail(pc[0], pd[0], cdx, cdxtail);
    Two_Diff_Tail(pa[1], pd[1], ady, adytail);
    Two_Diff_Tail(pb[1], pd[1], bdy, bdytail);
    Two_Diff_Tail(pc[1], pd[1], cdy, cdytail);
    Two_Diff_Tail(pa[2], pd[2], adz, adztail);
    Two_Diff_Tail(pb[2], pd[2], bdz, bdztail);
    Two_Diff_Tail(pc[2], pd[2], cdz, cdztail);

    if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0)
        && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0)
        && (adztail == 0.0) && (bdztail == 0.0) && (cdztail == 0.0)) {
            return det;
    }

    errbound = predConsts[ O3derrboundC ] * permanent + predConsts[ Resulterrbound ] * Absolute(det);
    det += (adz * ((bdx * cdytail + cdy * bdxtail)
        - (bdy * cdxtail + cdx * bdytail))
        + adztail * (bdx * cdy - bdy * cdx))
        + (bdz * ((cdx * adytail + ady * cdxtail)
        - (cdy * adxtail + adx * cdytail))
        + bdztail * (cdx * ady - cdy * adx))
        + (cdz * ((adx * bdytail + bdy * adxtail)
        - (ady * bdxtail + bdx * adytail))
        + cdztail * (adx * bdy - ady * bdx));
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return orient3dExact( predConsts, pa, pb, pc, pd );
}

__forceinline__ __device__ RealType orient3dFastAdaptExact
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType det;
    RealType permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2];
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy) 
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
        + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
        + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);
    errbound = predConsts[ O3derrboundA ] * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return orient3dAdaptExact( predConsts, pa, pb, pc, pd, permanent ); 
}

/*****************************************************************************/
/*                                                                           */
/*  inspherefast()   Approximate 3D insphere test.  Nonrobust.               */
/*  insphereexact()   Exact 3D insphere test.  Robust.                       */
/*  insphereslow()   Another exact 3D insphere test.  Robust.                */
/*  insphere()   Adaptive exact 3D insphere test.  Robust.                   */
/*                                                                           */
/*               Return a positive value if the point pe lies inside the     */
/*               sphere passing through pa, pb, pc, and pd; a negative value */
/*               if it lies outside; and zero if the five points are         */
/*               cospherical.  The points pa, pb, pc, and pd must be ordered */
/*               so that they have a positive orientation (as defined by     */
/*               orient3d()), or the sign of the result will be reversed.    */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In insphere() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, insphere() is usually quite */
/*  fast, but will run more slowly when the input points are cospherical or  */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

__noinline__ __device__ void two_mult_sub
(
const RealType *predConsts, 
const RealType *pa, 
const RealType *pb, 
RealType       *ab
)
{
    RealType axby1, axby0, bxay1, bxay0; 

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);
}


__noinline__ __device__ int mult_add
(
const RealType     *predConsts, 
RealType           *a,  
int                lena, 
RealType           fa,
RealType           *b, 
int                lenb, 
RealType           fb,
RealType           *c, 
int                lenc, 
RealType           fc, 
RealType           *tempa, 
RealType           *tempb, 
RealType           *tempab, 
RealType           *ret
) 
{
    int tempalen = d_scale_expansion_zeroelim(predConsts, lena, a, fa, tempa);
    int tempblen = d_scale_expansion_zeroelim(predConsts, lenb, b, fb, tempb);
    int tempablen = d_fast_expansion_sum_zeroelim(tempalen, tempa, tempblen, tempb, tempab); 
    tempalen = d_scale_expansion_zeroelim(predConsts, lenc, c, fc, tempa);
    return d_fast_expansion_sum_zeroelim(tempalen, tempa, tempablen, tempab, ret);
}

__noinline__ __device__ int calc_det
(
const RealType     *predConsts, 
RealType           *a,
int                alen,
RealType           *b,
int                blen, 
RealType           *c,
int                clen,
RealType           *d,
int                dlen, 
const RealType     *f,
RealType           *temp1a, 
RealType           *temp1b,
RealType           *temp2,
RealType           *temp8x,
RealType           *temp8y,
RealType           *temp8z,
RealType           *temp16, 
RealType           *ret24
)
{
    int temp1alen = d_fast_expansion_sum_zeroelim(alen, a, blen, b, temp1a);
    int temp1blen = d_fast_expansion_sum_zeroelim(clen, c, dlen, d, temp1b);
    for (int i = 0; i < temp1blen; i++) {
        temp1b[i] = -temp1b[i];
    }
    int temp2len = d_fast_expansion_sum_zeroelim(temp1alen, temp1a, temp1blen, temp1b, temp2);
    int xlen = d_squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[0], temp8x);
    int ylen = d_squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[1], temp8y);
    int len = d_fast_expansion_sum_zeroelim(xlen, temp8x, ylen, temp8y, temp16);
    int zlen = d_squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[2], temp8z);

    return d_fast_expansion_sum_zeroelim(len, temp16, zlen, temp8z, ret24);
}

__noinline__ __device__ int calc_det_adapt
(
const RealType     *predConsts, 
RealType           *a,
int                alen,
const RealType     x,
const RealType     y,
const RealType     z,
RealType           *tempx,
RealType           *tempy,
RealType           *tempz,
RealType           *temp2, 
RealType           *ret3
)
{
    int xlen = d_squared_scale_expansion_zeroelim(predConsts, alen, a, x, tempx);
    int ylen = d_squared_scale_expansion_zeroelim(predConsts, alen, a, y, tempy);
    int zlen = d_squared_scale_expansion_zeroelim(predConsts, alen, a, z, tempz);
    int len = d_fast_expansion_sum_zeroelim(xlen, tempx, ylen, tempy, temp2);

    return d_fast_expansion_sum_zeroelim(len, temp2, zlen, tempz, ret3);
}

__device__ RealType insphereExact
(
const RealType* predConsts, 
RealType*       predData,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
const RealType* pe
)
{
    // Index into global memory
    RealType* temp96    = predData; predData += Temp96Size;     //  abcd, bcde, cdea, deab, eabc;
    RealType* det384x   = predData; predData += Det384xSize;
    RealType* det384y   = predData; predData += Det384ySize;
    RealType* detxy     = predData; predData += DetxySize;  
    RealType* adet      = predData; predData += AdetSize;
    RealType* bdet      = predData; predData += BdetSize;
    RealType* abdet     = predData; predData += AbdetSize;
    RealType* cdedet    = predData; 
    RealType* det384z   = det384x; 
    RealType* cdet      = adet;
    RealType* ddet      = bdet;
    RealType* edet      = adet;
    RealType* cddet     = abdet; 
    RealType* temp48a   = det384x; 
    RealType* temp48b   = det384y; 

    RealType ab[4], bc[4], cd[4], de[4], ea[4];
    RealType ac[4], bd[4], ce[4], da[4], eb[4];
    RealType temp8a[8], temp8b[8], temp16[16];
    RealType abc[24], bcd[24], cde[24], dea[24], eab[24];
    RealType abd[24], bce[24], cda[24], deb[24], eac[24];
    int abclen, bcdlen, cdelen, dealen, eablen;
    int abdlen, bcelen, cdalen, deblen, eaclen;
    int alen, blen, clen, dlen, elen;
    int ablen, cdlen, cd_elen;

    // (pa[0] * pb[1]) # (pb[0] * pa[1]) => ab[0..3]
    two_mult_sub( predConsts, pa, pb, ab ); 
    two_mult_sub( predConsts, pb, pc, bc ); 
    two_mult_sub( predConsts, pc, pd, cd ); 
    two_mult_sub( predConsts, pd, pe, de ); 
    two_mult_sub( predConsts, pe, pa, ea ); 
    two_mult_sub( predConsts, pa, pc, ac ); 
    two_mult_sub( predConsts, pb, pd, bd ); 
    two_mult_sub( predConsts, pc, pe, ce ); 
    two_mult_sub( predConsts, pd, pa, da ); 
    two_mult_sub( predConsts, pe, pb, eb ); 

    // pa[2] # pb[2] # pc[2] => abc[24]
    abclen = mult_add( predConsts, bc, 4, pa[2], ac, 4, -pb[2], ab, 4, pc[2], temp8a, temp8b, temp16, abc ); 
    bcdlen = mult_add( predConsts, cd, 4, pb[2], bd, 4, -pc[2], bc, 4, pd[2], temp8a, temp8b, temp16, bcd ); 
    cdelen = mult_add( predConsts, de, 4, pc[2], ce, 4, -pd[2], cd, 4, pe[2], temp8a, temp8b, temp16, cde ); 
    dealen = mult_add( predConsts, ea, 4, pd[2], da, 4, -pe[2], de, 4, pa[2], temp8a, temp8b, temp16, dea ); 
    eablen = mult_add( predConsts, ab, 4, pe[2], eb, 4, -pa[2], ea, 4, pb[2], temp8a, temp8b, temp16, eab ); 
    abdlen = mult_add( predConsts, bd, 4, pa[2], da, 4, pb[2], ab, 4, pd[2], temp8a, temp8b, temp16, abd ); 
    bcelen = mult_add( predConsts, ce, 4, pb[2], eb, 4, pc[2], bc, 4, pe[2], temp8a, temp8b, temp16, bce ); 
    cdalen = mult_add( predConsts, da, 4, pc[2], ac, 4, pd[2], cd, 4, pa[2], temp8a, temp8b, temp16, cda ); 
    deblen = mult_add( predConsts, eb, 4, pd[2], bd, 4, pe[2], de, 4, pb[2], temp8a, temp8b, temp16, deb ); 
    eaclen = mult_add( predConsts, ac, 4, pe[2], ce, 4, pa[2], ea, 4, pc[2], temp8a, temp8b, temp16, eac ); 

    ///////////////////////////

    // => deab
    clen = calc_det( predConsts, eab, eablen, deb, deblen, abd, abdlen, dea, dealen, pc, 
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, cdet ); 

    // => eabc
    dlen = calc_det( predConsts, abc, abclen, eac, eaclen, bce, bcelen, eab, eablen, pd, 
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, ddet ); 

    cdlen     = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

    // => abcd
    elen = calc_det( predConsts, bcd, bcdlen, abd, abdlen, cda, cdalen, abc, abclen, pe, 
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, edet ); 

    cd_elen    = d_fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet);

    // => bcde
    alen = calc_det( predConsts, cde, cdelen, bce, bcelen, deb, deblen, bcd, bcdlen, pa, 
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, adet ); 

    // => cdea
    blen = calc_det( predConsts, dea, dealen, cda, cdalen, eac, eaclen, cde, cdelen, pb, 
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, bdet ); 

    ablen     = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

    RealType ret = d_fast_expansion_sum_sign(ablen, abdet, cd_elen, cdedet);

    return ret; 
}

__device__ RealType insphereAdaptExact
(
const RealType*     predConsts, 
RealType*           predData,
const RealType*     pa,
const RealType*     pb,
const RealType*     pc,
const RealType*     pd,
const RealType*     pe,
const RealType      permanent
)
{
    RealType* tempData = predData; 

    // Index into global memory
    RealType* xdet      = tempData; tempData += 96;
    RealType* ydet      = tempData; tempData += 96;
    RealType* zdet      = tempData; tempData += 96;
    RealType* xydet     = tempData; tempData += 192;
    RealType* adet      = tempData; tempData += 288;
    RealType* bdet      = tempData; tempData += 288;
    RealType* abdet     = tempData; tempData += 576;
    RealType* cddet     = tempData; 
    RealType* cdet      = adet; 
    RealType* ddet      = bdet; 

    RealType aex, bex, cex, dex, aey, bey, cey, dey, aez, bez, cez, dez;
    RealType det, errbound;

    RealType aexbey1, bexaey1, bexcey1, cexbey1;
    RealType cexdey1, dexcey1, dexaey1, aexdey1;
    RealType aexcey1, cexaey1, bexdey1, dexbey1;
    RealType aexbey0, bexaey0, bexcey0, cexbey0;
    RealType cexdey0, dexcey0, dexaey0, aexdey0;
    RealType aexcey0, cexaey0, bexdey0, dexbey0;
    RealType ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    RealType ab3, bc3, cd3, da3, ac3, bd3;
    RealType abeps, bceps, cdeps, daeps, aceps, bdeps;
    RealType temp8a[8], temp8b[8], temp16[16], temp24[24];
    int temp24len;

    int alen, blen, clen, dlen;
    int ablen, cdlen;

    RealType aextail, bextail, cextail, dextail;
    RealType aeytail, beytail, ceytail, deytail;
    RealType aeztail, beztail, ceztail, deztail;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    Two_Product(aex, bey, aexbey1, aexbey0);
    Two_Product(bex, aey, bexaey1, bexaey0);
    Two_Two_Diff(aexbey1, aexbey0, bexaey1, bexaey0, ab3, ab[2], ab[1], ab[0]);
    ab[3] = ab3;

    Two_Product(bex, cey, bexcey1, bexcey0);
    Two_Product(cex, bey, cexbey1, cexbey0);
    Two_Two_Diff(bexcey1, bexcey0, cexbey1, cexbey0, bc3, bc[2], bc[1], bc[0]);
    bc[3] = bc3;

    Two_Product(cex, dey, cexdey1, cexdey0);
    Two_Product(dex, cey, dexcey1, dexcey0);
    Two_Two_Diff(cexdey1, cexdey0, dexcey1, dexcey0, cd3, cd[2], cd[1], cd[0]);
    cd[3] = cd3;

    Two_Product(dex, aey, dexaey1, dexaey0);
    Two_Product(aex, dey, aexdey1, aexdey0);
    Two_Two_Diff(dexaey1, dexaey0, aexdey1, aexdey0, da3, da[2], da[1], da[0]);
    da[3] = da3;

    Two_Product(aex, cey, aexcey1, aexcey0);
    Two_Product(cex, aey, cexaey1, cexaey0);
    Two_Two_Diff(aexcey1, aexcey0, cexaey1, cexaey0, ac3, ac[2], ac[1], ac[0]);
    ac[3] = ac3;

    Two_Product(bex, dey, bexdey1, bexdey0);
    Two_Product(dex, bey, dexbey1, dexbey0);
    Two_Two_Diff(bexdey1, bexdey0, dexbey1, dexbey0, bd3, bd[2], bd[1], bd[0]);
    bd[3] = bd3;

    temp24len = mult_add( predConsts, cd, 4, -bez, bd, 4, cez, bc, 4, -dez, temp8a, temp8b, temp16, temp24 ); 
    alen      = calc_det_adapt( predConsts, temp24, temp24len, aex, aey, aez, xdet, ydet, zdet, xydet, adet ); 

    temp24len = mult_add( predConsts, da, 4, cez, ac, 4, dez, cd, 4, aez, temp8a, temp8b, temp16, temp24 ); 
    blen      = calc_det_adapt( predConsts, temp24, temp24len, bex, bey, bez, xdet, ydet, zdet, xydet, bdet ); 

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

    temp24len = mult_add( predConsts, ab, 4, -dez, bd, 4, -aez, da, 4, -bez, temp8a, temp8b, temp16, temp24 ); 
    clen      = calc_det_adapt( predConsts, temp24, temp24len, cex, cey, cez, xdet, ydet, zdet, xydet, cdet ); 

    temp24len = mult_add( predConsts, bc, 4, aez, ac, 4, -bez, ab, 4, cez, temp8a, temp8b, temp16, temp24 ); 
    dlen      = calc_det_adapt( predConsts, temp24, temp24len, dex, dey, dez, xdet, ydet, zdet, xydet, ddet ); 

    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
    det = d_fast_expansion_sum_estimate(ablen, abdet, cdlen, cddet);

    errbound = predConsts[ IsperrboundB ] * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pe[0], aex, aextail);
    Two_Diff_Tail(pa[1], pe[1], aey, aeytail);
    Two_Diff_Tail(pa[2], pe[2], aez, aeztail);
    Two_Diff_Tail(pb[0], pe[0], bex, bextail);
    Two_Diff_Tail(pb[1], pe[1], bey, beytail);
    Two_Diff_Tail(pb[2], pe[2], bez, beztail);
    Two_Diff_Tail(pc[0], pe[0], cex, cextail);
    Two_Diff_Tail(pc[1], pe[1], cey, ceytail);
    Two_Diff_Tail(pc[2], pe[2], cez, ceztail);
    Two_Diff_Tail(pd[0], pe[0], dex, dextail);
    Two_Diff_Tail(pd[1], pe[1], dey, deytail);
    Two_Diff_Tail(pd[2], pe[2], dez, deztail);
    if ((aextail == 0.0) && (aeytail == 0.0) && (aeztail == 0.0)
        && (bextail == 0.0) && (beytail == 0.0) && (beztail == 0.0)
        && (cextail == 0.0) && (ceytail == 0.0) && (ceztail == 0.0)
        && (dextail == 0.0) && (deytail == 0.0) && (deztail == 0.0)) {
            return det;
    }

    errbound = predConsts[ IsperrboundC ] * permanent + predConsts[ Resulterrbound ] * Absolute(det);
    abeps = (aex * beytail + bey * aextail)
        - (aey * bextail + bex * aeytail);
    bceps = (bex * ceytail + cey * bextail)
        - (bey * cextail + cex * beytail);
    cdeps = (cex * deytail + dey * cextail)
        - (cey * dextail + dex * ceytail);
    daeps = (dex * aeytail + aey * dextail)
        - (dey * aextail + aex * deytail);
    aceps = (aex * ceytail + cey * aextail)
        - (aey * cextail + cex * aeytail);
    bdeps = (bex * deytail + dey * bextail)
        - (bey * dextail + dex * beytail);
    det += (((bex * bex + bey * bey + bez * bez)
        * ((cez * daeps + dez * aceps + aez * cdeps)
        + (ceztail * da3 + deztail * ac3 + aeztail * cd3))
        + (dex * dex + dey * dey + dez * dez)
        * ((aez * bceps - bez * aceps + cez * abeps)
        + (aeztail * bc3 - beztail * ac3 + ceztail * ab3)))
        - ((aex * aex + aey * aey + aez * aez)
        * ((bez * cdeps - cez * bdeps + dez * bceps)
        + (beztail * cd3 - ceztail * bd3 + deztail * bc3))
        + (cex * cex + cey * cey + cez * cez)
        * ((dez * abeps + aez * bdeps + bez * daeps)
        + (deztail * ab3 + aeztail * bd3 + beztail * da3))))
        + 2.0 * (((bex * bextail + bey * beytail + bez * beztail)
        * (cez * da3 + dez * ac3 + aez * cd3)
        + (dex * dextail + dey * deytail + dez * deztail)
        * (aez * bc3 - bez * ac3 + cez * ab3))
        - ((aex * aextail + aey * aeytail + aez * aeztail)
        * (bez * cd3 - cez * bd3 + dez * bc3)
        + (cex * cextail + cey * ceytail + cez * ceztail)
        * (dez * ab3 + aez * bd3 + bez * da3)));

    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return insphereExact( predConsts, predData, pa, pb, pc, pd, pe ); 
}

__forceinline__ __device__ RealType insphereFastAdaptExact
(
const RealType* predConsts, 
RealType*       predData,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
const RealType* pe
)
{
    RealType aex, bex, cex, dex;
    RealType aey, bey, cey, dey;
    RealType aez, bez, cez, dez;
    RealType aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
    RealType aexcey, cexaey, bexdey, dexbey;
    RealType alift, blift, clift, dlift;
    RealType ab, bc, cd, da, ac, bd;
    RealType abc, bcd, cda, dab;
    RealType aezplus, bezplus, cezplus, dezplus;
    RealType aexbeyplus, bexaeyplus, bexceyplus, cexbeyplus;
    RealType cexdeyplus, dexceyplus, dexaeyplus, aexdeyplus;
    RealType aexceyplus, cexaeyplus, bexdeyplus, dexbeyplus;
    RealType det;
    RealType permanent, errbound;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    aexbey  = aex * bey;
    bexaey  = bex * aey;
    ab      = aexbey - bexaey;
    bexcey  = bex * cey;
    cexbey  = cex * bey;
    bc      = bexcey - cexbey;
    cexdey  = cex * dey;
    dexcey  = dex * cey;
    cd      = cexdey - dexcey;
    dexaey  = dex * aey;
    aexdey  = aex * dey;
    da      = dexaey - aexdey;

    aexcey  = aex * cey;
    cexaey  = cex * aey;
    ac      = aexcey - cexaey;
    bexdey  = bex * dey;
    dexbey  = dex * bey;
    bd      = bexdey - dexbey;

    abc = aez * bc - bez * ac + cez * ab;
    bcd = bez * cd - cez * bd + dez * bc;
    cda = cez * da + dez * ac + aez * cd;
    dab = dez * ab + aez * bd + bez * da;

    alift = aex * aex + aey * aey + aez * aez;
    blift = bex * bex + bey * bey + bez * bez;
    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;

    det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    aezplus = Absolute(aez);
    bezplus = Absolute(bez);
    cezplus = Absolute(cez);
    dezplus = Absolute(dez);
    aexbeyplus = Absolute(aexbey);
    bexaeyplus = Absolute(bexaey);
    bexceyplus = Absolute(bexcey);
    cexbeyplus = Absolute(cexbey);
    cexdeyplus = Absolute(cexdey);
    dexceyplus = Absolute(dexcey);
    dexaeyplus = Absolute(dexaey);
    aexdeyplus = Absolute(aexdey);
    aexceyplus = Absolute(aexcey);
    cexaeyplus = Absolute(cexaey);
    bexdeyplus = Absolute(bexdey);
    dexbeyplus = Absolute(dexbey);
    permanent = ((cexdeyplus + dexceyplus) * bezplus
        + (dexbeyplus + bexdeyplus) * cezplus
        + (bexceyplus + cexbeyplus) * dezplus)
        * alift
        + ((dexaeyplus + aexdeyplus) * cezplus
        + (aexceyplus + cexaeyplus) * dezplus
        + (cexdeyplus + dexceyplus) * aezplus)
        * blift
        + ((aexbeyplus + bexaeyplus) * dezplus
        + (bexdeyplus + dexbeyplus) * aezplus
        + (dexaeyplus + aexdeyplus) * bezplus)
        * clift
        + ((bexceyplus + cexbeyplus) * aezplus
        + (cexaeyplus + aexceyplus) * bezplus
        + (aexbeyplus + bexaeyplus) * cezplus)
        * dlift;

    errbound = predConsts[ IsperrboundA ] * permanent;

    if ((det > errbound) || (-det > errbound))
        return det;

    return insphereAdaptExact( predConsts, predData, pa, pb, pc, pd, pe, permanent );   // Needs exact predicate
}

__forceinline__ __device__ RealType insphereFast
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
const RealType* pe
)
{
    RealType aex, bex, cex, dex;
    RealType aey, bey, cey, dey;
    RealType aez, bez, cez, dez;
    RealType aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
    RealType aexcey, cexaey, bexdey, dexbey;
    RealType alift, blift, clift, dlift;
    RealType ab, bc, cd, da, ac, bd;
    RealType abplus, bcplus, cdplus, daplus, acplus, bdplus;
    RealType aezplus, bezplus, cezplus, dezplus;
    RealType det;
    RealType permanent, errbound;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    alift = aex * aex + aey * aey + aez * aez;
    blift = bex * bex + bey * bey + bez * bez;
    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;

    aexbey      = aex * bey;
    bexaey      = bex * aey;
    abplus      = Absolute(aexbey) + Absolute(bexaey); 
    ab          = aexbey - bexaey;

    bexcey      = bex * cey;
    cexbey      = cex * bey;
    bcplus      = Absolute(bexcey) + Absolute(cexbey);
    bc          = bexcey - cexbey;

    cexdey      = cex * dey;
    dexcey      = dex * cey;
    cdplus      = Absolute(cexdey) + Absolute(dexcey); 
    cd          = cexdey - dexcey;

    dexaey      = dex * aey;
    aexdey      = aex * dey;
    daplus      = Absolute(dexaey) + Absolute(aexdey); 
    da          = dexaey - aexdey;

    aexcey      = aex * cey;
    cexaey      = cex * aey;
    acplus      = Absolute(aexcey) + Absolute(cexaey);
    ac          = aexcey - cexaey;

    bexdey      = bex * dey;
    dexbey      = dex * bey;
    bdplus      = Absolute(bexdey) + Absolute(dexbey); 
    bd          = bexdey - dexbey;

    det = ( cd * blift - bd * clift + bc * dlift ) * aez
        + (-cd * alift - da * clift - ac * dlift ) * bez
        + ( bd * alift + da * blift + ab * dlift ) * cez
        + (-bc * alift + ac * blift - ab * clift ) * dez; 

    aezplus = Absolute(aez);
    bezplus = Absolute(bez);
    cezplus = Absolute(cez);
    dezplus = Absolute(dez);
    permanent = ( cdplus * blift + bdplus * clift + bcplus * dlift ) * aezplus 
        + ( cdplus * alift + daplus * clift + acplus * dlift ) * bezplus 
        + ( bdplus * alift + daplus * blift + abplus * dlift ) * cezplus
        + ( bcplus * alift + acplus * blift + abplus * clift ) * dezplus; 

    // [IsperrboundA]
    errbound = predConsts[ IsperrboundA ] * permanent;

    if ((det > errbound) || (-det > errbound))
    {
        return det;
    }

    return 0;   // Needs exact predicate
}

__forceinline__ __device__ float insphereDet
(
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
const RealType* pe
)
{
    float aex, bex, cex, dex;
    float aey, bey, cey, dey;
    float aez, bez, cez, dez;
    float aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
    float aexcey, cexaey, bexdey, dexbey;
    float alift, blift, clift, dlift;
    float ab, bc, cd, da, ac, bd;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    alift = aex * aex + aey * aey + aez * aez;
    blift = bex * bex + bey * bey + bez * bez;
    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;

    aexbey      = aex * bey;
    bexaey      = bex * aey;
    ab          = aexbey - bexaey;

    bexcey      = bex * cey;
    cexbey      = cex * bey;
    bc          = bexcey - cexbey;

    cexdey      = cex * dey;
    dexcey      = dex * cey;
    cd          = cexdey - dexcey;

    dexaey      = dex * aey;
    aexdey      = aex * dey;
    da          = dexaey - aexdey;

    aexcey      = aex * cey;
    cexaey      = cex * aey;
    ac          = aexcey - cexaey;

    bexdey      = bex * dey;
    dexbey      = dex * bey;
    bd          = bexdey - dexbey;

    return ( cd * blift - bd * clift + bc * dlift ) * aez
        + (-cd * alift - da * clift - ac * dlift ) * bez
        + ( bd * alift + da * blift + ab * dlift ) * cez
        + (-bc * alift + ac * blift - ab * clift ) * dez; 
}

///////////////////////////////////////////////////////////////////// Lifted //

// det  = ( pa[0]^2 + pa[1]^2 + pa[2]^2 ) - ( pb[0]^2 + pb[1]^2 + pc[1]^2 )
__device__ RealType orient1dExact_Lifted
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb, 
bool            lifted
)
{
    if (!lifted) 
        return (pa[0] - pb[0]); 

    RealType axax1, ayay1, azaz1, bxbx1, byby1, bzbz1;
    RealType axax0, ayay0, azaz0, bxbx0, byby0, bzbz0;
    RealType aterms[4], bterms[4], cterms[4];
    RealType aterms3, bterms3, cterms3;
    RealType v[8];
    int vlength;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pa[0], axax1, axax0);
    Two_Product(pb[0], pb[0], bxbx1, bxbx0);
    Two_Two_Diff(axax1, axax0, bxbx1, bxbx0,
        aterms3, aterms[2], aterms[1], aterms[0]);
    aterms[3] = aterms3;

    Two_Product(pa[1], pa[1], ayay1, ayay0);
    Two_Product(pb[1], pb[1], byby1, byby0);
    Two_Two_Diff(ayay1, ayay0, byby1, byby0,
        bterms3, bterms[2], bterms[1], bterms[0]);
    bterms[3] = bterms3;

    Two_Product(pa[2], pa[2], azaz1, azaz0);
    Two_Product(pb[2], pb[2], bzbz1, bzbz0);
    Two_Two_Diff(azaz1, azaz0, bzbz1, bzbz0,
        cterms3, cterms[2], cterms[1], cterms[0]);
    cterms[3] = cterms3;

    vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);

    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

__device__ RealType orient2dExact_Lifted
(
const RealType* predConsts, 
RealType*       predData,
const RealType* pa,
const RealType* pb,
const RealType* pc,
bool            lifted
)
{
    RealType* aterms = predData; predData += 24;
    RealType* bterms = predData; predData += 24;
    RealType* cterms = predData; predData += 24;
    RealType* v      = predData; 

    RealType aax1, aax0, aay1, aay0, aaz[2]; 
    RealType temp[4]; 
    RealType palift[6], pblift[6], pclift[6]; 
    RealType xy1terms[12], xy2terms[12]; 

    int palen, pblen, pclen; 
    int xy1len, xy2len; 
    int alen, blen, clen; 
    int vlen; 

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    // Compute the lifted coordinate
    if (lifted) 
    {
        Two_Product(pa[0], pa[0], aax1, aax0); 
        Two_Product(-pa[1], pa[1], aay1, aay0); 
        Two_Product(pa[2], pa[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        palen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);

        Two_Product(pb[0], pb[0], aax1, aax0); 
        Two_Product(-pb[1], pb[1], aay1, aay0); 
        Two_Product(pb[2], pb[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        pblen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);

        Two_Product(pc[0], pc[0], aax1, aax0); 
        Two_Product(-pc[1], pc[1], aay1, aay0); 
        Two_Product(pc[2], pc[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        pclen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);
    }
    else {
        palen = 1; palift[0] = pa[1]; 
        pblen = 1; pblift[0] = pb[1]; 
        pclen = 1; pclift[0] = pc[1]; 
    }

    // Compute the determinant as usual
    xy1len = d_scale_expansion_zeroelim(predConsts, pblen, pblift, pa[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pclen, pclift, -pa[0], xy2terms);
    alen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, aterms);

    xy1len = d_scale_expansion_zeroelim(predConsts, pclen, pclift, pb[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, palen, palift, -pb[0], xy2terms);
    blen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bterms);

    xy1len = d_scale_expansion_zeroelim(predConsts, palen, palift, pc[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pblen, pblift, -pc[0], xy2terms);
    clen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cterms);

    vlen = d_fast_expansion_sum_zeroelim(alen, aterms, blen, bterms, v);

    return d_fast_expansion_sum_sign(vlen, v, clen, cterms);
}

__device__ RealType orient3dExact_Lifted
(
const RealType* predConsts, 
RealType*       predData,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
bool            lifted
)
{
    // Index into global memory
    RealType* ab     = predData; predData += 24;
    RealType* bc     = predData; predData += 24;
    RealType* cd     = predData; predData += 24;
    RealType* da     = predData; predData += 24;
    RealType* ac     = predData; predData += 24;
    RealType* bd     = predData; predData += 24;
    RealType* temp48 = predData; predData += 48;
    RealType* cda    = predData; predData += 72;
    RealType* dab    = predData; predData += 72;
    RealType* abc    = predData; predData += 72;
    RealType* bcd    = predData; predData += 72;
    RealType* adet   = predData; predData += 144;
    RealType* bdet   = predData; predData += 144;
    RealType* cdet   = predData; predData += 144;
    RealType* ddet   = predData; predData += 144;
    RealType* abdet  = predData; predData += 288;
    RealType* cddet  = predData; 

    RealType aax1, aax0, aay1, aay0, aaz[2]; 
    RealType temp[4]; 
    RealType palift[6], pblift[6], pclift[6], pdlift[6]; 
    RealType xy1terms[12], xy2terms[12]; 

    int templen; 
    int palen, pblen, pclen, pdlen; 
    int xy1len, xy2len; 
    int ablen, bclen, cdlen, dalen, aclen, bdlen; 
    int cdalen, dablen, abclen, bcdlen; 
    int alen, blen, clen, dlen; 

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    // Compute the lifted coordinate
    if (lifted) {
        Two_Product(pa[0], pa[0], aax1, aax0); 
        Two_Product(-pa[1], pa[1], aay1, aay0); 
        Two_Product(pa[2], pa[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        palen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);

        Two_Product(pb[0], pb[0], aax1, aax0); 
        Two_Product(-pb[1], pb[1], aay1, aay0); 
        Two_Product(pb[2], pb[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        pblen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);

        Two_Product(pc[0], pc[0], aax1, aax0); 
        Two_Product(-pc[1], pc[1], aay1, aay0); 
        Two_Product(pc[2], pc[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        pclen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);

        Two_Product(pd[0], pd[0], aax1, aax0); 
        Two_Product(-pd[1], pd[1], aay1, aay0); 
        Two_Product(pd[2], pd[2], aaz[1], aaz[0]); 
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
        pdlen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pdlift);
    }
    else
    {
        palen = 1; palift[0] = pa[2]; 
        pblen = 1; pblift[0] = pb[2]; 
        pclen = 1; pclift[0] = pc[2]; 
        pdlen = 1; pdlift[0] = pd[2]; 
    }

    // Compute the determinant as usual
    xy1len = d_scale_expansion_zeroelim(predConsts, pblen, pblift, pa[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, palen, palift, -pb[1], xy2terms);
    ablen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ab);

    xy1len = d_scale_expansion_zeroelim(predConsts, pclen, pclift, pb[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pblen, pblift, -pc[1], xy2terms);
    bclen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bc);

    xy1len = d_scale_expansion_zeroelim(predConsts, pdlen, pdlift, pc[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pclen, pclift, -pd[1], xy2terms);
    cdlen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cd);

    xy1len = d_scale_expansion_zeroelim(predConsts, palen, palift, pd[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pdlen, pdlift, -pa[1], xy2terms);
    dalen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, da);

    xy1len = d_scale_expansion_zeroelim(predConsts, pclen, pclift, pa[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, palen, palift, -pc[1], xy2terms);
    aclen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ac);

    xy1len = d_scale_expansion_zeroelim(predConsts, pdlen, pdlift, pb[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(predConsts, pblen, pblift, -pd[1], xy2terms);
    bdlen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bd);

    templen = d_fast_expansion_sum_zeroelim(cdlen, cd, dalen, da, temp48);
    cdalen = d_fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, cda);
    templen = d_fast_expansion_sum_zeroelim(dalen, da, ablen, ab, temp48);
    dablen = d_fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, dab);

    for (int i = 0; i < bdlen; i++) 
        bd[i] = -bd[i];
    for (int i = 0; i < aclen; i++) 
        ac[i] = -ac[i];

    templen = d_fast_expansion_sum_zeroelim(ablen, ab, bclen, bc, temp48);
    abclen = d_fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, abc);
    templen = d_fast_expansion_sum_zeroelim(bclen, bc, cdlen, cd, temp48);
    bcdlen = d_fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, bcd);

    alen = d_scale_expansion_zeroelim(predConsts, bcdlen, bcd, pa[0], adet);
    blen = d_scale_expansion_zeroelim(predConsts, cdalen, cda, -pb[0], bdet);
    clen = d_scale_expansion_zeroelim(predConsts, dablen, dab, pc[0], cdet);
    dlen = d_scale_expansion_zeroelim(predConsts, abclen, abc, -pd[0], ddet);

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

    return d_fast_expansion_sum_sign(ablen, abdet, cdlen, cddet);
}

__device__ RealType orient3dFastExact_Lifted
(
const RealType* predConsts, 
RealType*       predData,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd,
bool            lifted
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType det;
    RealType permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2]; 
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    if ( lifted ) 
    {
        adz = adx * adx + ady * ady + adz * adz; 
        bdz = bdx * bdx + bdy * bdy + bdz * bdz; 
        cdz = cdx * cdx + cdy * cdy + cdz * cdz; 
    }

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy) 
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute( adz )
        + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute( bdz )
        + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute( cdz );

    if ( lifted ) 
        errbound = predConsts[ O3derrboundAlifted ] * permanent;
    else
        errbound = predConsts[ O3derrboundA ] * permanent;

    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return orient3dExact_Lifted( predConsts, predData, pa, pb, pc, pd, lifted ); 
}

////////////////////////////////////////////////////////////////////////////////
