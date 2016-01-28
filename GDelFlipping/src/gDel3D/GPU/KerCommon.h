/*
Author: Cao Thanh Tung, Ashwin Nanjappa
Date:   05-Aug-2014

===============================================================================

Copyright (c) 2011, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/gdel3d.html

If you use gDel3D and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of Singapore nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#include <device_launch_parameters.h>
#include <device_functions.h>
#undef __CUDACC__
#endif

#include "GPUDecl.h"

#define BLOCK_DIM    blockDim.x
#define THREAD_IDX   threadIdx.x

const int MeanVertDegree = 8; 

__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock   = blockDim.x;
    const int curThreadIdx      = ( blockIdx.x * threadsPerBlock ) + threadIdx.x;
    return curThreadIdx;
}

__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid     = gridDim.x;
    const int threadsPerBlock   = blockDim.x;
    const int threadNum         = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

/////////////////////////////////////////////// Some helper functions
#ifdef REAL_TYPE_FP32
struct __align__(8) RealType2 
{
    float _c0, _c1; 
};
#else
struct __align__(16) RealType2 
{
    double _c0, _c1; 
};
#endif

__forceinline__ __device__ 
Point3 loadPoint3( const Point3 *pointArr, int idx ) 
{
    if ( idx % 2 == 0 ) {
        const RealType2 ptxy = ( ( RealType2 *) pointArr )[ idx * 3 / 2 ]; 
        const Point3 pt = {
            ptxy._c0,
            ptxy._c1,
            pointArr[ idx ]._p[2] 
        }; 

        return pt; 
    } 
    else { 
        const RealType2 ptyz = ( ( RealType2 *) pointArr )[ idx * 3 / 2 + 1 ]; 
        const Point3 pt = {
            pointArr[ idx ]._p[0],
            ptyz._c0,
            ptyz._c1 
        }; 
        
        return pt; 
    }
}

__forceinline__ __device__ 
Tet loadTet( Tet* tetArr, int idx )
{
    int4 temp   = ( ( int4 * ) tetArr )[ idx ]; 
    Tet tet     = { temp.x, temp.y, temp.z, temp.w }; 

    return tet; 
}

__forceinline__ __device__ 
TetOpp loadOpp( TetOpp* oppArr, int idx )
{
    int4 temp   = ( ( int4 * ) oppArr )[ idx ]; 
    TetOpp opp  = { temp.x, temp.y, temp.z, temp.w }; 

    return opp; 
}

__forceinline__ __device__ 
void storeTet( Tet* tetArr, int idx, const Tet tet )
{
    int4 temp = { tet._v[ 0 ], tet._v[ 1 ], tet._v[ 2 ], tet._v[ 3 ] }; 

    ( ( int4 * ) tetArr )[ idx ] = temp; 
}

__forceinline__ __device__ 
void storeOpp( TetOpp* oppArr, int idx, const TetOpp opp )
{
    int4 temp = { opp._t[ 0 ], opp._t[ 1 ], opp._t[ 2 ], opp._t[ 3 ] }; 
        
    ( ( int4 * ) oppArr )[ idx ] = temp; 
}

__forceinline__ __device__ 
FlipItem loadFlip( FlipItem* flipArr, int idx )
{
    int4 t1 = ( ( int4 * ) flipArr )[ idx * 2 + 0 ]; 
    int4 t2 = ( ( int4 * ) flipArr )[ idx * 2 + 1 ]; 

    FlipItem flip = { t1.x, t1.y, t1.z, t1.w, t2.x, t2.y, t2.z, t2.w }; 

    return flip; 
}

__forceinline__ __device__ 
FlipItemTetIdx loadFlipTetIdx( FlipItem* flipArr, int idx )
{
    int4 t = ( ( int4 * ) flipArr )[ idx * 2 + 1 ]; 

    FlipItemTetIdx flip = { t.y, t.z, t.w }; 

    return flip; 
}

__forceinline__ __device__ 
void storeFlip( FlipItem* tetArr, int idx, const FlipItem flip )
{
    int4 t1 = { flip._v[ 0 ], flip._v[ 1 ], flip._v[ 2 ], flip._v[ 3 ] }; 
    int4 t2 = { flip._v[ 4 ], flip._t[ 0 ], flip._t[ 1 ], flip._t[ 2 ] }; 

    ( ( int4 * ) tetArr )[ idx * 2 + 0 ] = t1; 
    ( ( int4 * ) tetArr )[ idx * 2 + 1 ] = t2; 
}

__forceinline__ __host__ __device__ 
int roundUp( int num, int div ) 
{
    return ( ( num + div - 1 ) / div ) * div; 
}

// Escape -1 (special value)
__forceinline__ __device__
int makePositive( int v )
{
    CudaAssert( v < 0 );
    return -( v + 2 );
}

// Escape -1 (special value)
__forceinline__ __device__
int makeNegative( int v )
{
    CudaAssert( v >= 0 );
    return -( v + 2 );
}

/////////////////////////////////////////////////////////////////////// Vote //
__forceinline__ __device__ int makeVoteVal( int tetIdx, char flipInfo )
{
    return ( tetIdx << 4 ) | ( flipInfo & 0x0F );
}

__forceinline__ __device__ int getVoteTetIdx( int voteVal )
{
    return ( voteVal >> 4 );
}

__forceinline__ __device__ char getVoteFlipInfo( int voteVal )
{
    return ( voteVal & 0x0F );
}
