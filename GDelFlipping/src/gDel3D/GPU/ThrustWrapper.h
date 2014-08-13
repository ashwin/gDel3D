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

Neither the name of the National University of University nor the names of its contributors
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

#include "../CommonTypes.h"
#include "GPUDecl.h"

////
// Device iterators
////

typedef IntDVec::iterator                       IntDIter;
typedef thrust::tuple< int, int >               IntTuple2;
typedef thrust::tuple< IntDIter, IntDIter >     IntDIterTuple2;
typedef thrust::zip_iterator< IntDIterTuple2 >  IntZipDIter;

////
// Functions
////
void thrust_free_all();

void thrust_sort_by_key
(
DevVector<int>::iterator keyBeg, 
DevVector<int>::iterator keyEnd, 
thrust::zip_iterator< 
    thrust::tuple< 
        DevVector<int>::iterator,
        DevVector<Point3>::iterator > > valueBeg
)
;
void thrust_transform_GetMortonNumber
(
DevVector<Point3>::iterator inBeg, 
DevVector<Point3>::iterator inEnd, 
DevVector<int>::iterator    outBeg,
RealType                    minVal, 
RealType                    maxVal
)
;
int makeInPlaceIncMapAndSum
( 
IntDVec& inVec 
)
;
int compactIfNegative
( 
DevVector<int>& inVec 
)
;
int compactIfNegative
( 
DevVector<int>& inVec,
DevVector<int>& temp 
)
;
void compactBothIfNegative
( 
IntDVec& vec0, 
IntDVec& vec1 
)
;
int thrust_copyIf_IsActiveTetra
(
const CharDVec& inVec,
IntDVec&        outVec
)
;
int thrust_copyIf_Insertable
(
const IntDVec& stencil,
IntDVec&       outVec
)
;
////
// Thrust helper functors
////

struct GetMortonNumber
{
    RealType _minVal, _range; 

    GetMortonNumber( RealType minVal, RealType maxVal ) 
        : _minVal( minVal ), _range( maxVal - minVal ) {}

    // Note: No performance benefit by changing by-reference to by-value here
    // Note: No benefit by making this __forceinline__
    __host__ __device__ int operator () ( const Point3& point ) const
    {
        const int Gap16 = 0x030000FF;   // Creates 16-bit gap between value bits
        const int Gap08 = 0x0300F00F;   // ... and so on ...
        const int Gap04 = 0x030C30C3;   // ...
        const int Gap02 = 0x09249249;   // ...

        const int minInt = 0x0; 
        const int maxInt = 0x3ff; 
        
        int mortonNum = 0; 

        // Iterate coordinates of point
        for ( int vi = 0; vi < 3; ++vi )
        {
            // Read
            int v = int( ( point._p[ vi ] - _minVal ) / _range * 1024.0 ); 

            if ( v < minInt ) 
                v = minInt; 

            if ( v > maxInt ) 
                v = maxInt; 

            // Create 2-bit gaps between the 10 value bits
            // Ex: 1001001001001001001001001001
            v = ( v | ( v << 16 ) ) & Gap16;
            v = ( v | ( v <<  8 ) ) & Gap08;
            v = ( v | ( v <<  4 ) ) & Gap04;
            v = ( v | ( v <<  2 ) ) & Gap02;

            // Interleave bits of x-y-z coordinates
            mortonNum |= ( v << vi ); 
        }

        return mortonNum;
    }
};

struct IsNegative
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x < 0 );
    }
};

struct IsFlip23
{
    __host__ __device__ bool operator() ( const int val )
    {
        return ( ( ( val >> 2 ) & 0x3 ) == 3 );
    }
};

struct IsNotNegative
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x >= 0 );
    }
};

// Check if first value in tuple2 is negative
struct IsIntTuple2Negative
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int x = thrust::get<0>( tup );
        return ( x < 0 );
    }
};

// Check if tetra is active
struct IsTetActive
{
    __host__ __device__ bool operator() ( char tetInfo )
    {
        return ( isTetAlive( tetInfo ) && ( Changed == getTetCheckState( tetInfo ) ) );
    }
};

// Check if tetra is alive
struct TetAliveStencil
{
    __host__ __device__ int operator() ( char tetInfo )
    {
        return isTetAlive( tetInfo ) ? 1 : 0;
    }
};
