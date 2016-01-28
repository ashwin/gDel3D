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

// STL
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <cfloat>

// CUDA
#include "GPU/CudaWrapper.h"

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#ifdef REAL_TYPE_FP32
typedef float RealType;
#else
typedef double RealType;
#endif

typedef unsigned char uchar;

///////////////////////////////////////////////////////////////////// Orient //
enum Orient
{
    OrientNeg   = -1,
    OrientZero  = +0,
    OrientPos   = +1
};

__forceinline__ __host__ __device__ Orient flipOrient( Orient ord )
{
    //assert( OrientZero != ord );
    return (ord == OrientPos) ? OrientNeg : ( ord == OrientNeg ? OrientPos : OrientZero ); 
}

// Our orientation is opposite of Shewchuk
__forceinline__ __host__ __device__ Orient ortToOrient( RealType det )
{
    return ( det < 0 ) ? OrientPos : ( ( det > 0 ) ? OrientNeg : OrientZero );
}

__forceinline__ __host__ __device__ Orient sphToOrient( RealType det )
{
    return ( det > 0 ) ? OrientPos : ( ( det < 0 ) ? OrientNeg : OrientZero );
}

/////////////////////////////////////////////////////////////////////// Side //
enum Side
{
    SideIn   = -1,
    SideZero = +0,
    SideOut  = +1
};

// Our orientation is defined opposite of Shewchuk
// For us, 0123 means seen from 3, 012 are in CCW order
// Given this orientation, Shewchuk's insphere value will
// be +ve if point is *outside* sphere and vice versa.
__forceinline__ __host__ __device__ Side sphToSide( RealType det )
{
    return ( det > 0 ) ? SideOut : ( ( det < 0 ) ? SideIn : SideZero );
}

////////////////////////////////////////////////////////////////////// Constants
// Order of 3 vertices as seen from one vertex of tetra
__device__ const int TetViAsSeenFrom[4][3] = {
    { 1, 3, 2 }, // From 0
    { 0, 2, 3 }, // From 1
    { 0, 3, 1 }, // From 2
    { 0, 1, 2 }, // Default view is from "3"
};

__device__ const int TetNextViAsSeenFrom[4][4] = {
    { -1, 0, 2, 1 }, // From 0
    { 0, -1, 1, 2 }, // From 1
    { 0, 2, -1, 1 }, // From 2
    { 0, 1, 2, -1 }, // From 3
}; 

////////////////////////////////////////////////////////////////////// Helper functions
template< typename T > 
__forceinline__ __host__ __device__ T min( T a, T b ) 
{
    if ( a < b ) 
        return a; 
    else
        return b; 
}


template< typename T > 
__forceinline__ __host__ __device__ bool isBitSet( T c, int bit )
{
    return ( 1 == ( ( c >> bit ) & 1 ) );
}

template< typename T >
__forceinline__ __host__ __device__ void setBitState( T& c, int bit, bool state )
{
    const T val = ( 1 << bit );
    c = state
        ? ( c | val )
        : ( c & ~val );
}

////////////////////////////////////////////////////////////////////// Points
struct Point3
{
    RealType _p[ 3 ];

    bool lessThan( const Point3& pt ) const
    {
        if ( _p[0] < pt._p[0] ) return true; 
        if ( _p[0] > pt._p[0] ) return false; 
        if ( _p[1] < pt._p[1] ) return true; 
        if ( _p[1] > pt._p[1] ) return false; 
        if ( _p[2] < pt._p[2] ) return true; 

        return false; 
    }

    bool operator < ( const Point3& pt ) const
    {
        return lessThan( pt );
    }

    void printOut() const
    {
        std::cout << _p[0] << " " << _p[1] << " " << _p[2] << std::endl;
    }
};

//////////////////////////////////////////////////////////////////////// Tet //
struct Tet
{
    int _v[4];

    __forceinline__ __host__ __device__ bool has( int v ) const
    {
        return ( _v[0] == v || _v[1] == v || _v[2] == v || _v[3] == v );
    }

    // Assumption: it is there!
    __forceinline__ __host__ __device__ int getIndexOf( int v ) const
    {
        if ( _v[0] == v ) return 0;
        if ( _v[1] == v ) return 1;
        if ( _v[2] == v ) return 2;
        if ( _v[3] == v ) return 3;

        // CUDA 5.5 compiler bug! An assert is required. 
#ifdef __CUDA_ARCH__
        CudaAssert( false );
#else
        assert( false );
#endif
        return -1;
    }

    __forceinline__ __host__ __device__ int minIdx() const
    {
        int idx1 = min( _v[0], _v[1] ); 
        int idx2 = min( _v[2], _v[3] ); 

        return min( idx1, idx2 ); 
    }
};

__forceinline__ __host__ __device__ Tet makeTet( int v0, int v1, int v2, int v3 ) 
{
    const Tet newTet = { v0, v1, v2, v3 }; 

    return newTet; 
}

__forceinline__ bool isValidTetVi( int vi )
{
    return ( vi >= 0 && vi < 4 );
}

///////////////////////////////////////////////////////////////////// TetOpp //
// ...76543210
//       ^^^^^ vi (2 bits)
//       |||__ internal
//       ||___ special
//       |____ sphere fail
// Rest is tetIdx

__forceinline__ __host__ __device__ int getOppValTet( int val )
{
    return ( val >> 5 );
}

__forceinline__ __host__ __device__ void setOppValTet( int &val, int idx )
{
    val = ( val & 0x1f ) | ( idx << 5 ); 
}

__forceinline__ __host__ __device__ int getOppValVi( int val )
{
    return ( val & 3 );
}

__forceinline__ __host__ __device__ int makeOppVal( int tetIdx, int oppTetVi ) 
{
    return ( tetIdx << 5 ) | oppTetVi;
}

struct TetOpp
{
    int _t[4];

    __forceinline__ __host__ __device__ void setOpp( int vi, int tetIdx, int oppTetVi ) 
    {
        _t[ vi ] = makeOppVal( tetIdx, oppTetVi );
    }

    __forceinline__ __host__ __device__ void setOppInternal( int vi, int tetIdx, int oppTetVi ) 
    {
        _t[ vi ] = ( tetIdx << 5 ) | ( 1 << 2 ) | oppTetVi;
    }

    __forceinline__ __host__ __device__ void setOppInternal( int vi ) 
    {
        setBitState( _t[ vi ], 2, true ); 
    }

    __forceinline__ __host__ __device__ void setOppSpecial( int vi, bool state ) 
    {
        setBitState( _t[ vi ], 3, state ); 
    }

    __forceinline__ __host__ __device__ bool isNeighbor( int tetIdx ) const
    {
        return ( (_t[0] >> 5) == tetIdx ||
                 (_t[1] >> 5) == tetIdx ||
                 (_t[2] >> 5) == tetIdx ||
                 (_t[3] >> 5) == tetIdx ); 
    }

    __forceinline__ __host__ __device__ int getIdxOf( int tetIdx ) const
    {
        if ( ( _t[0] >> 5 ) == tetIdx ) return 0;
        if ( ( _t[1] >> 5 ) == tetIdx ) return 1;
        if ( ( _t[2] >> 5 ) == tetIdx ) return 2;
        if ( ( _t[3] >> 5 ) == tetIdx ) return 3;
        return -1;
    }

    __forceinline__ __host__ __device__ bool isOppSpecial( int vi ) const
    {
        return isBitSet( _t[ vi ], 3 ); 
    }

    __forceinline__ __host__ __device__ int getOppTetVi( int vi ) const
    {
        if ( -1 == _t[ vi ] ) 
            return -1; 
        
        return ( _t[ vi ] & (~(7 << 2)) );
    }

    __forceinline__ __host__ __device__ bool isOppInternal( int vi ) const
    {
        return isBitSet( _t[ vi ], 2 ); 
    }

    __forceinline__ __host__ __device__ int getOppTet( int vi ) const
    {
        return getOppValTet( _t[ vi ] );
    }

    __forceinline__ __host__ __device__ void setOppTet( int vi, int idx ) 
    {
        return setOppValTet( _t[ vi ], idx );
    }

    __forceinline__ __host__ __device__ int getOppVi( int vi ) const
    {
        return getOppValVi( _t[ vi ] );
    }

    __forceinline__ __host__ __device__ void setOppSphereFail( int vi )
    {
        setBitState( _t[ vi ], 4, true ); 
    }

    __forceinline__ __host__ __device__ bool isOppSphereFail( int vi ) const
    {
        return isBitSet( _t[ vi ], 4 ); 
    }
};

///////////////////////////////////////////////// Flip //
enum FlipType {
    Flip32,
    Flip23,
    FlipNone
};

// corOrdVi-botCorVi-botVi-1 (2-bits each)
// flipType = 0 : 2-3; 1 : 3-2
__forceinline__ __device__
char makeFlip( int botVi, int botCorOrdVi )
{
    return ( botVi  | ( botCorOrdVi << 2 ) );
}

__forceinline__ __device__ int getFlipBotVi( char c )       
{ return ( c & 0x3 ); }

__forceinline__ __device__ int getFlipBotCorOrdVi( char c ) 
{ return ( ( c >> 2 ) & 0x3 ); }

__forceinline__ __device__ FlipType getFlipType( char c )
{
    if ( getFlipBotCorOrdVi( c ) == 3 ) return Flip23;
    return Flip32;
}

//////////////////////////////////////////////////////////////////// TetInfo //
// Tet info
// 76543210
//      ^^^ 0: Dead      1: Alive
//      ||_ 0: Checked   1: Changed
//      |__ 0: NotInStar 1: InStar

enum TetCheckState
{
    Checked,
    Changed,
};

__forceinline__ __host__ __device__ bool isTetAlive( char c )
{
    return isBitSet( c, 0 );
}

__forceinline__ __host__ __device__ void setTetAliveState( char& c, bool b )
{
    setBitState( c, 0, b );
}

__forceinline__ __host__ __device__ bool isTetEmpty( char c )
{
    return isBitSet( c, 2 );
}

__forceinline__ __host__ __device__ void setTetEmptyState( char& c, bool b )
{
    setBitState( c, 2, b );
}

__forceinline__ __host__ __device__ TetCheckState getTetCheckState( char c )
{
    return isBitSet( c, 1 ) ? Changed : Checked;
}

__forceinline__ __host__ __device__ void setTetCheckState( char& c, TetCheckState s )
{
    if ( Checked == s ) setBitState( c, 1, false );
    else                setBitState( c, 1, true );
}

//////////////////////////////////////////////////////////// Host containers //
typedef thrust::host_vector< int >       IntHVec;
typedef thrust::host_vector< char >      CharHVec;
typedef thrust::host_vector< Point3 >    Point3HVec;

typedef thrust::host_vector< Tet >       TetHVec;
typedef thrust::host_vector< TetOpp >    TetOppHVec;

///////////////////////// Parameters /////////////////////////////////////
struct Statistics
{
    double initTime;
    double splitTime;
    double flipTime;
    double relocateTime;
    double sortTime; 
    double splayingTime; 
    double outTime;
    double totalTime;

    int failVertNum; 
    int finalStarNum;
    int totalFlipNum; 

    Statistics() 
    {
        reset(); 
    }

    void reset()
    {
        initTime        = .0;
        splitTime       = .0;
        flipTime        = .0;
        relocateTime    = .0;
        sortTime        = .0; 
        splayingTime    = .0; 
        outTime         = .0;
        totalTime       = .0;

        failVertNum     = 0; 
        finalStarNum    = 0;
        totalFlipNum    = 0; 
    }

    void accumulate( Statistics s )
    {
        initTime        += s.initTime; 
        splitTime       += s.splitTime;
        flipTime        += s.flipTime;
        relocateTime    += s.relocateTime;
        sortTime        += s.sortTime;
        splayingTime    += s.splayingTime;
        outTime         += s.outTime;
        totalTime       += s.totalTime;

        failVertNum     += s.failVertNum;
        finalStarNum    += s.finalStarNum;
        totalFlipNum    += s.totalFlipNum;
    }

    void average( int div )
    {
        initTime        /= div;
        splitTime       /= div;
        flipTime        /= div;
        relocateTime    /= div;
        sortTime        /= div;
        splayingTime    /= div;
        outTime         /= div;
        totalTime       /= div;
                           
        failVertNum     /= div;
        finalStarNum    /= div;
        totalFlipNum    /= div;
    }
};

enum InsertionRule
{
    InsCircumcenter,
    InsCentroid,
    InsRandom
};

const std::string InsRuleStr[] =
{
    "Circumcenter",
    "Centroid",
    "Random"
};

struct GDelParams
{
    bool noSplaying;    // Disable star splaying
    bool insertAll;         // Insert all before flipping
    bool noSorting;         // Disable sorting of input points and the tetras
    bool verbose;           // Print out some debugging informations

    InsertionRule insRule;  // Different rule for choosing points to insert in each round

    GDelParams() 
    {
        noSplaying  = false; 
        insertAll   = false; 
        noSorting   = false; 
        verbose     = false; 
        insRule     = InsCircumcenter;
    }
};

struct GDelOutput
{
    TetHVec     tetVec; 
    TetOppHVec  tetOppVec;
    CharHVec    tetInfoVec;
    IntHVec     failVertVec;
    IntHVec     vertTetVec;
    Point3      ptInfty; 

    // Statistics
    Statistics stats; 
}; 

