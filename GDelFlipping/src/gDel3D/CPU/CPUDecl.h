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

#include <thrust/host_vector.h>
#include <cassert>

struct Facet
{
    int     _from;
    int     _fromIdx; 
    int     _to;
    int     _v0;
    int     _v1; 
};

enum TriStatus {
    TriFree,
    TriValid,
    TriNew,
};

struct Tri
{
    int _v[3];

    inline bool has( int v ) const
    {
        return ( _v[0] == v || _v[1] == v || _v[2] == v );
    }

    inline bool has( int v0, int v1, int v2 ) const
    {
        return ( _v[0] == v0 || _v[1] == v0 || _v[2] == v0 )
            && ( _v[0] == v1 || _v[1] == v1 || _v[2] == v1 )
            && ( _v[0] == v2 || _v[1] == v2 || _v[2] == v2 );
    }

    inline int indexOf( int v ) const
    {
        if ( _v[0] == v ) return 0;
        if ( _v[1] == v ) return 1;
        if ( _v[2] == v ) return 2;

        assert( false );

        return -1;
    }

    inline bool equal( const Tri& tri ) const
    {
        return ( ( _v[0] == tri._v[0] ) && ( _v[1] == tri._v[1] ) && ( _v[2] == tri._v[2] ) );
    }

    inline bool operator == ( const Tri& tri ) const
    {
        return equal( tri );
    }

    inline bool lessThan( const Tri& tri ) const
    {
        if ( _v[0] < tri._v[0] ) return true; 
        if ( _v[0] > tri._v[0] ) return false; 
        if ( _v[1] < tri._v[1] ) return true; 
        if ( _v[1] > tri._v[1] ) return false; 
        if ( _v[2] < tri._v[2] ) return true; 
        return false; // Equal 
    }

    inline bool operator < ( const Tri& tri ) const
    {
        return lessThan( tri );
    }
};

// Note: Keep bit positions same as TetOpp
struct TriOpp
{
    int _t[3];

    inline void setOpp( int vi, int oppLoc, int oppVi ) 
    {
        _t[ vi ] = ( oppLoc << 5 ) | oppVi;
    }

    inline int getOppTri( int vi ) const
    {
        return ( _t[ vi ] >> 5 );
    }

    inline int getOppVi( int vi ) const
    {
        return ( _t[ vi ] & 3 );
    }

    inline void setOppTri( int vi, int idx )
    {
        _t[ vi ] = ( _t[ vi ] & 0x1f ) | ( idx << 5 ); 
    }

    inline void setOppVi( int vi, int oppVi )
    {
        _t[ vi ] = ( _t[ vi ] & (~0x3) ) | ( oppVi ); 
    }

    inline void setOppSphereFail( int vi )
    {
        _t[ vi ] |= ( 1 << 4 );
    }

    inline bool isOppSphereFail( int vi ) const
    {
        return ( 1 == ( ( _t[ vi ] >> 4 ) & 1 ) );
    }

    inline int getOppTriVi( int vi ) const
    {
        return ( _t[ vi ] & (~0x1c) ); 
    }

    inline void setOppSpecial( int vi, bool state ) 
    {
        setBitState( _t[ vi ], 3, state ); 
    }

    inline bool isOppSpecial( int vi ) const
    {
        return isBitSet( _t[ vi ], 3 ); 
    }
};

inline int getOppTri( int val ) 
{
    return ( val >> 5 );
}

inline int getOppVi( int val ) 
{
    return ( val & 3 );
}

//////////////////////////////////////////////////////////// Containers
typedef int StackItem; 

typedef thrust::host_vector< float >     FloatHVec;
typedef thrust::host_vector< RealType >  RealHVec;
typedef thrust::host_vector< Tri >       TriHVec;
typedef thrust::host_vector< TriOpp >    TriOppHVec;
typedef thrust::host_vector< TriStatus > TriStatusHVec;
typedef thrust::host_vector< Facet >     FacetHVec;
typedef thrust::host_vector< StackItem > StackHVec; 

typedef IntHVec::iterator						IntHIter; 
typedef thrust::tuple< IntHIter, IntHIter >		IntHIterTuple2;
typedef thrust::zip_iterator< IntHIterTuple2 >	IntZipHIter;

//////////////////////////////////////////////////////////// Helper functions
inline int encode( int triIdx, int vi ) 
{
    return ( triIdx << 2 ) | vi; 
}

inline void decode( int code, int* idx, int* vi ) 
{
    *idx = ( code >> 2 ); 
    *vi = ( code & 3 ); 
}

inline TetOpp makeTetOpp( int v0, int v1, int v2, int v3 ) 
{
    const TetOpp opp = { v0, v1, v2, v3 }; 
    return opp; 
}

