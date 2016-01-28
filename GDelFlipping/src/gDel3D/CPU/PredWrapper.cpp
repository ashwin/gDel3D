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

#include "PredWrapper.h"

void PredWrapper::init( const Point3HVec& pointVec, Point3 ptInfty )
{
	_pointArr	= &pointVec[0]; 
	_pointNum	= pointVec.size(); 
	_infIdx		= _pointNum; 
	_ptInfty	= ptInfty; 

    exactinit();
}

const Point3& PredWrapper::getPoint( int idx ) const
{
	return ( idx == _infIdx ) ? _ptInfty : _pointArr[ idx ]; 
}

int PredWrapper::pointNum() const 
{
	return _pointNum + 1; 
}

Orient PredWrapper::doOrient3DAdapt( int v0, int v1, int v2, int v3 ) const
{
    assert(     ( v0 != v1 ) && ( v0 != v2 ) && ( v0 != v3 )
                &&  ( v1 != v2 ) && ( v1 != v3 )
                &&  ( v2 != v3 )
                &&  "Duplicate indices in orientation!" );

    const Point3 p[] = { 
		getPoint( v0 ), getPoint( v1 ), getPoint( v2 ), getPoint( v3 ) 
	};
    
	RealType det  = orient3d( p[0]._p, p[1]._p, p[2]._p, p[3]._p );

    if ( (v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx) || (v3 == _infIdx) ) 
        det = -det; 

    return ortToOrient( det );
}

Orient PredWrapper::doOrient3DFast( int v0, int v1, int v2, int v3 ) const
{
    assert(     ( v0 != v1 ) && ( v0 != v2 ) && ( v0 != v3 )
                &&  ( v1 != v2 ) && ( v1 != v3 )
                &&  ( v2 != v3 )
                &&  "Duplicate indices in orientation!" );

    const Point3 p[] = { 
		getPoint( v0 ), getPoint( v1 ), getPoint( v2 ), getPoint( v3 ) 
	};
    
	RealType det  = orient3dfast( p[0]._p, p[1]._p, p[2]._p, p[3]._p );

    if ( (v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx) || (v3 == _infIdx) ) 
        det = -det; 

    return ortToOrient( det );
}

////////////////////////////////////////////////////////////////// Orient 3D //

Orient PredWrapper::doOrient3DSoSOnly
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
const RealType* p3,
int v0,
int v1,
int v2,
int v3
) const
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////

    const int DIM    = 3;
    const int NUM    = DIM + 1;
    const RealType* p[NUM] = { p0, p1, p2, p3 };
    int v[NUM]       = { v0, v1, v2, v3 };
    int pn           = 1;

    for ( int i = 0; i < 3; ++i )
    {
        int minI = i; 

        for ( int j = i + 1; j < 4; ++j ) 
            if ( v[j] < v[minI] )
                minI = j; 

        if ( minI != i )
        {
            int tempI = v[minI]; 
            v[minI]   = v[i]; 
            v[i]      = tempI; 
            pn        = -pn;

            //cuSwap( p[minI], p[i] );
            std::swap( p[minI], p[i] );
        }
    }

    RealType result = 0;
    RealType pa2[2], pb2[2], pc2[2];
    int depth;

	for ( depth = 0; depth < 14; ++depth )
	{
		switch ( depth )
		{
		case 0:
			pa2[0] = p[1][0];
			pa2[1] = p[1][1];
			pb2[0] = p[2][0];
			pb2[1] = p[2][1];
			pc2[0] = p[3][0];
			pc2[1] = p[3][1];
			break;
		case 1:
			pa2[0] = p[1][0];
			pa2[1] = p[1][2];
			pb2[0] = p[2][0];
			pb2[1] = p[2][2];
			pc2[0] = p[3][0];
			pc2[1] = p[3][2];
			break;
		case 2:
			pa2[0] = p[1][1];
			pa2[1] = p[1][2];
			pb2[0] = p[2][1];
			pb2[1] = p[2][2];
			pc2[0] = p[3][1];
			pc2[1] = p[3][2];
			break;
		case 3:
			pa2[0] = p[0][0];
			pa2[1] = p[0][1];
			pb2[0] = p[2][0];
			pb2[1] = p[2][1];
			pc2[0] = p[3][0];
			pc2[1] = p[3][1];
			break;
		case 4:
			result = p[2][0] - p[3][0];
			break;
		case 5:
			result = p[2][1] - p[3][1];
			break;
		case 6:
			pa2[0] = p[0][0];
			pa2[1] = p[0][2];
			pb2[0] = p[2][0];
			pb2[1] = p[2][2];
			pc2[0] = p[3][0];
			pc2[1] = p[3][2];
			break;
		case 7:
			result = p[2][2] - p[3][2];
			break;
		case 8:
			pa2[0] = p[0][1];
			pa2[1] = p[0][2];
			pb2[0] = p[2][1];
			pb2[1] = p[2][2];
			pc2[0] = p[3][1];
			pc2[1] = p[3][2];
			break;
		case 9:
			pa2[0] = p[0][0];
			pa2[1] = p[0][1];
			pb2[0] = p[1][0];
			pb2[1] = p[1][1];
			pc2[0] = p[3][0];
			pc2[1] = p[3][1];
			break;
		case 10:
			result = p[1][0] - p[3][0];
			break;
		case 11:
			result = p[1][1] - p[3][1];
			break;
		case 12:
			result = p[0][0] - p[3][0];
			break;
        default:
			result = 1.0;
			break;
		}

        switch ( depth )
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 6:
        case 8:
        case 9:
            result = orient2d( pa2, pb2, pc2 );
        }
			
		if ( result != 0 )
			break;
	}

    switch ( depth )
    {
    case 1:
    case 3:
    case 5:
    case 8:
    case 10:
        result = -result;
    }

	const RealType det = result * pn;

    return ortToOrient( det );
}

Orient PredWrapper::doOrient3DSoS( int v0, int v1, int v2, int v3 ) const
{
    Orient ort = doOrient3DAdapt( v0, v1, v2, v3 );

    if ( OrientZero == ort )
	{
		const RealType* p[] = { 
			getPoint( v0 )._p, 
			getPoint( v1 )._p, 
			getPoint( v2 )._p, 
			getPoint( v3 )._p 
		};

		ort = doOrient3DSoSOnly( p[0], p[1], p[2], p[3], v0, v1, v2, v3 );

		if ( (v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx) || (v3 == _infIdx) ) 
			ort = flipOrient( ort ); 
	}

	return ort; 
}

Orient PredWrapper::doOrient3DSoS( int v0, int v1, int v2, int v3, const IntHVec& orgPointIdx ) const
{
    Orient ort = doOrient3DAdapt( v0, v1, v2, v3 );

    if ( OrientZero == ort )
	{
		const RealType* p[] = { 
			getPoint( v0 )._p, 
			getPoint( v1 )._p, 
			getPoint( v2 )._p, 
			getPoint( v3 )._p 
		};

        if ( orgPointIdx.size() != 0 )	// Sorted
        {
            v0 = orgPointIdx[ v0 ]; 
            v1 = orgPointIdx[ v1 ]; 
            v2 = orgPointIdx[ v2 ]; 
            v3 = orgPointIdx[ v3 ]; 
        }

		ort = doOrient3DSoSOnly( p[0], p[1], p[2], p[3], v0, v1, v2, v3 );

		if ( (v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx) || (v3 == _infIdx) ) 
			ort = flipOrient( ort ); 
	}

	return ort; 
  
}

///////////////////////////////////////////////////////////////////// Sphere //

Side PredWrapper::doInsphereAdapt( Tet tet, int v ) const
{
    const RealType* p[]  = { 
		getPoint( tet._v[0] )._p, 
		getPoint( tet._v[1] )._p, 
		getPoint( tet._v[2] )._p, 
		getPoint( tet._v[3] )._p, 
		getPoint( v )._p
	}; 

    if ( v == _infIdx ) 
	{
		const RealType det = -orient3d( p[ 0 ], p[ 1 ], p[ 2 ], p[ 3 ] );

        return sphToSide( det );
	}

    if ( tet.has( _infIdx ) ) 
    {
        const int infVi = tet.getIndexOf( _infIdx ); 
        const int* ord  = TetViAsSeenFrom[ infVi ]; 

        // Check convexity, looking from inside
        const RealType det = -orient3d( p[ ord[0] ], p[ ord[2] ], p[ ord[1] ], p[4] ); 

        return sphToSide( det); 
    }

    const RealType det = insphere( p[0], p[1], p[2], p[3], p[4] );
    const Side s       = sphToSide( det );
    return s;
}

Side PredWrapper::doInsphereFast( Tet tet, int v ) const
{
    if ( v == _infIdx || tet.has( _infIdx ) ) 
        return SideZero; 

    const RealType* p[]  = { 
		getPoint( tet._v[0] )._p, 
		getPoint( tet._v[1] )._p, 
		getPoint( tet._v[2] )._p, 
		getPoint( tet._v[3] )._p, 
		getPoint( v )._p
	}; 

    const RealType det = inspherefast( p[0], p[1], p[2], p[3], p[4] );
    const Side s       = sphToSide( det );
    return s;
}

// Our orientation is defined opposite of Shewchuk
// For us, 0123 means seen from 3, 012 are in CCW order
// Given this orientation, Shewchuk's insphere value will
// be +ve if point is *outside* sphere and vice versa.
RealType PredWrapper::getInSphereVal( Tet tet, int v ) const
{
    const RealType* p[]  = { 
		getPoint( tet._v[0] )._p, 
		getPoint( tet._v[1] )._p, 
		getPoint( tet._v[2] )._p, 
		getPoint( tet._v[3] )._p, 
		getPoint( v )._p
	}; 

    RealType det; 
    
    if ( tet.has( _infIdx ) ) 
    {
        int infVi = tet.getIndexOf( _infIdx ); 

        const int *fv = TetViAsSeenFrom[ infVi ]; 

		det = orient3ddet( p[ fv[0] ], p[ fv[1] ], p[ fv[2] ], p[4] ); 
    }
    else 
        det = inspheredet( p[0], p[1], p[2], p[3], p[4] ); 

	return det; 
}

////////////////////////////////////////////////////////////////// Orient 4D //

Orient PredWrapper::doOrientation4SoSOnly
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
const RealType* p3,
const RealType* p4,
int             pi0,
int             pi1,
int             pi2,
int             pi3,
int             pi4
) const
{
    const int DIM = 4;
    const int NUM = DIM + 1;

    // Sort indices & note their order

    int idx[NUM]  = { pi0, pi1, pi2, pi3, pi4 };
    int ord[NUM]  = { 0, 1, 2, 3, 4 };
    int swapCount = 0;

    for ( int i = 0; i < ( NUM - 1 ); ++i )
    {
        for ( int j = ( NUM - 1 ); j > i; --j )
        {
            if ( idx[j] < idx[j - 1] )
            {
                cuSwap( idx[j], idx[j - 1] );
                cuSwap( ord[j], ord[j - 1] );   // Note order
                ++swapCount;
            }
        }
    }

    // Sort points in sorted index order.

    const RealType* pt4Arr[NUM] = { p0, p1, p2, p3, p4 };
    const RealType* ip[NUM];

    for ( int i = 0; i < NUM; ++i )
        ip[i] = pt4Arr[ ord[i] ];

    // Calculate determinants

    RealType op[NUM-1][DIM-1] = {0};
    RealType det			  = 0;
    int depth                 = 0;

    // Setup determinants
    while ( 0.0 == det )
    {
        ++depth;    // Increment depth, begins from 1

        switch ( depth )
        {
        case 0:
            //CudaAssert( false && "Depth cannot be ZERO! This happens only for non-ZERO determinant!" );
            break;

        case 1:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 2:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 3:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];    op[0][2] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 4:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];    op[0][2] = ip[1][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 5:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 6:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 7:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 8:
            op[0][0] = ip[2][1];    op[0][1] = ip[2][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 9:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 10:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 11:
            op[0][0] = ip[2][1];    op[0][1] = ip[2][0];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;
            
        case 12:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 13:
            op[0][0] = ip[2][2];    op[0][1] = ip[2][0];    op[0][2] = ip[2][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 14:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 15:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 16:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 17:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 18:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 19:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 20:
            op[0][0] = ip[3][0];
            op[1][0] = ip[4][0];
            break;

        case 21:
            op[0][0] = ip[3][1];
            op[1][0] = ip[4][1];
            break;

        case 22:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 23:
            op[0][0] = ip[3][2];
            op[1][0] = ip[4][2];
            break;

        case 24:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 25:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 26:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 27:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][0];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;

        case 28:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 29:
            op[0][0] = ip[3][0];    op[0][1] = ip[3][1];    op[0][2] = ip[3][2];
            op[1][0] = ip[4][0];    op[1][1] = ip[4][1];    op[1][2] = ip[4][2];
            break;

        case 30:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][0];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;
            
        case 31:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][2];    op[1][2] = ip[1][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 32:
            op[0][0] = ip[1][2];    op[0][1] = ip[1][0];    op[0][2] = ip[1][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 33:
            op[0][0] = ip[0][2];    op[0][1] = ip[0][0];    op[0][2] = ip[0][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 34:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[1][1];    op[1][1] = ip[1][2];    op[1][2] = ip[1][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 35:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[2][0];    op[2][1] = ip[2][1];    op[2][2] = ip[2][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 36:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 37:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 38:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 39:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 40:
            op[0][0] = ip[2][0];
            op[1][0] = ip[4][0];
            break;

        case 41:
            op[0][0] = ip[2][1];
            op[1][0] = ip[4][1];
            break;

        case 42:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 43:
            op[0][0] = ip[2][2];
            op[1][0] = ip[4][2];
            break;

        case 44:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 45:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 46:
            op[0][0] = ip[1][0];
            op[1][0] = ip[4][0];
            break;

        case 47:
            op[0][0] = ip[1][1];
            op[1][0] = ip[4][1];
            break;

        case 48:
            op[0][0] = ip[0][0];
            op[1][0] = ip[4][0];
            break;

        case 49:
            // See below for result
            break;

        default:
            CudaAssert( false && "Invalid SoS depth!" );
            break;

        }   // switch ( depth )

        // *** Calculate determinant

        bool lifted = false; 

        switch ( depth )
        {
        // 3D orientation involving the lifted coordinate
        case 2:  case 3:  case 4:  case 9:  case 12:
        case 14: case 25: case 31: case 34:

        // 2D orientation involving the lifted coordinate
        case 10: case 11: case 13: case 26: case 27: case 28:
        case 30: case 32: case 33: 

        // 1D orientation involving the lifted coordinate
        case 29:
            lifted = true; 
            break;
        }

        switch ( depth )
        {
        // 3D orientation determinant
        case 1:  case 5:  case 15: case 35:

        // 3D orientation involving the lifted coordinate
        case 2:  case 3:  case 4:  case 9:  case 12: 
        case 14: case 25: case 31: case 34:
            det = orient3dfastexact_lifted( op[0], op[1], op[2], op[3], lifted );
            break;

        // 2D orientation determinant
        case 6:  case 7:  case 8:  case 16: case 17: case 18: 
		case 19: case 22: case 24: case 36: case 37: case 38: 
		case 39: case 42: case 44: case 45: 

        // 2D orientation involving the lifted coordinate
        case 10: case 11: case 13: case 26: case 27:
        case 28: case 30: case 32: case 33:
            det = orient2dexact_lifted( op[0], op[1], op[2], lifted );
            break;

        // 1D orientation determinant
        case 20: case 21: case 23: case 40: case 41: case 43: 
		case 46: case 47: case 48:

        // 1D orientation involving the lifted coordinate
        case 29:
            det = orient1dexact_lifted( op[0], op[1], lifted );
            break;

        case 49:
            // Last depth, always POS
            det = +1.0;
            break;

        default:
            assert( false && "Invalid SoS depth!" );
            break;
        }
    }   // while ( 0 == orient )

    // Flip result for certain depths. (See SoS paper.)

    switch ( depth )
    {
    // -ve result
    case 1:  case 3:  case 7:  case 9:  case 11: case 14: case 16: 
	case 15: case 18: case 20: case 22: case 23: case 26: case 30: 
	case 31: case 32: case 37: case 39: case 41: case 44: case 46:
        det = -det;
        break;

    default:
        // Do nothing
        break;
    }

    ////
    // Flip result for odd swap count
    ////

    if ( ( swapCount % 2 ) != 0 )
        det = -det;

    return sphToOrient( det );
}

Orient PredWrapper::doOrient4DAdaptSoS( Tet t, int v ) const
{
    assert(    t._v[0] != t._v[1] && t._v[0] != t._v[2] && t._v[0] != t._v[3] && t._v[0] != v
            && t._v[1] != t._v[2] && t._v[1] != t._v[3] && t._v[1] != v
            && t._v[2] != t._v[3] && t._v[2] != v
            && t._v[3] != v );

    // Fast

	const RealType* p[]  = { 
		getPoint( t._v[0] )._p, getPoint( t._v[1] )._p, 
		getPoint( t._v[2] )._p, getPoint( t._v[3] )._p, 
		getPoint( v )._p
	}; 

    if ( v == _infIdx ) 
		return doOrient3DSoS( t._v[0], t._v[1], t._v[2], t._v[3] );

    if ( t.has( _infIdx ) ) 
    {
        const int infVi = t.getIndexOf( _infIdx ); 
        const int* ord  = TetViAsSeenFrom[ infVi ]; 

        // Check convexity, looking from inside
        return doOrient3DSoS( t._v[ ord[0] ], t._v[ ord[2] ], t._v[ ord[1] ], v ); 
    }

	const RealType sph0 = insphere( p[0], p[1], p[2], p[3], p[4] );
    const Orient ord0   = sphToOrient( sph0 );

    if ( OrientZero != ord0 )
        return ord0;

    // SoS

    const Orient ord2 = doOrientation4SoSOnly(
        p[0], p[1], p[2], p[3], p[4],
        t._v[0], t._v[1], t._v[2], t._v[3], v
        );

    return ord2;
}

Orient PredWrapper::doOrient4DAdaptSoS( Tet t, int v, const IntHVec& orgPointIdx ) const
{
    assert(    t._v[0] != t._v[1] && t._v[0] != t._v[2] && t._v[0] != t._v[3] && t._v[0] != v
            && t._v[1] != t._v[2] && t._v[1] != t._v[3] && t._v[1] != v
            && t._v[2] != t._v[3] && t._v[2] != v
            && t._v[3] != v );

    // Fast

	const RealType* p[]  = { 
		getPoint( t._v[0] )._p, getPoint( t._v[1] )._p, 
		getPoint( t._v[2] )._p, getPoint( t._v[3] )._p, 
		getPoint( v )._p
	}; 

    if ( v == _infIdx ) 
		return doOrient3DSoS( t._v[0], t._v[1], t._v[2], t._v[3], orgPointIdx );

    if ( t.has( _infIdx ) ) 
    {
        const int infVi = t.getIndexOf( _infIdx ); 
        const int* ord  = TetViAsSeenFrom[ infVi ]; 

        // Check convexity, looking from inside
        return doOrient3DSoS( t._v[ ord[0] ], t._v[ ord[2] ], t._v[ ord[1] ], v, orgPointIdx ); 
    }

	const RealType sph0 = insphere( p[0], p[1], p[2], p[3], p[4] );
    const Orient ord0   = sphToOrient( sph0 );

    if ( OrientZero != ord0 )
        return ord0;

    // SoS
    if ( orgPointIdx.size() != 0 )  // Sorted
    {
        t._v[0] = orgPointIdx[ t._v[0] ]; 
        t._v[1] = orgPointIdx[ t._v[1] ]; 
        t._v[2] = orgPointIdx[ t._v[2] ]; 
        t._v[3] = orgPointIdx[ t._v[3] ]; 
        v       = orgPointIdx[ v ]; 
    }

    const Orient ord2 = doOrientation4SoSOnly(
        p[0], p[1], p[2], p[3], p[4],
        t._v[0], t._v[1], t._v[2], t._v[3], v
        );

    return ord2;
}

RealType PredWrapper::distToCentroid( Tet tet, int v ) const
{
    Point3 p[] = { getPoint( tet._v[0] ), getPoint( tet._v[1] ), getPoint( tet._v[2] ), getPoint( tet._v[3] ), getPoint( v ) }; 

    RealType cx = ( p[0]._p[0] + p[1]._p[0] + p[2]._p[0] + p[3]._p[0] ) / 4.0f; 
    RealType cy = ( p[0]._p[1] + p[1]._p[1] + p[2]._p[1] + p[3]._p[1] ) / 4.0f; 
    RealType cz = ( p[0]._p[2] + p[1]._p[2] + p[2]._p[2] + p[3]._p[2] ) / 4.0f; 

    RealType dist = (( cx - p[4]._p[0] ) * ( cx - p[4]._p[0] ) + ( cy - p[4]._p[1] ) * ( cy - p[4]._p[1] )
        + ( cz - p[4]._p[2] ) * ( cz - p[4]._p[2] )); 
    
    if ( tet.has( _infIdx ) ) 
        dist = -dist; 

    return -dist; 
}

