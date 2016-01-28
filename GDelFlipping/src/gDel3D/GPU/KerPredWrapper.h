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

#include "DPredWrapper.h"
#include "KerShewchuk.h"

void DPredWrapper::init
( 
Point3* pointArr, 
int     pointNum, 
int*    orgPointIdx,
int     infIdx,
int     PredTotalThreadNum 
)
{
    _pointArr       = pointArr; 
    _pointNum       = pointNum; 
    _orgPointIdx    = orgPointIdx; 
    _infIdx         = infIdx; 
 
    // Prepare exact predicates temporary storage

    //std::cout << "Exact predicates scratch space: "
    //    << ( PredicateTotalSize * PredTotalThreadNum * sizeof( RealType ) ) / 1024 / 1024
    //    << "MB" << std::endl; 

    _predConsts = cuNew< RealType >( DPredicateBoundNum );
    _predData   = cuNew< RealType >( PredicateTotalSize * PredTotalThreadNum );

    kerInitPredicate<<< 1, 1 >>>( _predConsts );
    CudaCheckError();   
}

void DPredWrapper::cleanup()
{
    cuDelete( &_predConsts ); 
    cuDelete( &_predData ); 
}

__forceinline__ __device__ const Point3& DPredWrapper::getPoint( int idx ) const
{
    //CudaAssert( idx < _pointNum || idx == _infIdx ); 
    return _pointArr[ idx ]; 
}

__forceinline__ __device__ __host__ int DPredWrapper::pointNum() const
{
	return _pointNum; 
}

__forceinline__ __device__ Orient DPredWrapper::doOrient3DFast( 
    int v0, int v1, int v2, int v3, 
    Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const
{
    RealType det = orient3dFast( _predConsts, p0._p, p1._p, p2._p, p3._p );

    CudaAssert( v3 != _infIdx ); 

    if ( v0 == _infIdx | v1 == _infIdx | v2 == _infIdx ) 
        det = -det; 

    return ortToOrient( det );
}

__forceinline__ __device__ Orient DPredWrapper::doOrient3DFast( 
    int v0, int v1, int v2, int v3 ) const
{
    const Point3 pt[] = { getPoint( v0 ), getPoint( v1 ), getPoint( v2 ), getPoint( v3 ) }; 

    return doOrient3DFast( v0, v1, v2, v3, pt[0], pt[1], pt[2], pt[3] ); 
}


__forceinline__ __device__ Orient DPredWrapper::doOrient3DFastAdaptExact( 
    Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const
{
    const RealType det = orient3dFastAdaptExact( _predConsts, p0._p, p1._p, p2._p, p3._p );
    return ortToOrient( det );
}

__forceinline__ __device__ RealType DPredWrapper::doOrient2DFastExact
( 
const RealType* p0, 
const RealType* p1, 
const RealType* p2 
) const
{
    return orient2dFastExact( _predConsts, p0, p1, p2 );
}

// Exact 3D Orientation check must have failed (i.e. returned 0)
// No Infinity point here!!! 
__forceinline__ __device__ Orient DPredWrapper::doOrient3DSoSOnly( 
    int v0, int v1, int v2, int v3, 
    Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////

    const int DIM = 3;
    const int NUM = DIM + 1;
    const RealType* p[NUM] = { p0._p, p1._p, p2._p, p3._p }; 
    int pn = 1;

    if ( v0 > v2 ) { cuSwap( v0, v2 ); cuSwap( p[0], p[2] ); pn = -pn; }
    if ( v1 > v3 ) { cuSwap( v1, v3 ); cuSwap( p[1], p[3] ); pn = -pn; }
    if ( v0 > v1 ) { cuSwap( v0, v1 ); cuSwap( p[0], p[1] ); pn = -pn; }
    if ( v2 > v3 ) { cuSwap( v2, v3 ); cuSwap( p[2], p[3] ); pn = -pn; }
    if ( v1 > v2 ) { cuSwap( v1, v2 ); cuSwap( p[1], p[2] ); pn = -pn; }

    RealType result = 0;
    RealType pa2[2], pb2[2], pc2[2];
    int depth;

	for ( depth = 0; depth < 14; ++depth )
	{
		switch ( depth )
		{
		case 0:
			pa2[0] = p[1][0];   pa2[1] = p[1][1];
			pb2[0] = p[2][0];   pb2[1] = p[2][1];
			pc2[0] = p[3][0];   pc2[1] = p[3][1];
			break;
		case 1:
			/*pa2[0] = p[1][0];*/ pa2[1] = p[1][2];
			/*pb2[0] = p[2][0];*/ pb2[1] = p[2][2];
			/*pc2[0] = p[3][0];*/ pc2[1] = p[3][2];
			break;
		case 2:
			pa2[0] = p[1][1];   //pa2[1] = p[1][2];
			pb2[0] = p[2][1];   //pb2[1] = p[2][2];
			pc2[0] = p[3][1];   //pc2[1] = p[3][2];
			break;
		case 3:
			pa2[0] = p[0][0];   pa2[1] = p[0][1];
			pb2[0] = p[2][0];   pb2[1] = p[2][1];
			pc2[0] = p[3][0];   pc2[1] = p[3][1];
			break;
		case 4:
			result = p[2][0] - p[3][0];
			break;
		case 5:
			result = p[2][1] - p[3][1];
			break;
		case 6:
			/*pa2[0] = p[0][0];*/ pa2[1] = p[0][2];
			/*pb2[0] = p[2][0];*/ pb2[1] = p[2][2];
			/*pc2[0] = p[3][0];*/ pc2[1] = p[3][2];
			break;
		case 7:
			result = p[2][2] - p[3][2];
			break;
		case 8:
			pa2[0] = p[0][1];   //pa2[1] = p[0][2];
			pb2[0] = p[2][1];   //pb2[1] = p[2][2];
			pc2[0] = p[3][1];   //pc2[1] = p[3][2];
			break;
		case 9:
			pa2[0] = p[0][0];   pa2[1] = p[0][1];
			pb2[0] = p[1][0];   pb2[1] = p[1][1];
			pc2[0] = p[3][0];   pc2[1] = p[3][1];
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
        case 0: case 1: case 2: case 3: case 6: case 8: case 9:
            result = doOrient2DFastExact( pa2, pb2, pc2 );
        }
			
		if ( result != 0 )
			break;
	}

    switch ( depth )
    {
    case 1: case 3: case 5: case 8: case 10:
        result = -result;
    }

	const RealType det = result * pn;

    return ortToOrient( det );
}

__forceinline__ __device__ Orient DPredWrapper::doOrient3DSoS(
    int v0, int v1, int v2, int v3, 
    Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const
{
    // Fast-Exact
    Orient ord = doOrient3DFastAdaptExact( p0, p1, p2, p3 );
    
    if ( OrientZero == ord )
    {
        // SoS
        if ( _orgPointIdx != NULL ) 
        {
            v0 = _orgPointIdx[ v0 ]; 
            v1 = _orgPointIdx[ v1 ]; 
            v2 = _orgPointIdx[ v2 ]; 
            v3 = _orgPointIdx[ v3 ]; 
        }

        ord = doOrient3DSoSOnly( v0, v1, v2, v3, p0, p1, p2, p3 );
    }

    CudaAssert( v3 != _infIdx ); 

    if ( v0 == _infIdx | v1 == _infIdx | v2 == _infIdx ) 
        ord = flipOrient( ord ); 

    return ord; 
}

__forceinline__ __device__ Orient DPredWrapper::doOrient3DSoS(
    int v0, int v1, int v2, int v3 ) const
{
    const Point3 pt[] = { getPoint( v0 ), getPoint( v1 ), getPoint( v2 ), getPoint( v3 ) }; 

    return doOrient3DSoS( v0, v1, v2, v3, pt[0], pt[1], pt[2], pt[3] ); 
}

__forceinline__ __device__ Orient DPredWrapper::doOrient1DExact_Lifted
(
const RealType* p0,
const RealType* p1,
bool            lifted
) const
{
    const RealType det = orient1dExact_Lifted( _predConsts, p0, p1, lifted );
    return sphToOrient( det );
}

__forceinline__ __device__ Orient DPredWrapper::doOrient2DExact_Lifted
(
      RealType* curPredData,
const RealType* p0,
const RealType* p1,
const RealType* p2,
bool            lifted
) const
{
    const RealType det = orient2dExact_Lifted( _predConsts, curPredData, p0, p1, p2, lifted );
    return sphToOrient( det );
}

__forceinline__ __device__ Orient DPredWrapper::doOrient3DFastExact_Lifted
(
      RealType* curPredData,
const RealType* p0,
const RealType* p1,
const RealType* p2,
const RealType* p3,
bool            lifted
) const
{
    const RealType det = orient3dFastExact_Lifted( _predConsts, curPredData, p0, p1, p2, p3, lifted );
    return sphToOrient( det );
}

// Note: Only called when exact computation returns ZERO
// Reference: Simulation of Simplicity paper by Edelsbrunner and M\ucke
__device__ Side DPredWrapper::doInSphereSoSOnly( RealType* curPredData,
    int pi0, int pi1, int pi2, int pi3, int pi4, 
    Point3 p0, Point3 p1, Point3 p2, Point3 p3, Point3 p4
) const
{
    const int DIM = 4;
    const int NUM = DIM + 1;

    // Sort indices & note their order
    const RealType* ip[NUM] = { p0._p, p1._p, p2._p, p3._p, p4._p }; 
    int swapCount   = 0;

    if ( pi0 > pi4 ) { cuSwap( pi0, pi4 ); cuSwap( ip[0], ip[4] ); ++swapCount; }
    if ( pi1 > pi3 ) { cuSwap( pi1, pi3 ); cuSwap( ip[1], ip[3] ); ++swapCount; }
    if ( pi0 > pi2 ) { cuSwap( pi0, pi2 ); cuSwap( ip[0], ip[2] ); ++swapCount; }
    if ( pi2 > pi4 ) { cuSwap( pi2, pi4 ); cuSwap( ip[2], ip[4] ); ++swapCount; }
    if ( pi0 > pi1 ) { cuSwap( pi0, pi1 ); cuSwap( ip[0], ip[1] ); ++swapCount; }
    if ( pi2 > pi3 ) { cuSwap( pi2, pi3 ); cuSwap( ip[2], ip[3] ); ++swapCount; }
    if ( pi1 > pi4 ) { cuSwap( pi1, pi4 ); cuSwap( ip[1], ip[4] ); ++swapCount; }
    if ( pi1 > pi2 ) { cuSwap( pi1, pi2 ); cuSwap( ip[1], ip[2] ); ++swapCount; }
    if ( pi3 > pi4 ) { cuSwap( pi3, pi4 ); cuSwap( ip[3], ip[4] ); ++swapCount; }

    // Note: This is a check placed only to overcome a CUDA compiler bug!
    // As of CUDA 4 and sm_21, this bug happens if there are many SoS checks
    // caused by co-planar input points. Comment this if you wish, but do NOT
    // delete it, so it can be turned on when needed.
    //if ( ( p0 == p1 ) || ( p1 == p2 ) || ( p2 == p3 ) || ( p3 == p4 ) )
    //    printf( "Duplicate!\n" );

    CudaAssert(     ( pi0 != pi1 ) && ( pi0 != pi2 ) && ( pi0 != pi3 ) && ( pi0 != pi4 )
                &&  ( pi1 != pi2 ) && ( pi1 != pi3 ) && ( pi1 != pi4 )
                &&  ( pi2 != pi3 ) && ( pi2 != pi4 )
                &&  ( pi3 != pi4 )
                &&  "Duplicate indices in SoS orientation!" );

    // Calculate determinants

    RealType op[NUM-1][DIM-1] = {0};
    Orient orient             = OrientZero;
    int depth                 = 0;

    // Setup determinants
    while ( OrientZero == orient )
    {
        ++depth;    // Increment depth, begins from 1

        bool lifted = false; 

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

        case 2: lifted = true; 
            //op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            //op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            //op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            //op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 3: lifted = true; 
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];    op[0][2] = ip[1][1];
            /*op[1][0] = ip[2][0];*/    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            /*op[2][0] = ip[3][0];*/    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            /*op[3][0] = ip[4][0];*/    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 4: lifted = true; 
            // We know this is hit sometime
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/    op[0][2] = ip[1][0];
            op[1][0] = ip[2][1];    /*op[1][1] = ip[2][2];*/    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    /*op[2][1] = ip[3][2];*/    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    /*op[3][1] = ip[4][2];*/    op[3][2] = ip[4][0];
            break;

        case 5:
            ////CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 6:
            ////CudaAssert( false && "Here!" );
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 7:
            ////CudaAssert( false && "Here!" );
            /*op[0][0] = ip[2][0];*/    op[0][1] = ip[2][2];
            /*op[1][0] = ip[3][0];*/    op[1][1] = ip[3][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 8:
            ////CudaAssert( false && "Here!" );
            op[0][0] = ip[2][1];    /*op[0][1] = ip[2][2];*/
            op[1][0] = ip[3][1];    /*op[1][1] = ip[3][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 9: lifted = true; 
            ////CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 10: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 11: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][1];    op[0][1] = ip[2][0];    /*op[0][2] = ip[2][2];*/
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    /*op[1][2] = ip[3][2];*/
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    /*op[2][2] = ip[4][2];*/
            break;
            
        case 12: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 13: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][2];    op[0][1] = ip[2][0];    op[0][2] = ip[2][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 14: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 15:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 16:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 17:
            //CudaAssert( false && "Here!" );
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];
            /*op[1][0] = ip[3][0];*/    op[1][1] = ip[3][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 18:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/
            op[1][0] = ip[3][1];    /*op[1][1] = ip[3][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 19:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 20:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[3][0];
            op[1][0] = ip[4][0];
            break;

        case 21:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[3][1];
            op[1][0] = ip[4][1];
            break;

        case 22:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 23:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[3][2];
            op[1][0] = ip[4][2];
            break;

        case 24:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 25: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 26: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 27: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][1];    op[0][1] = ip[1][0];    /*op[0][2] = ip[1][2];*/
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    /*op[1][2] = ip[3][2];*/
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    /*op[2][2] = ip[4][2];*/
            break;

        case 28: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 29: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[3][0];    op[0][1] = ip[3][1];    op[0][2] = ip[3][2];
            op[1][0] = ip[4][0];    op[1][1] = ip[4][1];    op[1][2] = ip[4][2];
            break;

        case 30: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][1];    op[0][1] = ip[0][0];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;
            
        case 31: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][2];    op[1][2] = ip[1][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 32: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][2];    op[0][1] = ip[1][0];    op[0][2] = ip[1][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 33: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][2];    op[0][1] = ip[0][0];    op[0][2] = ip[0][1];
            //op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            //op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 34: lifted = true; 
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[1][1];    op[1][1] = ip[1][2];    op[1][2] = ip[1][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 35:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[2][0];    op[2][1] = ip[2][1];    op[2][2] = ip[2][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 36:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 37:
            //CudaAssert( false && "Here!" );
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];
            /*op[1][0] = ip[2][0];*/    op[1][1] = ip[2][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 38:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/
            op[1][0] = ip[2][1];    /*op[1][1] = ip[2][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 39:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 40:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][0];
            op[1][0] = ip[4][0];
            break;

        case 41:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][1];
            op[1][0] = ip[4][1];
            break;

        case 42:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 43:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[2][2];
            op[1][0] = ip[4][2];
            break;

        case 44:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 45:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 46:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][0];
            op[1][0] = ip[4][0];
            break;

        case 47:
            //CudaAssert( false && "Here!" );
            op[0][0] = ip[1][1];
            op[1][0] = ip[4][1];
            break;

        case 48:
            //CudaAssert( false && "Here!" );
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

        switch ( depth )
        {
        // 3D orientation determinant
        case 1: case 5: case 15: case 35:
        // 3D orientation involving the lifted coordinate
        case 2: case 3: case 4: case 9: case 12: case 14: case 25: case 31: case 34:
            orient = doOrient3DFastExact_Lifted( curPredData, op[0], op[1], op[2], op[3], lifted );
            break;

        // 2D orientation determinant
        case 6: case 7: case 8: case 16: case 17: case 18: case 19: case 22: case 24:
        case 36: case 37: case 38: case 39: case 42: case 44: case 45:
        // 2D orientation involving the lifted coordinate
        case 10: case 11: case 13: case 26: case 27: case 28: case 30: case 32: case 33:
            orient = doOrient2DExact_Lifted( curPredData, op[0], op[1], op[2], lifted );
            break;

        // 1D orientation determinant
        case 20: case 21: case 23: case 40: case 41: case 43: case 46: case 47: case 48:
        // 1D orientation involving the lifted coordinate
        case 29:
            orient = doOrient1DExact_Lifted( op[0], op[1], lifted );
            break;

        case 49:
            // Last depth, always POS
            orient = OrientPos;
            break;

        default:
            CudaAssert( false && "Invalid SoS depth!" );
            break;
        }
    }   // while ( 0 == orient )

    ////
    // Flip result for certain depths. (See SoS paper.)
    ////

    switch ( depth )
    {
    // -ve result
    case 1: case 3: case 7: case 9: case 11: case 14: case 16: case 15: case 18:
    case 20: case 22: case 23: case 26: case 30: case 31: case 32: case 37: case 39:
    case 41: case 44: case 46:
        orient = flipOrient( orient );
        break;

    default:
        // Do nothing
        break;
    }

    ////
    // Flip result for odd swap count
    ////

    if ( ( swapCount % 2 ) != 0 )
        orient = flipOrient( orient );

    return (Side) orient;
}

// Condition: p0p1p2p3 is positively oriented; 
// i.e. p0p1p2 is positive when looking from p3
__forceinline__ __device__ Side DPredWrapper::doInSphereFast( 
    Tet tet, int vert, const Point3 pt[], Point3 ptVert ) const
{
    if ( vert == _infIdx ) 
        return SideOut; 

    RealType det;

    if ( tet.has( _infIdx ) ) 
    {
        const int infVi = tet.getIndexOf( _infIdx ); 
        const int* ord  = TetViAsSeenFrom[ infVi ]; 

        // Check convexity, looking from inside
        det = -orient3dFast( _predConsts, pt[ ord[0] ]._p, pt[ ord[2] ]._p, pt[ ord[1] ]._p, ptVert._p ); 
    }
    else
        det = insphereFast( _predConsts, pt[0]._p, pt[1]._p, pt[2]._p, pt[3]._p, ptVert._p );

    return sphToSide( det );
}

__forceinline__ __device__ Side DPredWrapper::doInSphereFastAdaptExact( 
    RealType *curPredData, Point3 p0, Point3 p1, Point3 p2, Point3 p3, Point3 p4 ) const
{
    const RealType det1 = insphereFastAdaptExact( _predConsts, curPredData, 
        p0._p, p1._p, p2._p, p3._p, p4._p );
    const Side ord1     = sphToSide( det1 );
    return ord1;
}

__forceinline__ __device__ Side DPredWrapper::doInSphereSoS( 
    Tet tet, int vert, const Point3 pt[], Point3 ptVert ) const
{
    if ( vert == _infIdx ) 
        return SideOut; 

    if ( tet.has( _infIdx ) ) 
    {
        const int infVi = tet.getIndexOf( _infIdx ); 
        const int* ord  = TetViAsSeenFrom[ infVi ]; 

        // Check convexity, looking from inside
        const Orient ort = doOrient3DSoS( 
            tet._v[ ord[0] ], tet._v[ ord[2] ], tet._v[ ord[1] ], vert, 
            pt[ ord[0] ], pt[ ord[2] ], pt[ ord[1] ], ptVert ); 

        return sphToSide( ort ); 
    }

    const int curPredDataIdx    = getCurThreadIdx() * PredicateTotalSize;
    RealType* curPredData       = &_predData[ curPredDataIdx ]; 

    const Side s0 = doInSphereFastAdaptExact( curPredData, pt[0], pt[1], pt[2], pt[3], ptVert );

    if ( SideZero != s0 )
        return s0;

    // SoS
    if ( _orgPointIdx != NULL ) 
    {
        tet._v[0] = _orgPointIdx[ tet._v[0] ]; 
        tet._v[1] = _orgPointIdx[ tet._v[1] ]; 
        tet._v[2] = _orgPointIdx[ tet._v[2] ]; 
        tet._v[3] = _orgPointIdx[ tet._v[3] ]; 
        vert      = _orgPointIdx[ vert ]; 
    }

    const Side s2 = doInSphereSoSOnly( curPredData,
        tet._v[0], tet._v[1], tet._v[2], tet._v[3], vert, 
        pt[0], pt[1], pt[2], pt[3], ptVert );
    return s2;
}

__forceinline__ __device__ float DPredWrapper::distToCentroid( Tet tet, int v ) const
{
    Point3 p[] = { getPoint( tet._v[0] ), getPoint( tet._v[1] ), getPoint( tet._v[2] ), getPoint( tet._v[3] ), getPoint( v ) }; 

    float dist; 

    if ( tet.has( _infIdx ) ) 
    {
        int infVi = tet.getIndexOf( _infIdx ); 

        const int *fv = TetViAsSeenFrom[ infVi ]; 

        dist = -orient3dDet( p[ fv[0] ]._p, p[ fv[1] ]._p, p[ fv[2] ]._p, p[4]._p ); 
    }
    else 
    {
        float cx = ( p[0]._p[0] + p[1]._p[0] + p[2]._p[0] + p[3]._p[0] ) / 4.0f; 
        float cy = ( p[0]._p[1] + p[1]._p[1] + p[2]._p[1] + p[3]._p[1] ) / 4.0f; 
        float cz = ( p[0]._p[2] + p[1]._p[2] + p[2]._p[2] + p[3]._p[2] ) / 4.0f; 

        dist = 1.0 / (( cx - p[4]._p[0] ) * ( cx - p[4]._p[0] ) + ( cy - p[4]._p[1] ) * ( cy - p[4]._p[1] )
            + ( cz - p[4]._p[2] ) * ( cz - p[4]._p[2] )); 
    }

    return dist; 
}

__forceinline__ __device__ float DPredWrapper::inSphereDet( Tet tet, int v ) const
{
    Point3 p[] = { getPoint( tet._v[0] ), getPoint( tet._v[1] ), getPoint( tet._v[2] ), getPoint( tet._v[3] ), getPoint( v ) }; 

    float det; 
    
    if ( tet.has( _infIdx ) ) 
    {
        int infVi = tet.getIndexOf( _infIdx ); 

        const int *fv = TetViAsSeenFrom[ infVi ]; 

        // Note: reverse the orientation
        det = -orient3dDet( p[ fv[0] ]._p, p[ fv[1] ]._p, p[ fv[2] ]._p, p[4]._p ); 
    }
    else 
        det = -insphereDet( p[0]._p, p[1]._p, p[2]._p, p[3]._p, p[4]._p ); 

    return det; 
}
