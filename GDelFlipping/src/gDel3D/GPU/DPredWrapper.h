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

#include "../CommonTypes.h"
#include "HostToKernel.h"

class DPredWrapper
{
private: 
    Point3      *_pointArr;
    int*        _orgPointIdx;
	int			_pointNum; 

    RealType*   _predConsts;
    RealType*   _predData;

    __forceinline__ __device__ Orient doOrient3DFastAdaptExact( Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const;

    __forceinline__ __device__ RealType doOrient2DFastExact( 
        const RealType* p0, 
        const RealType* p1, 
        const RealType* p2 ) const;

    __forceinline__ __device__ Orient doOrient3DSoSOnly( 
        int v0, int v1, int v2, int v3, 
        Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const;

    __forceinline__ __device__ Orient doOrient1DExact_Lifted(
        const RealType* p0,
        const RealType* p1,
        bool            lifted
    ) const; 

    __forceinline__ __device__ Orient doOrient2DExact_Lifted
    (
              RealType* curPredData,
        const RealType* p0,
        const RealType* p1,
        const RealType* p2,
        bool            lifted
    ) const; 

    __forceinline__ __device__ Orient doOrient3DFastExact_Lifted
    (
              RealType* curPredData,
        const RealType* p0,
        const RealType* p1,
        const RealType* p2,
        const RealType* p3,
        bool            lifted
    ) const;

    __device__ Side doInSphereSoSOnly
    (
        RealType* curPredData,
        int pi0, int pi1, int pi2, int pi3, int pi4, 
        Point3 p0, Point3 p1, Point3 p2, Point3 p3, Point3 p4
    ) const;

    __forceinline__ __device__ Side doInSphereFastAdaptExact( 
        RealType *curPredData, Point3 p0, Point3 p1, Point3 p2, Point3 p3, Point3 p4 ) const;

public: 
    int _infIdx;

    void init( 
        Point3* pointArr, 
        int     pointNum, 
        int*    orgPointIdx,
        int     infIdx, 
        int     PredTotalThreadNum 
    ); 

    void cleanup(); 

	__forceinline__ __device__ __host__ int pointNum() const; 

    __forceinline__ __device__ const Point3& getPoint( int idx ) const; 

    __forceinline__ __device__ Orient doOrient3DFast( 
        int v0, int v1, int v2, int v3, 
        Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const;
    
    __forceinline__ __device__ Orient doOrient3DFast( 
        int v0, int v1, int v2, int v3 ) const;

    __forceinline__ __device__ Orient doOrient3DSoS( 
        int v0, int v1, int v2, int v3, 
        Point3 p0, Point3 p1, Point3 p2, Point3 p3 ) const;

    __forceinline__ __device__ Orient doOrient3DSoS(
        int v0, int v1, int v2, int v3 ) const; 

    __forceinline__ __device__ Side doInSphereFast( 
        Tet tet, int vert, const Point3 pt[], Point3 ptVert ) const;

    __forceinline__ __device__ Side doInSphereSoS( 
        Tet tet, int vert, const Point3 pt[], Point3 ptVert ) const;

    __forceinline__ __device__ float distToCentroid( Tet tet, int v ) const;

    __forceinline__ __device__ float inSphereDet( Tet tet, int v ) const;
        
    __forceinline__ __device__ float inDist( Tet tet, int v ) const;

    __forceinline__ __device__ float maxDist( Tet tet, int v ) const;

    __forceinline__ __device__ float splitSphere( Tet tet, int v ) const;
}; 

