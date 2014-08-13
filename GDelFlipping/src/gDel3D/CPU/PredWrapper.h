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
#include "predicates.h"

class PredWrapper 
{
private:
	const Point3*	_pointArr; 
	Point3			_ptInfty; 
	int			    _pointNum; 

	Orient doOrient3DSoSOnly
	(
	const RealType* p0,
	const RealType* p1,
	const RealType* p2,
	const RealType* p3,
	int v0,
	int v1,
	int v2,
	int v3
	) const;

	Orient doOrientation4SoSOnly
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
	) const;
	
public: 
    int _infIdx;

    void init( const Point3HVec& pointVec, Point3 ptInfty ); 

	const Point3& getPoint( int idx ) const; 
	int pointNum() const; 

	Orient doOrient3DFast( int v0, int v1, int v2, int v3 ) const;
	Orient doOrient3DAdapt( int v0, int v1, int v2, int v3 ) const;
	Orient doOrient3DSoS( int v0, int v1, int v2, int v3 ) const;
	Orient doOrient3DSoS( int v0, int v1, int v2, int v3, const IntHVec& orgPointIdx ) const;

	Side doInsphereFast( Tet tet, int v ) const;
	Side doInsphereAdapt( Tet tet, int v ) const;
	Orient doOrient4DAdaptSoS( Tet tet, int v ) const;
	Orient doOrient4DAdaptSoS( Tet tet, int v, const IntHVec& orgPointIdx ) const;

	RealType getInSphereVal( Tet tet, int v ) const;
    RealType distToCentroid( Tet tet, int v ) const;

}; 
