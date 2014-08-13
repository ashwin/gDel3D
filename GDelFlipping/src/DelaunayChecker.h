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

#include "gDel3D/CommonTypes.h"
#include "gDel3D/CPU/PredWrapper.h"

class DelaunayChecker 
{
private: 
	Point3HVec*   _pointVec; 
    GDelOutput*   _output; 

	PredWrapper  _predWrapper; 

    int getVertexCount();
    int getSegmentCount();
    int getTriangleCount();
    int getTetraCount();

    void printTetraAndOpp( int ti, const Tet& tet, const TetOpp& opp );

public: 
    DelaunayChecker( Point3HVec* pointVec, GDelOutput* output ); 

    void checkEuler();
    void checkAdjacency();
    void checkOrientation();
    bool checkDelaunay( bool writeFile = false );
}; 

struct Segment
{
    int _v[2];

    inline bool equal( const Segment& seg ) const
    {
        return ( ( _v[0] == seg._v[0] ) && ( _v[1] == seg._v[1] ) );
    }

    inline bool operator == ( const Segment& seg ) const
    {
        return equal( seg );
    }

    inline bool lessThan( const Segment& seg ) const
    {
        if ( _v[0] < seg._v[0] )
            return true;
        if ( _v[0] > seg._v[0] )
            return false;
        if ( _v[1] < seg._v[1] )
            return true;

        return false; 
    }

    inline bool operator < ( const Segment& seg ) const
    {
        return lessThan( seg );
    }
};

const int TetSegNum = 6;
const int TetSeg[ TetSegNum ][2] = {
    { 0, 1 },
    { 0, 2 },
    { 0, 3 },
    { 1, 2 },
    { 1, 3 },
    { 2, 3 },
};

const int TetFaceNum = 4;
const int TetFace[ TetFaceNum ][3]  = {
    { 0, 1, 2 },
    { 0, 1, 3 },
    { 0, 2, 3 },
    { 1, 2, 3 },
};

