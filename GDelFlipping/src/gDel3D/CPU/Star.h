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

#include "CPUDecl.h"
#include "PredWrapper.h"

const int StarAvgTriNum = 32;

struct Star
{
private: 
	const PredWrapper  _predWrapper;
	FacetHVec*		   _facetStk; 	

    bool doFlipping( int triIdx, int vi, IntHVec* stack, IntHVec* isNonExtreme, int visitId );
    void flip31( int triIdx, int vi, IntHVec *stack );
    void flip22( int triIdx, int vi, IntHVec *stack );

    int markBeneathTriangles( 
        int         insVert, 
        IntHVec*    stack, 
        IntHVec*    visited, 
        int     visitId 
        );

    void stitchVertToHole( int insVert, int benTriIdx );
    int getFreeTri( int& freeTriIdx );
    void findFirstHoleSegment( int benTriIdx, int& firstTriIdx, 
        int& firstHoleTriIdx, int& firstVi );
    void makeOppLocal( const TetHVec& tetVec, const IntHVec* tetTriMap );
    void changeNewToValid();
	void addOneDeletionToQueue( int toVert );
	void addOneTriToQue( int fromIdx, const Tri& tri );
	void checkVertDeleted( int triIdx, int vi );
    int locateVert( int inVert ) const; 
    void pushFanToStack( int triIdx, int vi, IntHVec* stack );

public:
    int           _vert;
    TriHVec       _triVec;
    TriOppHVec    _triOppVec;
    IntHVec       _tetIdxVec;
    TriStatusHVec _triStatusVec;
	
	Star( const PredWrapper predWrapper );
    void clone( const Star& s ); 
    void clear();
    bool flipping( IntHVec *stack, FacetHVec *facetStk, IntHVec* isNonExtreme, int visitId ); 

    bool insertToStar( 
        int         insVert, 
        FacetHVec*  stk, 
        IntHVec*    stack, 
        IntHVec*    visited, 
        int         visitId 
        ); 

    void getProof(
        int  inVert, // Point that lies inside star
        int* proofArr
        );
    bool hasLinkVert( int inVert ) const;
    int getLinkTriIdx( const Tri& inTri ) const;
};
