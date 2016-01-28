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

#include "../PerfTimer.h"

#include "Star.h"
#include "PredWrapper.h"

void Star::addOneDeletionToQueue( int toVert ) 
{
	Facet f; 

	// toVert is drowned _vert's star, we want to delete _vert 
	// from toVert's star if possible. So treat this as if 
	// we're inserting toVert into _vert's star but fail, to 
	// generate a proof back. 
	f._from		= toVert;
	f._to		= _vert; 
	f._fromIdx	= -1; 

	_facetStk->push_back( f ); 
}

void Star::addOneTriToQue( int fromIdx, const Tri& tri )
{
    Facet facet;
    facet._from     = _vert;
    facet._fromIdx  = fromIdx; 

    int minNext = -1, min = 0; 

    if ( tri._v[ 0 ] > _vert ) minNext = 0; 
    if ( tri._v[ 1 ] < tri._v[ min ] ) min = 1; 
    if ( tri._v[ 1 ] > _vert && ( minNext == -1 || tri._v[ 1 ] < tri._v[ minNext ] ) ) minNext = 1; 
    if ( tri._v[ 2 ] < tri._v[ min ] ) min = 2; 
    if ( tri._v[ 2 ] > _vert && ( minNext == -1 || tri._v[ 2 ] < tri._v[ minNext ] ) ) minNext = 2; 

    if ( minNext == -1 ) 
        minNext = min; 

    facet._to   = tri._v[ minNext ];
    facet._v0   = tri._v[ ( minNext + 1 ) % 3 ];
    facet._v1   = tri._v[ ( minNext + 2 ) % 3 ];

    _facetStk->push_back( facet );

    return;
}

Star::Star( const PredWrapper predWrapper ) 
    : _predWrapper( predWrapper )
{
    _triVec.reserve( StarAvgTriNum );
    _triOppVec.reserve( StarAvgTriNum );
    _tetIdxVec.reserve( StarAvgTriNum );
    _triStatusVec.reserve( StarAvgTriNum );
}

void Star::clear()
{
    _triVec.clear(); 
    _triOppVec.clear(); 
    _tetIdxVec.clear(); 
    _triStatusVec.clear(); 
}

void Star::flip31( int triIdx, int vi, IntHVec* stack )
{ 
    const int corVi   = ( vi + 1 ) % 3; 
    const TriOpp opp  = _triOppVec[ triIdx ]; 
    const Tri tri     = _triVec[ triIdx ];  
    const int oppTri  = opp.getOppTri( vi ); 
    const int oppVi   = opp.getOppVi( vi ); 
    const int sideTri = opp.getOppTri( corVi ); 
    const int sideVi  = opp.getOppVi( corVi ); 
    const int oppVert = _triVec[ oppTri ]._v[ oppVi ]; 

    const Tri newTri = { tri._v[ vi ], tri._v[ corVi ], oppVert }; 

    _triVec[ triIdx ]    = newTri; 
    _tetIdxVec[ triIdx ] = -1; 

    const int oppV0 = _triOppVec[ oppTri ].getOppTriVi( (oppVi + 1) % 3 ); 
    const int oppV1 = _triOppVec[ sideTri ].getOppTriVi( (sideVi + 2) % 3 ); 
    const int oppV2 = opp.getOppTriVi( (vi + 2) % 3 ); 

    const TriOpp newOpp = { oppV0, oppV1, oppV2 }; 

    _triOppVec[ triIdx ] = newOpp; 

    _triOppVec[ getOppTri( oppV0 ) ].setOpp( getOppVi( oppV0 ), triIdx, 0 ); 
    _triOppVec[ getOppTri( oppV1 ) ].setOpp( getOppVi( oppV1 ), triIdx, 1 ); 
    _triOppVec[ getOppTri( oppV2 ) ].setOpp( getOppVi( oppV2 ), triIdx, 2 ); 

    _triStatusVec[ oppTri ]  = TriFree; 
    _triStatusVec[ sideTri ] = TriFree; 

	// A vertex is deleted, push in a stack item for consistency checking
	addOneDeletionToQueue( tri._v[ (vi + 2) % 3 ] ); 

    // Recursive flipping
    stack->push_back( encode( triIdx, 0 ) ); 
    stack->push_back( encode( triIdx, 1 ) ); 
    stack->push_back( encode( triIdx, 2 ) ); 
}

void Star::flip22( int triIdx, int vi, IntHVec* stack )
{
    const TriOpp opp  = _triOppVec[ triIdx ]; 
    const Tri tri     = _triVec[ triIdx ];  
    const int oppTri  = opp.getOppTri( vi ); 
    const int oppVi   = opp.getOppVi( vi ); 
    const int oppVert = _triVec[ oppTri ]._v[ oppVi ]; 

    const Tri newTri0 = { tri._v[ vi ], oppVert, tri._v[ (vi + 2) % 3 ] }; 
    const Tri newTri1 = { oppVert, tri._v[ vi ], tri._v[ (vi + 1) % 3 ] }; 

    _triVec[ triIdx ] = newTri0; 
    _triVec[ oppTri ] = newTri1; 

    _tetIdxVec[ triIdx ] = -1; 
    _tetIdxVec[ oppTri ] = -1; 

    const int oppV0 = opp.getOppTriVi( (vi + 2) % 3 ); 
    const int oppV1 = _triOppVec[ oppTri ].getOppTriVi( (oppVi + 1) % 3 ); 
    const int oppV2 = _triOppVec[ oppTri ].getOppTriVi( (oppVi + 2) % 3 ); 
    const int oppV3 = opp.getOppTriVi( (vi + 1) % 3 ); 

    TriOpp newOpp0 = { oppV2, oppV3, -1 }; 
    newOpp0.setOpp( 2, oppTri, 2 ); 

    _triOppVec[ triIdx ] = newOpp0; 

    TriOpp newOpp1 = { oppV0, oppV1, -1 }; 
    newOpp1.setOpp( 2, triIdx, 2 ); 

    _triOppVec[ oppTri ] = newOpp1; 

    _triOppVec[ getOppTri( oppV0 ) ].setOpp( getOppVi( oppV0 ), oppTri, 0 ); 
    _triOppVec[ getOppTri( oppV1 ) ].setOpp( getOppVi( oppV1 ), oppTri, 1 ); 
    _triOppVec[ getOppTri( oppV2 ) ].setOpp( getOppVi( oppV2 ), triIdx, 0 ); 
    _triOppVec[ getOppTri( oppV3 ) ].setOpp( getOppVi( oppV3 ), triIdx, 1 ); 

    // Recursive flipping
    stack->push_back( encode( triIdx, 0 ) ); 
    stack->push_back( encode( triIdx, 1 ) ); 

    stack->push_back( encode( oppTri, 0 ) ); 
    stack->push_back( encode( oppTri, 1 ) ); 
}

// Flip-Flop algorithm
bool Star::doFlipping( int startIdx, int startVi, IntHVec* stack, IntHVec* isNonExtreme, int visitId )
{
    stack->clear(); 

    stack->push_back( encode( startIdx, startVi ) ); 

    int triIdx, vi; 

    int count = 0; 

    while ( stack->size() > 0 ) 
    {
        decode( stack->back(), &triIdx, &vi ); 
        stack->pop_back(); 
        ++count;

        ////////////
        if ( _triStatusVec[ triIdx ] == TriFree ) continue;  

        // Get the opposite vertex
        const TriOpp opp = _triOppVec[ triIdx ]; 
        const Tri tri    = _triVec[ triIdx ];  

        const int oppTri = opp.getOppTri( vi ); 
        const int oppVi  = opp.getOppVi( vi ); 

        const int oppVert = _triVec[ oppTri ]._v[ oppVi ]; 

        // Find the min labeled vert in the configuration
        int minVert = INT_MAX; 

        if ( (*isNonExtreme)[ oppVert ] == visitId ) minVert = oppVert; 

        if ( tri._v[0] < minVert && (*isNonExtreme)[ tri._v[0] ] == visitId ) minVert = tri._v[0]; 
        if ( tri._v[1] < minVert && (*isNonExtreme)[ tri._v[1] ] == visitId ) minVert = tri._v[1]; 
        if ( tri._v[2] < minVert && (*isNonExtreme)[ tri._v[2] ] == visitId ) minVert = tri._v[2]; 

        // Skip if flipping increases the min non-extreme vert's degree
        if ( minVert == tri._v[ vi ] || minVert == oppVert ) continue; 

        Orient ort = _predWrapper.doOrient4DAdaptSoS( makeTet( tri._v[ 0 ], tri._v[ 1 ], tri._v[ 2 ], _vert ), oppVert ); 

        if ( OrientPos == ort && minVert == INT_MAX ) 
            continue; 

        // Can I do a 3-1 flip?
        if ( opp.getOppTri( (vi + 1) % 3 ) == _triOppVec[ oppTri ].getOppTri( (oppVi + 2) % 3 ) ) 
        {
            if ( ort == OrientNeg || minVert == tri._v[ (vi + 2) % 3 ] ) 
                flip31( triIdx, vi, stack ); 

            continue; 
        }

        if ( opp.getOppTri( (vi + 2) % 3 ) == _triOppVec[ oppTri ].getOppTri( (oppVi + 1) % 3 ) ) 
        {
            if ( ort == OrientNeg || minVert == tri._v[ (vi + 1) % 3 ] ) 
                flip31( triIdx, (vi + 2) % 3, stack ); 

            continue; 
        }

        // Check 2-2 flippability
        Orient or1 = _predWrapper.doOrient3DSoS( tri._v[ vi ], tri._v[ (vi + 1) % 3 ],
            oppVert, _vert ); 

        if ( OrientNeg == or1 ) 
        {
            // Mark non-extreme vert
            if ( ort == OrientNeg && (*isNonExtreme)[ tri._v[ (vi + 1) % 3 ] ] != visitId  )
            {
                (*isNonExtreme)[ tri._v[ (vi + 1) % 3 ] ] = visitId; 

                pushFanToStack( triIdx, (vi + 2) % 3, stack ); 
            }

            continue; 
        }

        Orient or2 = _predWrapper.doOrient3DSoS( tri._v[ (vi + 2) % 3 ], tri._v[ vi ],
            oppVert, _vert ); 

        if ( OrientNeg == or2 ) 
        {
            if ( ort == OrientNeg && (*isNonExtreme)[ tri._v[ (vi + 2) % 3 ] ] != visitId ) 
            {
                (*isNonExtreme)[ tri._v[ (vi + 2) % 3 ] ] = visitId; 

                pushFanToStack( triIdx, vi, stack ); 
            }

            continue; 
        }

        flip22( triIdx, vi, stack ); 
    }

    return ( count > 1 ); 
}

void Star::pushFanToStack( int startIdx, int vi, IntHVec* stack )
{
    int triIdx = startIdx; 

    do {
        stack->push_back( encode( triIdx, vi ) );

        TriOpp& opp = _triOppVec[ triIdx ]; 

        triIdx = opp.getOppTri( vi ); 
        vi     = ( opp.getOppVi( vi ) + 2 ) % 3; 
    } while ( triIdx != startIdx ); 
}

bool Star::flipping( IntHVec* stack, FacetHVec* facetStk, IntHVec* isNonExtreme, int visitId )
{
	_facetStk = facetStk; 

    for ( int idx = 0; idx < _triVec.size(); ++idx ) 
    {
        if ( TriFree == _triStatusVec[ idx ] ) continue; 

        const TriOpp opp = _triOppVec[ idx ]; 

        for ( int vi = 0; vi < 3; ++vi ) 
        {
            if ( !opp.isOppSphereFail( vi ) ) continue; 

            if ( doFlipping( idx, vi, stack, isNonExtreme, visitId ) ) break; 
        }
    }

    return true;
}

bool Star::insertToStar
( 
int         insVert, 
FacetHVec*  stk, 
IntHVec*    stack, 
IntHVec*    visited, 
int         visitId 
)
{
	_facetStk = stk; 

	//std::cout << "  Inserting " << insVert << " into " << _vert << " --> "; 

    const int benTriIdx = markBeneathTriangles( insVert, stack, visited, visitId );

	//std::cout << (( -1 == benTriIdx ) ? "fail" : "success") <<  std::endl; 

    if ( -1 == benTriIdx ) return false;

    stitchVertToHole( insVert, benTriIdx );

    return true;
}

void Star::checkVertDeleted( int triIdx, int vi )
{
	int curVi	  = ( vi + 2 ) % 3; 
	int curTriIdx = triIdx; 

	bool vertDeleted = false; 

	// Rotate around vi
	while ( TriFree == _triStatusVec[ curTriIdx ] ) 
	{
        const TriOpp& curTriOpp = _triOppVec[ curTriIdx ];
        const int oppTriIdx		= curTriOpp.getOppTri( ( curVi + 2 ) % 3 ); 
        const TriStatus status  = _triStatusVec[ oppTriIdx ]; 

        // Continue moving
        curVi		= curTriOpp.getOppVi( ( curVi + 2 ) % 3 ); 
        curTriIdx   = oppTriIdx;

		if ( curTriIdx == triIdx ) 
		{
			vertDeleted = true; 
			break; 
		}
	}

	if ( vertDeleted ) 
		addOneDeletionToQueue( _triVec[ triIdx ]._v[ vi ] ); 
}

int Star::markBeneathTriangles
( 
int         insVert, 
IntHVec*    stack, 
IntHVec*    visited, 
int         visitId 
)
{
    int benTriIdx = -1;

    if ( stack != NULL )    
    {
        int container = locateVert( insVert ); 

        if ( container < 0 ) return -1; 

        stack->clear(); 
        stack->push_back( container ); 

        (*visited)[ container ] = visitId; 

        while ( !stack->empty() ) 
        {
            const int triIdx = stack->back(); 
            stack->pop_back(); 

            const Tri tri = _triVec[ triIdx ];

            const Orient ord = _predWrapper.doOrient4DAdaptSoS(
                makeTet( tri._v[0], tri._v[1], tri._v[2], _vert ), insVert );

            if ( OrientNeg == ord )
            {
                _triStatusVec[ triIdx ] = TriFree;  // Mark as deleted
                _tetIdxVec[ triIdx ]    = -1;
                benTriIdx               = triIdx;

                const TriOpp opp = _triOppVec[ triIdx ]; 

                for ( int vi = 0; vi < 3; ++vi ) 
                {
    			    checkVertDeleted( triIdx, vi ); 

                    const int oppTri = opp.getOppTri( vi ); 

                    if ( (*visited)[ oppTri ] != visitId ) 
                    {
                        stack->push_back( oppTri ); 

                        (*visited)[ oppTri ] = visitId; 
                    }
                }
            }
        }
    }
    else
    {
        // Scan all

        for ( int triIdx = 0; triIdx < ( int ) _triVec.size(); ++triIdx )
        {
            if ( TriFree == _triStatusVec[ triIdx ] ) continue;

            const Tri tri = _triVec[ triIdx ];

            const Orient ord = _predWrapper.doOrient4DAdaptSoS(
                makeTet( tri._v[0], tri._v[1], tri._v[2], _vert ), insVert );

            if ( OrientNeg == ord )
            {
                _triStatusVec[ triIdx ] = TriFree;  // Mark as deleted
                _tetIdxVec[ triIdx ]    = -1;
                benTriIdx               = triIdx;

			    checkVertDeleted( triIdx, 0 ); 
			    checkVertDeleted( triIdx, 1 ); 
			    checkVertDeleted( triIdx, 2 ); 
            }
        }
    }

    return benTriIdx;
}

void Star::findFirstHoleSegment
(
int  benTriIdx,
int& firstTriIdx,
int& firstVi,
int& firstHoleTriIdx
)
{
    // Check if input beneath triangle is on hole border

    const TriOpp triOppFirst = _triOppVec[ benTriIdx ];

    for ( int vi = 0; vi < 3; ++vi )
    {
        const int oppTriIdx = triOppFirst.getOppTri( vi );

        if ( TriFree != _triStatusVec[ oppTriIdx ] )
        {
            firstTriIdx     = oppTriIdx;
            firstHoleTriIdx = benTriIdx;
            firstVi         = triOppFirst.getOppVi( vi ); 

            return; 
        }                
    }

    // Iterate triangles

    for ( int triIdx = 0; triIdx < ( int ) _triVec.size(); ++triIdx )
    {
        if ( TriFree == _triStatusVec[ triIdx ] ) continue;

        const TriOpp triOpp = _triOppVec[ triIdx ];

        for ( int vi = 0; vi < 3; ++vi )
        {
            const int triOppIdx = triOpp.getOppTri( vi );

            if ( TriFree == _triStatusVec[ triOppIdx ] )
            {
                firstTriIdx     = triIdx;
                firstVi         = vi;
                firstHoleTriIdx = triOppIdx;

                return; 
            }                
        }
    }

    assert( false && "No hole border triangle found!" );

    return;
}

int Star::getFreeTri( int& freeTriIdx )
{
    int begTriIdx = freeTriIdx;

    while ( freeTriIdx < _triVec.size() ) 
    {
        if ( TriFree == _triStatusVec[ freeTriIdx ] )
            return freeTriIdx;

        ++freeTriIdx;
    }

    const int oldSize = ( int ) _triVec.size();
    const int newSize = oldSize + 1;

    _triVec.resize( newSize );
    _triOppVec.resize( newSize );
    _tetIdxVec.resize( newSize );
    _triStatusVec.push_back( TriFree );

    freeTriIdx = oldSize;

    return freeTriIdx;
}

void Star::changeNewToValid()
{
    for ( int triIdx = 0; triIdx < ( int ) _triStatusVec.size(); ++triIdx )
    {
        if ( TriNew == _triStatusVec[ triIdx ] )
            _triStatusVec[ triIdx ] = TriValid;
    }

    return;
}

void Star::stitchVertToHole
(
int        insVert,
int        benTriIdx
)
{
    int firstTriIdx    = -1;
    int firstVi        = -1;
    int firstNewTriIdx = -1;

    findFirstHoleSegment( benTriIdx, firstTriIdx, firstVi, firstNewTriIdx );

    assert( -1 != firstTriIdx );

    // Get first two vertices of hole
    int curTriIdx       = firstTriIdx;
    const Tri& curTri   = _triVec[ curTriIdx ];
    const int firstVert = curTri._v[ ( firstVi + 1 ) % 3 ]; 
    int curVi           = ( firstVi + 2 ) % 3; 
    int curVert         = curTri._v[ curVi ];

    //*** Stitch first triangle

    const Tri firstNewTri           = { insVert, curVert, firstVert };
    _triVec[ firstNewTriIdx ]       = firstNewTri;
    _tetIdxVec[ firstNewTriIdx ]    = -1;
    _triStatusVec[ firstNewTriIdx ] = TriNew;

    addOneTriToQue( firstNewTriIdx, firstNewTri );

    // Adjancency with opposite triangle
    TriOpp& firstNewTriOpp = _triOppVec[ firstNewTriIdx ]; 
    firstNewTriOpp.setOpp( 0, firstTriIdx, firstVi );

    TriOpp& firstTriOpp = _triOppVec[ firstTriIdx ];
    firstTriOpp.setOpp( firstVi, firstNewTriIdx, 0 );

    // Walk around outside of hole, stitching rest of triangles

    int freeTriIdx    = 0; // Start from begin of array
    int prevNewTriIdx = firstNewTriIdx; 

    // Walk outside the hole in CW direction

    while ( curVert != firstVert ) 
    {
        // Check opposite triangle
        const TriOpp& curTriOpp = _triOppVec[ curTriIdx ];
        const int gloOppTriIdx  = curTriOpp.getOppTri( ( curVi + 2 ) % 3 ); 
        const TriStatus status  = _triStatusVec[ gloOppTriIdx ]; 

        assert( gloOppTriIdx >= 0 && gloOppTriIdx <= _triVec.size() ); 

        // Tri is outside the hole
        if ( ( TriFree != status ) && ( TriNew != status ) )
        {
            // Continue moving
            const int oppVi = curTriOpp.getOppVi( ( curVi + 2 ) % 3 ); 
            curVi           = ( oppVi + 2 ) % 3;                
            curTriIdx       = gloOppTriIdx;
        }
        // Tri is in hole
        else
        {
            const int newTriIdx =
                ( TriFree == status )
                ? gloOppTriIdx  // Reuse hole triangle
                : getFreeTri( freeTriIdx );

            // Get the next vertex in the hole boundary
            const int oppVi    = ( curVi + 2 ) % 3;
            const Tri& curTri  = _triVec[ curTriIdx ];
            const int nextVert = curTri._v[ ( curVi + 1 ) % 3 ]; 

            // New triangle
            const Tri newTri = { insVert, nextVert, curVert };

            // Adjancency with opposite triangle

            TriOpp& curTriOpp = _triOppVec[ curTriIdx ];
            curTriOpp.setOpp( oppVi, newTriIdx, 0 );

            TriOpp& newTriOpp = _triOppVec[ newTriIdx ];
            newTriOpp.setOpp( 0, curTriIdx, oppVi );

            // Adjacency with previous new triangle

            TriOpp& prevTriOpp = _triOppVec[ prevNewTriIdx ];
            prevTriOpp.setOpp( 2, newTriIdx, 1 );
            newTriOpp.setOpp( 1, prevNewTriIdx, 2 );
                
            // Last hole triangle
            if ( nextVert == firstVert )
            {
                TriOpp& firstTriOpp = _triOppVec[ firstNewTriIdx ];
                firstTriOpp.setOpp( 1, newTriIdx, 2 ); 
                newTriOpp.setOpp( 2, firstNewTriIdx, 1 );
            }

            // Store new triangle data
            _triVec[ newTriIdx ]       = newTri;
            _tetIdxVec[ newTriIdx ]    = -1;
            _triStatusVec[ newTriIdx ] = TriNew;

            // Add facet to queue
            addOneTriToQue( newTriIdx, newTri );

            // Prepare for next triangle
            prevNewTriIdx = newTriIdx; 

            // Move to the next vertex
            curVi   = ( curVi + 1 ) % 3; 
            curVert = nextVert; 
        }
    }

    changeNewToValid(); // Can be removed after rewriting stitching code

    return;
}

void Star::getProof
(
int   inVert, // Point that lies inside star
int*  proofArr
)
{
    // Pick one triangle as facet intersected by plane

    int locTriIdx = 0;
    
    for ( ; locTriIdx < ( int ) _triVec.size(); ++locTriIdx )
    {
        if ( TriFree != _triStatusVec[ locTriIdx ] ) break;
    }

    // Pick this triangle

    const Tri& firstTri = _triVec[ locTriIdx ];
    const int exVert    = firstTri._v[ 0 ]; // First proof point

    // Iterate through triangles to find another triangle
    // intersected by plane of (star, inVert, exVert)

    for ( ; locTriIdx < ( int ) _triVec.size(); ++locTriIdx )
    {
        if ( TriFree == _triStatusVec[ locTriIdx ] ) continue;

        // Ignore triangle if it has exVert

        const Tri tri = _triVec[ locTriIdx ];
        
        if ( tri.has( exVert ) ) continue;

        Orient ord[3];
        int vi = 0; 

        // Iterate through vertices in order
        for ( ; vi < 3; ++vi )
        {
            const int planeVert = tri._v[ vi ];
            const int testVert  = tri._v[ ( vi + 1 ) % 3 ];

            // Get order of testVert against the plane formed by (inVert, starVert, exVert, planeVert)
            ord[ vi ] = _predWrapper.doOrient4DAdaptSoS( 
				makeTet( _vert, inVert, exVert, planeVert ), testVert );

            // Check if orders match, they do if plane intersects facet
            if ( ( vi > 0 ) && ( ord[ vi - 1 ] != ord[ vi ] ) ) break;
        }

        // All the orders match, we got our proof
        if ( vi >= 3 ) break;
    }

	assert( locTriIdx < _triVec.size() ); 

    // Write proof vertices

    proofArr[0] = exVert;

    const Tri proofTri = _triVec[ locTriIdx ];

    for ( int vi = 0; vi < 3; ++vi )
        proofArr[ vi + 1 ] = proofTri._v[vi];
    
    return;
}

bool Star::hasLinkVert
(
int inVert
) const
{
    // Use triangles
    for ( int triIdx = 0; triIdx < ( int ) _triVec.size(); ++triIdx )
    {
        if ( TriFree == _triStatusVec[ triIdx ] ) continue;
        if ( _triVec[ triIdx ].has( inVert ) )    return true;
    }

    return false;
}

// Assumption: Vert is not on the link!
int Star::locateVert
(
int inVert
) const
{
    int triIdx = 0; 

    for ( ; triIdx < _triVec.size(); ++triIdx ) 
        if ( TriFree != _triStatusVec[ triIdx ] ) break; 

    if ( triIdx >= _triVec.size() ) return -1;   // No alive triangles

    int prevVi = -1; 

//    std::cout << "\nWalking: "; 

    // Start walking
    while (true) 
    {
        Tri tri = _triVec[ triIdx ]; 

        //if ( tri.has( inVert ) ) return triIdx;     // Found

        // Check 3 sides
        int vi = 0; 

        for ( ; vi < 3; ++vi ) 
        {
            if ( vi == prevVi ) continue;   // Skip the in-coming direction

            Orient ori = _predWrapper.doOrient3DSoS( tri._v[ ( vi + 1 ) % 3 ], tri._v[ (vi + 2) % 3 ],
                inVert, _vert ); 
           
            if ( OrientNeg == ori ) 
                break; 
        }

        if ( vi >= 3 )  // Found the containing cone
            break; 

        const int oppTi = _triOppVec[ triIdx ].getOppTri( vi ); 
        const int oppVi = _triOppVec[ triIdx ].getOppVi( vi ); 

        triIdx = oppTi; 
        prevVi = oppVi; 
    }

    return triIdx; 
}

int Star::getLinkTriIdx( const Tri& inTri ) const
{
    for ( int triIdx = 0; triIdx < ( int ) _triVec.size(); ++triIdx )
    {
        if ( TriFree == _triStatusVec[ triIdx ] ) continue;

        const Tri tri = _triVec[ triIdx ];

        if ( tri.has( inTri._v[0] ) && tri.has( inTri._v[1] ) && tri.has( inTri._v[2] ) )
            return triIdx;
    }

    return -1;
}

void Star::clone( const Star& s )
{
    clear(); 

    _vert           = s._vert; 

    _triVec.assign( s._triVec.begin(), s._triVec.end() ); 
    _triOppVec.assign( s._triOppVec.begin(), s._triOppVec.end() ); 
    _tetIdxVec.assign( s._tetIdxVec.begin(), s._tetIdxVec.end() ); 
    _triStatusVec.assign( s._triStatusVec.begin(), s._triStatusVec.end() ); 
}

