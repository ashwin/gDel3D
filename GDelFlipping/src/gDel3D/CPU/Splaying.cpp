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

#include "Splaying.h"

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

Splaying::Splaying( const GDelParams& params ) 
    : _params( params )
{
}

// Recursive function
void Splaying::findTetInVecRec
(
int               inVert,
const Tet&        inTet,
int               tetIdx,
int               visitId,
int&              tetFoundIdx
)
{
    if ( -1 != tetFoundIdx )                 return;
    if ( visitId == _tetVisit[ tetIdx ] ) return;

    // Check if tetra matches

    const Tet& tet = (*_tetVec)[ tetIdx ];

    if (   tet.has( inTet._v[0] )
        && tet.has( inTet._v[1] )
        && tet.has( inTet._v[2] )
        && tet.has( inTet._v[3] ) )
    {
        tetFoundIdx = tetIdx;
        return; // Get out!
    }

    // Visited
    
    _tetVisit[ tetIdx ] = visitId;

    // Visit 3 neighbours

    const TetOpp& tetOpp = (*_oppVec)[ tetIdx ];
    const int botVi      = tet.getIndexOf( inVert );
    const int* ordVi     = TetViAsSeenFrom[ botVi ];

    for ( int vi = 0; vi < 3; ++vi )
    {
        const int neiTetIdx = tetOpp.getOppTet( ordVi[ vi ] );

        findTetInVecRec( inVert, inTet, neiTetIdx, visitId, tetFoundIdx );
    }
    
    return;
}

// Search tetra vector for input tetra
int Splaying::findTetInVec
(
const Tet&        inTet,
int               visitId
)
{
    int tetFoundIdx = -1;

    findTetInVecRec(
        inTet._v[0],
        inTet,
        (*_vertTetVec)[ inTet._v[0] ],
        visitId,
        tetFoundIdx
        );

    return tetFoundIdx;
}

void Splaying::tetVisitCreateStar( int startIdx, int vert, Star* star )
{
    _intStack.clear(); 
    _intStack.push_back( startIdx ); 

    _tetVisit[ startIdx ] = _visitId; 

    while ( _intStack.size() > 0 ) 
    {
        const int tetIdx = _intStack.back(); 
        _intStack.pop_back(); 

        // Cut this tetra away from the triangulation
        // Can bring it back if needed. 
        setTetAliveState( (*_tetInfoVec)[ tetIdx ], false );

        // Make link triangle from tetra    
        const Tet& tet       = (*_tetVec)[ tetIdx ];
        const TetOpp& tetOpp = (*_oppVec)[ tetIdx ];

        //assert( tet.has( vert ) ); 

        const int botVi      = tet.getIndexOf( vert );
        const int* ordVi     = TetViAsSeenFrom[ botVi ];

        const TriOpp triOpp = { tetOpp._t[ ordVi[0] ], tetOpp._t[ ordVi[1] ], tetOpp._t[ ordVi[2] ] };

        // Add opp to star  
        star->_triOppVec.push_back( triOpp );

        // Note idx before it is pushbacked below
        _tetTriMap[ tetIdx ] = encode( ( int ) star->_triVec.size(), botVi ); 

        // Add link triangle to star
        const Tri tri = { tet._v[ ordVi[0] ], tet._v[ ordVi[1] ], tet._v[ ordVi[2] ] };

        star->_triVec.push_back( tri );
        star->_tetIdxVec.push_back( tetIdx );
        star->_triStatusVec.push_back( TriValid );

        // Visit neighbours
        for ( int vi = 0; vi < 3; ++vi )
        {
            int oppIdx = triOpp.getOppTri( vi ); 

            if ( _visitId != _tetVisit[ oppIdx ] )
            {
                _tetVisit[ oppIdx ] = _visitId; // Visited
                _intStack.push_back( oppIdx ); 
            }
        }
    }
    
    return;
}

void Splaying::makeOppLocal( Star *star )
{
    for ( int triIdx = 0; triIdx < ( int ) star->_triVec.size(); ++triIdx )
    {
        assert( TriFree != star->_triStatusVec[ triIdx ] );

        TriOpp& triOpp = star->_triOppVec[ triIdx ];

        for ( int vi = 0; vi < 3; ++vi )
        {
            const int oppTetIdx = triOpp.getOppTri( vi );
            const int oppTetVi  = triOpp.getOppVi( vi );

            // TriIdx
            int oppTriIdx, oppBotVi; 
            
            decode( _tetTriMap[ oppTetIdx ], &oppTriIdx, &oppBotVi ); 
            
            const int oppTriVi = TetNextViAsSeenFrom[ oppBotVi ][ oppTetVi ];

            triOpp.setOppTri( vi, oppTriIdx ); 
            triOpp.setOppVi( vi, oppTriVi );
        }
    }

    return;
}

Star* Splaying::createFromTetra( int vert )
{
    Star *star = new Star( _predWrapper ); 

    star->clear(); 
    star->_vert = vert;

    tetVisitCreateStar( (*_vertTetVec)[ vert ], vert, star );

    makeOppLocal( star );

    ++_visitId; 

    return star;
}

void Splaying::makeFailedStarsAndQueue(
    IntHVec   &failVertVec
    )
{
    PerfTimer timer; 

    double totalTime     = 0.0; 
    double timeFlip      = 0.0; 
    double timeCreate    = 0.0; 
    double timeConstruct = 0.0; 
    double timeInit      = 0.0; 

    // Init vectors
    _starVec.assign( _predWrapper.pointNum(), ( Star* ) NULL );
    _tetVisit.assign( MAX( _tetVec->size(), _predWrapper.pointNum() ), -1 );

    timer.start(); 

    _stk.clear();

    _visitId = 0;
    
    // Get worksets and create stars
    for ( int vertIdx = 0; vertIdx < ( int ) failVertVec.size(); ++vertIdx )
    {
        const int failVert = failVertVec[ vertIdx ];

        _starVec[ failVert ] = createFromTetra( failVert ); 

        _starVec[ failVert ]->flipping( &_intStack, &_stk, &_tetVisit, _visitId++ ); 

        compareStarAddQueue( _starVec[ failVert ] ); 
    }

    // Active stars
    _actStarVec.assign( failVertVec.begin(), failVertVec.end() );
    
    timer.stop(); 
    totalTime = timer.value(); 

    if ( _params.verbose ) 
        std::cout << "  Initial facet items:            " << _stk.size() << std::endl;


    return;
}

void Splaying::compareStarAddQueue( Star* star )
{
    const int vert = star->_vert; 

    // Compare star with tetra
    for ( int triIdx = 0; triIdx < ( int ) star->_triVec.size(); ++triIdx )
    {
        if ( TriFree == star->_triStatusVec[ triIdx ] ) continue;
        if ( -1 != star->_tetIdxVec[ triIdx ] ) continue; 

        // Link triangle not in triangulation
        const Tri tri = star->_triVec[ triIdx ];

        Facet facet;
        facet._from     = vert;
        facet._fromIdx  = triIdx; 

        for ( int vi = 0; vi < 3; ++vi )
        {
            facet._to   = tri._v[ vi ];
            facet._v0   = tri._v[ ( vi + 1 ) % 3 ];
            facet._v1   = tri._v[ ( vi + 2 ) % 3 ];

            _stk.push_back( facet );
        }
    }

    return;
}

void Splaying::processQue()
{
    PerfTimer timer;  

    double timeTotal  = 0.0; 
    double timeInsert = 0.0; 
    double timeCheck  = 0.0; 
    double timeProof  = 0.0; 
    double timeCreate = 0.0; 

    int proofArr[4];

    IntHVec vertVec;
    vertVec.reserve( StarAvgTriNum );

    IntHVec workVec;

    int count       = 0;
    int validCount  = 0;
    int proofCount  = 0; 
    int insCount    = 0; 
    int totDeg      = 0; 

    timer.start(); 

    while ( !_stk.empty() )
    {
        ++count;

        const Facet facet = _stk.back();
        _stk.pop_back();

        // Get from-to stars
        const int fromVert = facet._from;
        const int toVert   = facet._to;

		if ( facet._fromIdx == -1 )		// A drowned vertex
		{
			if ( NULL == _starVec[ fromVert ] )
			{
				_starVec[ fromVert ] = createFromTetra( fromVert ); 

				_actStarVec.push_back( fromVert );
			}

			Star* fromStar	= _starVec[ fromVert ];
			Star* toStar	= _starVec[ toVert ];

			assert( toStar != NULL ); 

			// Already deleted?
            if ( !fromStar->hasLinkVert( toVert ) ) continue;  

            toStar->getProof( fromVert, proofArr );
            
			++proofCount; 

			for ( int pi = 0; pi < 4; ++pi )
			{
				const int proofVert = proofArr[pi];

				if ( fromVert == proofVert )              continue;
				
                if ( fromStar->hasLinkVert( proofVert ) ) continue; 

                totDeg += fromStar->_triVec.size(); 

                fromStar->insertToStar( proofVert, &_stk, &_intStack, &_tetVisit, _visitId++ );

				++insCount; 
			}
		}
		else
		{
			Star* fromStar = _starVec[ fromVert ];

			assert( fromStar != NULL ); 

			// Check if facet is still in from-star
			if ( TriFree == fromStar->_triStatusVec[ facet._fromIdx ]
				|| !fromStar->_triVec[ facet._fromIdx ].has( toVert, facet._v0, facet._v1 ) )
				continue;

			++validCount; 

			// Check if to- has triangle
			if ( NULL == _starVec[ toVert ] )
			{
                _starVec[ toVert ] = createFromTetra( toVert ); 

				_actStarVec.push_back( toVert );
			}

			Star* toStar = _starVec[ toVert ];

			const Tri checkTri = { fromVert, facet._v0, facet._v1 };

			for ( int vi = 0; vi < 3; ++vi )
			{
				const int vert = checkTri._v[vi];

                if ( toStar->hasLinkVert( vert ) ) continue;  

				++insCount; 

                if ( toStar->insertToStar( vert, &_stk, &_intStack, &_tetVisit, _visitId++ ) ) 
                    continue;  // Successful insertion

                toStar->getProof( vert, proofArr );

				++proofCount; 

				for ( int pi = 0; pi < 4; ++pi )
				{
					const int proofVert = proofArr[pi];

					if ( fromVert == proofVert )              continue;

                    if ( fromStar->hasLinkVert( proofVert ) ) continue;  

                    totDeg += fromStar->_triVec.size(); 

                    fromStar->insertToStar( proofVert, &_stk, &_intStack, &_tetVisit, _visitId++ );

					++insCount; 
				}
    
				break; 
			}
		}
    }

    timer.stop(); 

    timeTotal = timer.value(); 

    if ( _params.verbose )
    {
        std::cout << "  Processed facet items:          " << count << std::endl; 
        std::cout << "  Actual valid facet items:       " << validCount << std::endl; 
        std::cout << "  Insertion due to inconsistency: " << insCount << std::endl; 
        std::cout << "  Proofs generated:               " << proofCount << std::endl; 
    }
     
    return;
}

void Splaying::checkStarConsistency()
{
    std::cout << "\nChecking for star consistency...\n"; 
    for ( int fromVert = 0; fromVert < ( int ) _starVec.size(); ++fromVert )
    {
        const Star* star = _starVec[ fromVert ];

        if ( NULL == star ) continue;

        for ( int triIdx = 0; triIdx < ( int ) star->_triVec.size(); ++triIdx )
        {
            if ( TriFree == star->_triStatusVec[ triIdx ] ) continue;

            const Tri tri = star->_triVec[ triIdx ];
            const Tet tet = makeTet( tri._v[0], tri._v[1], tri._v[2], fromVert );

            // Check tetra in 3 other stars
            for ( int vi = 0; vi < 3; ++vi )
            {
                const int toVert   = tet._v[vi];
                const Star* toStar = _starVec[ toVert ];

                if ( NULL == toStar )
                {
                    if ( -1 == findTetInVec( tet, _visitId++ ) )
                        std::cout << "Tetra not found in triangulation\n";
                }
                else
                {
                    const int* ordVi = TetViAsSeenFrom[ vi ];
                    const Tri toTri  = { tet._v[ ordVi[0] ], tet._v[ ordVi[1] ], tet._v[ ordVi[2] ] };

                    if ( -1 == toStar->getLinkTriIdx( toTri ) )
                        std::cout << "Triangle not found in star: " << toVert
							<< " ( " << toTri._v[0] << " " << toTri._v[1] << " " 
							<< toTri._v[2] << " ) <--- " << fromVert << std::endl;
                }
            }
        }
    }

    return;
}

void Splaying::starsToTetra()
{
    if ( _params.verbose )
        std::cout << "  Final stars:                    " << _actStarVec.size() << "\n\n";

    int newTet = 0; 
    int delTet = 0; 

    PerfTimer timer; 

    timer.start(); 

    for ( int idx = 0; idx < ( int ) _actStarVec.size(); ++idx )
    {
        const int vert = _actStarVec[ idx ];
        Star* star     = _starVec[ vert ];

        // Iterate link triangles
        for ( int triIdx = 0; triIdx < ( int ) star->_triVec.size(); ++triIdx )
        {
            if ( TriFree == star->_triStatusVec[ triIdx ] ) continue;

            const Tri tri = star->_triVec[ triIdx ];
            int tetIdx    = star->_tetIdxVec[ triIdx ];

            if ( -1 != tetIdx ) 
            {
                setTetAliveState( (*_tetInfoVec)[ tetIdx ], true ); 
                continue;
            }

            // Find tet in 3 other stars
            int tetInStar[ 3 ]; 

            for ( int vi = 0; vi < 3; ++vi )
            {
                const int toVert = tri._v[ vi ];
                const Tri toTri  = { vert, tri._v[ ( vi + 1 ) % 3 ], tri._v[ ( vi + 2 ) % 3 ] };
                Star* toStar     = _starVec[ toVert ];

                assert( toStar != NULL && "This star cannot be unavailable." ); 

                const int triIdx = toStar->getLinkTriIdx( toTri );

                assert( triIdx != -1 && "Stars are not consistent!" ); 

                tetInStar[ vi ]  = triIdx; 

                if ( toStar->_tetIdxVec[ triIdx ] != -1 ) 
                    tetIdx = toStar->_tetIdxVec[ triIdx ]; 
            }

            if ( tetIdx == -1 ) 
            {
                // Create tetra
                tetIdx = ( int ) _tetVec->size();

                // Set data
                const Tet tet = makeTet( tri._v[0], tri._v[1], tri._v[2], vert );

                _tetVec->push_back( tet );
                _oppVec->push_back( makeTetOpp( -1, -1, -1, -1 ) );
                _tetInfoVec->push_back( 1 );

                ++newTet; 
            }
           
            star->_tetIdxVec[ triIdx ] = tetIdx;

            for ( int vi = 0; vi < 3; ++vi ) 
                _starVec[ tri._v[ vi ] ]->_tetIdxVec[ tetInStar[ vi ] ] = tetIdx; 

            // Set opp in 3 neighbours
            const TriOpp triOpp = star->_triOppVec[ triIdx ];
            const Tet tet       = (*_tetVec)[ tetIdx ];
            TetOpp& tetOpp      = (*_oppVec)[ tetIdx ];

            for ( int vi = 0; vi < 3; ++vi )
            {
                const int oppTriIdx    = triOpp.getOppTri( vi );
                const int oppTriVi     = triOpp.getOppVi( vi );
                const int oppTriTetIdx = star->_tetIdxVec[ oppTriIdx ];

                // Opp's tetIdx not available? Don't worry, he'll set ours. 
                if ( -1 == oppTriTetIdx ) continue;

                const int triVert  = tri._v[ vi ];
                const int curTetVi = tet.getIndexOf( triVert );

                const Tri oppTri   = star->_triVec[ oppTriIdx ];
                const int oppVert  = oppTri._v[ oppTriVi ];
                const Tet oppTet   = (*_tetVec)[ oppTriTetIdx ];
                const int oppTetVi = oppTet.getIndexOf( oppVert );

                // Set both ways
                TetOpp& oppTetOpp = (*_oppVec)[ oppTriTetIdx ];

                tetOpp.setOpp( curTetVi, oppTriTetIdx, oppTetVi );
                oppTetOpp.setOpp( oppTetVi, tetIdx, curTetVi );
            }

            // Set opp in 4th neighbour

            const Star* star2 = _starVec[ tri._v[0] ];
            const int triIdx2 = tetInStar[ 0 ]; 
            const int tetIdx2 = star2->_tetIdxVec[ triIdx2 ];

            const int vi2           = star2->_triVec[ triIdx2 ].indexOf( vert );
            const TriOpp triOpp2    = star2->_triOppVec[ triIdx2 ];
            const int oppTriIdx2    = triOpp2.getOppTri( vi2 );
            const int oppTriVi2     = triOpp2.getOppVi( vi2 );
            const int oppTriTetIdx2 = star2->_tetIdxVec[ oppTriIdx2 ];

            if ( -1 == oppTriTetIdx2 ) continue;

            const int oppVert  = star2->_triVec[ oppTriIdx2 ]._v[ oppTriVi2 ];
            const int oppTetVi = (*_tetVec)[ oppTriTetIdx2 ].getIndexOf( oppVert );
            const int curTetVi = tet.getIndexOf( vert );

            // Set both ways
            TetOpp& oppTetOpp2 = (*_oppVec)[ oppTriTetIdx2 ];

            tetOpp.setOpp( curTetVi, oppTriTetIdx2, oppTetVi );
            oppTetOpp2.setOpp( oppTetVi, tetIdx, curTetVi );
        }
    }

    timer.stop(); 

    return;
}

void Splaying::freeStars()
{
    for ( int vert = 0; vert < ( int ) _starVec.size(); ++vert )
    {
        if ( NULL != _starVec[ vert ] )
        {
            delete _starVec[ vert ];
            _starVec[ vert ] = NULL;
        }
    }

    StarPtrHVec().swap( _starVec ); 
    IntHVec().swap( _actStarVec ); 
    FacetHVec().swap( _stk ); 
    IntHVec().swap( _tetTriMap ); 

    return;
}

void Splaying::init( 
    const Point3HVec&   pointVec, 
    GDelOutput*         output
    )
{
    _predWrapper.init( pointVec, output->ptInfty ); 

    _tetVec      = &output->tetVec; 
    _oppVec      = &output->tetOppVec; 
    _tetInfoVec  = &output->tetInfoVec;
    _vertTetVec  = &output->vertTetVec; 
    _output      = output; 

    _tetTriMap.resize( _tetVec->size() );

    output->stats.failVertNum = output->failVertVec.size(); 

	_starVec.clear();
    _actStarVec.clear();
    _stk.clear();
}

void Splaying::fixWithStarSplaying( 
    const Point3HVec&   pointVec, 
    GDelOutput*         output
    )
{
    if ( _params.verbose ) 
        std::cout << "Star splaying: " << std::endl; 

    init( pointVec, output ); 

    PerfTimer timer;

    timer.start();
        if ( output->stats.failVertNum > 0 ) 
        {
            makeFailedStarsAndQueue( output->failVertVec );
            processQue();
            //checkStarConsistency();
            starsToTetra();
        }
    timer.stop(); 

    // Output
    _output->stats.splayingTime = timer.value(); 
    _output->stats.finalStarNum = ( int ) _actStarVec.size();
    _output->stats.totalTime   += _output->stats.splayingTime; 

    freeStars();

    return;
}

