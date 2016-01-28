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

#include "HostToKernel.h"
#include "KerCommon.h"
#include "KerDivision.h"

__global__ void
kerMakeFirstTetra
(
Tet*	tetArr,
TetOpp*	oppArr,
char*	tetInfoArr,
Tet     tet,
int     tetIdx,
int     infIdx
)
{
	const Tet tets[] = {
		{ tet._v[0], tet._v[1], tet._v[2], tet._v[3] }, 
		{ tet._v[1], tet._v[2], tet._v[3], infIdx }, 
		{ tet._v[0], tet._v[3], tet._v[2], infIdx }, 
		{ tet._v[0], tet._v[1], tet._v[3], infIdx },
		{ tet._v[0], tet._v[2], tet._v[1], infIdx }
	};

	const int oppTet[][4] = {
		{ 1, 2, 3, 4 }, 
		{ 2, 3, 4, 0 }, 
		{ 1, 4, 3, 0 }, 
		{ 1, 2, 4, 0 }, 
		{ 1, 3, 2, 0 }
	};

    const int oppVi[][4] = {
        { 3, 3, 3, 3 }, 
        { 0, 0, 0, 0 }, 
        { 0, 2, 1, 1 }, 
        { 1, 2, 1, 2 }, 
        { 2, 2, 1, 3 }
    }; 

	for ( int i = 0; i < 5; ++i ) 
	{
		tetArr[ tetIdx + i ]     = tets[ i ]; 
		tetInfoArr[ tetIdx + i ] = 1; 

        setTetEmptyState( tetInfoArr[ tetIdx + i ], false ); 

        TetOpp opp = { -1, -1, -1, -1 }; 

		for ( int j = 0; j < 4; ++j ) 
			opp.setOpp( j, tetIdx + oppTet[i][j], oppVi[i][j] ); 

		oppArr[ tetIdx + i ] = opp; 
	}
}

__global__ void
kerSplitTetra
(
KerIntArray newVertVec,
KerIntArray insVertVec, 
int*        vertArr, 
int*        vertTetArr,
int*        tetToVert,
Tet*        tetArr,
TetOpp*     oppArr,
char*       tetInfoArr,
int*        freeArr,
int*        vertFreeArr,
int         infIdx
)
{
    // Iterate current tetra
    for ( int idx = getCurThreadIdx(); idx < newVertVec._num; idx += getThreadNum() )
    {
        const int insIdx      = newVertVec._arr[ idx ]; 

        const int tetIdx      = makePositive( vertTetArr[ insIdx ] ); 
        const int splitVertex = vertArr[ insIdx ];
        const int newIdx      = ( splitVertex + 1 ) * MeanVertDegree - 1; 
        const int newTetIdx[4]  = { freeArr[ newIdx ], freeArr[ newIdx - 1 ], freeArr[ newIdx - 2 ], freeArr[ newIdx - 3 ] };

        // Update vertFree, 4 has been used.
        vertFreeArr[ splitVertex ] -= 4; 

        // Create new tets
        const TetOpp oldOpp = loadOpp( oppArr, tetIdx );
        const Tet tet       = loadTet( tetArr, tetIdx ); // Note: This slot will be overwritten below

        for ( int vi = 0; vi < 4; ++vi ) 
        {
            TetOpp newOpp = { -1, -1, -1, -1 }; 

            // Set internal adjacency
            newOpp.setOppInternal( 0, newTetIdx[ IntSplitFaceOpp[vi][0] ], IntSplitFaceOpp[vi][1] ); 
            newOpp.setOppInternal( 1, newTetIdx[ IntSplitFaceOpp[vi][2] ], IntSplitFaceOpp[vi][3] ); 
            newOpp.setOppInternal( 2, newTetIdx[ IntSplitFaceOpp[vi][4] ], IntSplitFaceOpp[vi][5] ); 

            // Set external adjacency
            //if ( -1 != oldOpp._t[ vi ] )
            {
                int neiTetIdx = oldOpp.getOppTet( vi );
                int neiTetVi  = oldOpp.getOppVi( vi );

                // Check if neighbour has split
                const int neiSplitIdx = tetToVert[ neiTetIdx ];

                if ( neiSplitIdx == INT_MAX ) // Neighbour is un-split
                {
                    oppArr[ neiTetIdx ].setOpp( neiTetVi, newTetIdx[ vi ], 3 ); // Point un-split neighbour back to this new tetra
                }
                else // Neighbour has split
                {
                    // Get neighbour's new split tetra that has this face
                    const int neiSplitVert  = vertArr[ neiSplitIdx ]; 
                    const int neiFreeIdx    = ( neiSplitVert + 1 ) * MeanVertDegree - 1; 

                    neiTetIdx = freeArr[ neiFreeIdx - neiTetVi ];
                    neiTetVi  = 3;
                }

                newOpp.setOpp( 3, neiTetIdx, neiTetVi ); // Point this tetra to neighbour
            }

            // Write split tetra and opp
            const Tet newTet = {
                tet._v[ TetViAsSeenFrom[vi][0] ],
                tet._v[ TetViAsSeenFrom[vi][1] ],
                tet._v[ TetViAsSeenFrom[vi][2] ],
                splitVertex
            };

            const int toTetIdx = newTetIdx[ vi ];
            storeTet( tetArr, toTetIdx, newTet );
            storeOpp( oppArr, toTetIdx, newOpp );
            
            setTetAliveState( tetInfoArr[ toTetIdx ], true ); 
            setTetCheckState( tetInfoArr[ toTetIdx ], Changed ); 
        }

        //*** Donate one tetra
        const int blkIdx  = tetIdx / MeanVertDegree; 
        const int vertIdx = ( blkIdx < insVertVec._num ) ? insVertVec._arr[ blkIdx ] : infIdx;  
        const int freeIdx = atomicAdd( &vertFreeArr[ vertIdx ], 1 ); 

        freeArr[ vertIdx * MeanVertDegree + freeIdx ] = tetIdx;

        setTetAliveState( tetInfoArr[ tetIdx ], false ); 
    }

    return;
}

// Note: tetVoteArr should *not* be modified here
__global__ void
kerMarkRejectedFlips
(
KerIntArray actTetVec,
TetOpp*     oppArr,
int*        tetVoteArr,
char*       tetInfoArr,
int*        voteArr,
int*        flipToTet,
int*        counterArr,
int         voteOffset
)
{
    __shared__ int s_flipNum, s_flipOffset; 

    int actTetNumRounded    = actTetVec._num; 

    if ( flipToTet != voteArr )  // Compact flips
    { 
        if ( threadIdx.x == 0 ) 
            s_flipNum = 0; 

        actTetNumRounded = roundUp( actTetVec._num, blockDim.x ); 

        __syncthreads(); 
    }

    for ( int idx = getCurThreadIdx(); idx < actTetNumRounded; idx += getThreadNum() )
    {
        int flipVal = -1; 

        const int tetIdx = ( idx < actTetVec._num ) ? actTetVec._arr[ idx ] : -1; 

        if ( tetIdx != -1 ) 
        {
            flipVal = voteArr[ idx ];

            if ( flipVal == -1 )    // No flip from this tet
            {
                setTetCheckState( tetInfoArr[ tetIdx ], Checked );
                actTetVec._arr[ idx ] = -1; 
            } 
            else 
            {
                const int voteVal       = voteOffset + tetIdx; 
                const int botVoteVal    = tetVoteArr[ tetIdx ]; 

                if ( botVoteVal != voteVal ) 
                    flipVal = -1; 
                else
                {
                    const char flipInfo  = getVoteFlipInfo( flipVal );
                    const FlipType fType = getFlipType( flipInfo );
                    const TetOpp& opp    = oppArr[ tetIdx ];
                    const int botVi      = getFlipBotVi( flipInfo );

                    // Top
                    const int topTetIdx  = opp.getOppTet( botVi );
                    const int topVoteVal = tetVoteArr[ topTetIdx ];

                    if ( topVoteVal != voteVal )
                        flipVal = -1; 
                    else if ( Flip32 == fType )
                    {
                        // Corner
                        const int* ordVi   = TetViAsSeenFrom[ botVi ];
                        const int corOrdVi = getFlipBotCorOrdVi( flipInfo );
                        const int botCorVi = ordVi[ corOrdVi ];

                        // Side
                        const int sideTetIdx  = opp.getOppTet( botCorVi );
                        const int sideVoteVal = tetVoteArr[ sideTetIdx ];

                        if ( sideVoteVal != voteVal )
                            flipVal = -1; 
                    }
                }

                if ( flipVal == -1 )
                    voteArr[ idx ] = -1; 
            }
        }

        if ( flipToTet != voteArr )  // Compact flips
        {
            int flipLocIdx = ( flipVal == -1 ? -1 : atomicAdd( &s_flipNum, 1 ) ); 

            __syncthreads(); 

            if ( s_flipNum > 0 ) 
            {
                if ( threadIdx.x == 0 ) 
                    s_flipOffset = atomicAdd( &counterArr[ CounterFlip ], s_flipNum ); 

                __syncthreads(); 

                if ( flipLocIdx != -1 ) 
                    flipToTet[ s_flipOffset + flipLocIdx ] = flipVal; 

                if ( threadIdx.x == 0 ) 
                    s_flipNum = 0; 

                __syncthreads(); 
            }
        }
    }

    if ( blockIdx.x == 0 && THREAD_IDX == 0 )
    {
        counterArr[ CounterExact ] = 0; 
    }

    return;
}

__global__ void
kerPickWinnerPoint
(
KerIntArray  vertexArr,
int*         vertexTetArr,
int*         vertSphereArr,
int*         tetSphereArr,
int*         tetVertArr
)
{
    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
    {
        const int vertSVal = vertSphereArr[ idx ];
        const int tetIdx   = vertexTetArr[ idx ];
        const int winSVal  = tetSphereArr[ tetIdx ];

        // Check if vertex is winner

        if ( winSVal == vertSVal )
            atomicMin( &tetVertArr[ tetIdx ], idx );
    }

    return;
}

//////////////// Some helper functions used in flipping and updating opps. 
__forceinline__ __device__ void setTetIdxVi( int &output, int oldVi, int ni, int newVi )
{
    output -= ( 0xF ) << ( oldVi * 4 );
    output += ( ( ni << 2) + newVi ) << ( oldVi * 4 ); 
}

__forceinline__ __device__ int getTetIdx( int input, int oldVi )
{
    int idxVi = ( input >> ( oldVi * 4 ) ) & 0xf; 

    return ( idxVi >> 2 ) & 0x3; 
}

__forceinline__ __device__ int getTetVi( int input, int oldVi )
{
    int idxVi = ( input >> ( oldVi * 4 ) ) & 0xf; 

    return idxVi & 0x3; 
}

__global__ void
kerFlip
(
KerIntArray flipToTet,
Tet*        tetArr,
TetOpp*     oppArr,
char*       tetInfoArr,
int2*       tetMsgArr,
FlipItem*   flipArr,
int*        flip23NewSlot, 
int*        vertFreeArr, 
int*        freeArr,
int*        actTetArr,
KerIntArray insVertVec,
int         infIdx,
int         orgFlipNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipToTet._num; flipIdx += getThreadNum() )
    {
        const int voteVal    = flipToTet._arr[ flipIdx ];
        const char flipInfo  = getVoteFlipInfo( voteVal );
        const FlipType fType = getFlipType( flipInfo );

        CudaAssert( FlipNone != fType );

        // Bottom tetra
        const int botTetIdx   = getVoteTetIdx( voteVal );
        const int botCorOrdVi = getFlipBotCorOrdVi( flipInfo );
        const int botTetVi    = getFlipBotVi( flipInfo );
        Tet botTet            = loadTet( tetArr, botTetIdx );
        const TetOpp& botOpp  = loadOpp( oppArr, botTetIdx );

        // Top tetra
        const int topTetIdx     = botOpp.getOppTet( botTetVi );
        const int topTetVi      = botOpp.getOppVi( botTetVi );
        Tet topTet              = loadTet( tetArr, topTetIdx );
        const int globFlipIdx   = orgFlipNum + flipIdx; 

        int encodedFaceVi = 0; 
        int sideTetIdx; 

        const int* corVi    = TetViAsSeenFrom[ botTetVi ];
        const int* corTopVi = TetViAsSeenFrom[ topTetVi ];

        int newIdx[3]   = { 0xFFFF, 0xFFFF, 0xFFFF }; 
    
        if ( Flip23 == fType )
        {
            const int corV[3] = { botTet._v[ corVi[0] ], botTet._v[ corVi[1] ], botTet._v[ corVi[2] ] };
            int topVi0        = TetNextViAsSeenFrom[ topTetVi ][ topTet.getIndexOf( corV[ 2 ] ) ];

            const int oldFaceVi[3][2] = {
                { corVi[ 2 ], corTopVi[ topVi0 ] },  // Old bottom tetra
                { corVi[ 0 ], corTopVi[ ( topVi0 + 2 ) % 3 ] },  // Old top tetra
                { corVi[ 1 ], corTopVi[ ( topVi0 + 1 ) % 3 ] } // Old side tetra
            };

            // Iterate new tetra
            for ( int ni = 0; ni < 3; ++ni )
            {
                // Set external face adjacencies
                setTetIdxVi( newIdx[ 0 ], oldFaceVi[ ni ][ 0 ], ni == 0 ? 3 : ni, 0 ); 
                setTetIdxVi( newIdx[ 1 ], oldFaceVi[ ni ][ 1 ], ni == 1 ? 3 : ni, 3 ); 

                encodedFaceVi = ( encodedFaceVi <<  4 ) 
                    | ( oldFaceVi[ ni ][ 0 ] << 2 ) 
                    | ( oldFaceVi[ ni ][ 1 ] ); 
            } // 3 new tetra

            // Create new tetra
            const int topVert = topTet._v[ topTetVi ];
            const int botVert = botTet._v[ botTetVi ];

            botTet = makeTet( topVert, corV[0], corV[1], botVert ); 
            topTet = makeTet( topVert, corV[1], corV[2], botVert ); 
       
            // Output the side tetra
            const Tet sideTet   = makeTet( topVert, corV[2], corV[0], botVert ); 
            sideTetIdx          = flip23NewSlot[ flipIdx ]; 
    
            storeTet( tetArr, sideTetIdx, sideTet ); 
        }
        else
        {
            // Side tetra       
            const int botCorVi   = corVi[ botCorOrdVi ];
            sideTetIdx           = botOpp.getOppTet( botCorVi );
            const int sideCorVi0 = botOpp.getOppVi( botCorVi );

            // BotVi
            const int botAVi = corVi[ ( botCorOrdVi + 1 ) % 3 ];
            const int botBVi = corVi[ ( botCorOrdVi + 2 ) % 3 ];
            const int botA   = botTet._v[ botAVi ];
            const int botB   = botTet._v[ botBVi ];

            // Top vi
            const int botCor    = botTet._v[ botCorVi ];
            const int topCorVi  = topTet.getIndexOf( botCor );
            const int topLocVi  = TetNextViAsSeenFrom[ topTetVi ][ topCorVi ];
            const int topAVi    = corTopVi[ ( topLocVi + 2 ) % 3 ]; 
            const int topBVi    = corTopVi[ ( topLocVi + 1 ) % 3 ]; 

            // Side vi
            const int sideCorVi1 = oppArr[ topTetIdx ].getOppVi( topCorVi );
            const int sideLocVi  = TetNextViAsSeenFrom[ sideCorVi0 ][ sideCorVi1 ];
            const int* sideOrdVi = TetViAsSeenFrom[ sideCorVi0 ];
            const int sideAVi    = sideOrdVi[ ( sideLocVi + 1 ) % 3 ]; 
            const int sideBVi    = sideOrdVi[ ( sideLocVi + 2 ) % 3 ]; 

            const int oldFaceVi[3][2] = {
                { botAVi, botBVi },  // Old bottom tetra
                { topAVi, topBVi },  // Old top tetra
                { sideAVi, sideBVi } // Old side tetra
            };

            // Set external face adjacencies
            for ( int ti = 0; ti < 3; ++ti ) // Iterate old tetra
            {
                setTetIdxVi( newIdx[ ti ], oldFaceVi[ti][0], 1 == ti ? 3 : 1, Flip32NewFaceVi[ti][0] ); 
                setTetIdxVi( newIdx[ ti ], oldFaceVi[ti][1], 0 == ti ? 3 : 0, Flip32NewFaceVi[ti][1] ); 

                encodedFaceVi = ( encodedFaceVi <<  4 ) 
                    | ( oldFaceVi[ ti ][ 0 ] << 2 ) 
                    | ( oldFaceVi[ ti ][ 1 ] ); 
            }

            // Write down the new tetra idx
            tetMsgArr[ sideTetIdx ] = make_int2( newIdx[ 2 ], globFlipIdx ); 

            // Vertices of old 3 tetra
            const int botTetV  = botTet._v[ botTetVi ];
            const int topTetV  = topTet._v[ topTetVi ];

            botTet = makeTet( botCor, topTetV, botTetV, botA ); 
            topTet = makeTet( botCor, botTetV, topTetV, botB ); 

            //*** Donate one tetra
            const int insIdx  = sideTetIdx / MeanVertDegree;
            const int vertIdx = ( insIdx < insVertVec._num ) ? insVertVec._arr[ insIdx ] : infIdx;  
            const int freeIdx = atomicAdd( &vertFreeArr[ vertIdx ], 1 ); 

            freeArr[ vertIdx * MeanVertDegree + freeIdx ] = sideTetIdx;
        }

        // Write down the new tetra idx
        tetMsgArr[ botTetIdx ] = make_int2( newIdx[ 0 ], globFlipIdx ); 
        tetMsgArr[ topTetIdx ] = make_int2( newIdx[ 1 ], globFlipIdx ); 

        // Update the bottom and top tetra
        storeTet( tetArr, botTetIdx, botTet ); 
        storeTet( tetArr, topTetIdx, topTet ); 

        // Store faceVi
        flip23NewSlot[ flipIdx ] = encodedFaceVi; 
        
        bool tetEmpty = isTetEmpty( tetInfoArr[ botTetIdx ] ) && 
            isTetEmpty( tetInfoArr[ topTetIdx ] ); 

        if ( fType == Flip32 && tetEmpty ) 
            tetEmpty = isTetEmpty( tetInfoArr[ sideTetIdx ] ); 

        // Record the flip
        FlipItem flipItem = { botTet._v[0], botTet._v[1], botTet._v[2], 
            botTet._v[3], topTet._v[ fType == Flip23 ? 2 : 3 ],
            botTetIdx, topTetIdx, 
            ( fType == Flip32 ) ? makeNegative( sideTetIdx ) : sideTetIdx };

        if ( tetEmpty ) 
            flipItem._v[ 0 ] = -1; 

        if ( actTetArr != NULL ) 
        {
            actTetArr[ flipIdx ] = 
                ( Checked == getTetCheckState( tetInfoArr[ topTetIdx ] ) )
                ? topTetIdx : -1;
            actTetArr[ flipToTet._num + flipIdx ] = 
                ( fType == Flip23 ) 
                ? sideTetIdx : -1; 
        }

        char botTetState    = 3;    // Alive + Changed
        char topTetState    = 3; 
        char sideTetState   = 3; 

        setTetEmptyState( botTetState, tetEmpty ); 
        setTetEmptyState( topTetState, tetEmpty ); 

        if ( fType == Flip23 ) 
            setTetEmptyState( sideTetState, tetEmpty ); 
        else
            setTetAliveState( sideTetState, false ); 

        tetInfoArr[ botTetIdx ]  = botTetState;
        tetInfoArr[ topTetIdx ]  = topTetState;
        tetInfoArr[ sideTetIdx ] = sideTetState;

        storeFlip( flipArr, globFlipIdx, flipItem ); 
    }

    return;
}

__global__ void
kerUpdateOpp
(
FlipItem*    flipVec,
TetOpp*      oppArr,
int2*        tetMsgArr,
int*         encodedFaceViArr,
int          orgFlipNum,
int          flipNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipNum; flipIdx += getThreadNum() )
    {
        FlipItemTetIdx flipItem = loadFlipTetIdx( flipVec, flipIdx ); 
        FlipType fType          = ( flipItem._t[ 2 ] < 0 ) ? Flip32 : Flip23; 

        int encodedFaceVi = encodedFaceViArr[ flipIdx ]; 

        int     extOpp[6]; 
        TetOpp  opp; 

        opp = loadOpp( oppArr, flipItem._t[ 0 ] ); 

        if ( Flip23 == fType ) {
            extOpp[ 0 ] = opp.getOppTetVi( ( encodedFaceVi >> 10 ) & 3 ); 
            extOpp[ 2 ] = opp.getOppTetVi( ( encodedFaceVi >> 6 ) & 3 ); 
            extOpp[ 4 ] = opp.getOppTetVi( ( encodedFaceVi >> 2 ) & 3 ); 
        } else {
            extOpp[ 0 ] = opp.getOppTetVi( ( encodedFaceVi >> 10 ) & 3 ); 
            extOpp[ 1 ] = opp.getOppTetVi( ( encodedFaceVi >> 8 ) & 3 ); 
        }

        opp = loadOpp( oppArr, flipItem._t[ 1 ] ); 

        if ( Flip23 == fType ) {
            extOpp[ 1 ] = opp.getOppTetVi( ( encodedFaceVi >> 8 ) & 3 ); 
            extOpp[ 3 ] = opp.getOppTetVi( ( encodedFaceVi >> 4 ) & 3 ); 
            extOpp[ 5 ] = opp.getOppTetVi( ( encodedFaceVi >> 0 ) & 3 ); 
        } else {
            extOpp[ 2 ] = opp.getOppTetVi( ( encodedFaceVi >> 6 ) & 3 ); 
            extOpp[ 3 ] = opp.getOppTetVi( ( encodedFaceVi >> 4 ) & 3 ); 

            opp = loadOpp( oppArr, makePositive( flipItem._t[ 2 ] ) ); 

            extOpp[ 4 ] = opp.getOppTetVi( ( encodedFaceVi >> 2 ) & 3 ); 
            extOpp[ 5 ] = opp.getOppTetVi( ( encodedFaceVi >> 0 ) & 3 ); 
        }

        // Ok, update with neighbors
        for ( int i = 0; i < 6; ++i ) 
        {
            int newTetIdx, vi; 
            int tetOpp = extOpp[ i ]; 

            // No neighbor
            //if ( -1 == tetOpp ) continue; 

            int oppIdx = getOppValTet( tetOpp ); 
            int oppVi  = getOppValVi( tetOpp ); 
        
            const int2 msg = tetMsgArr[ oppIdx ]; 

            if ( msg.y < orgFlipNum )    // Neighbor not flipped
            {
                // Set my neighbor's opp
                if ( fType == Flip23 ) {
                    newTetIdx   = flipItem._t[ i / 2 ]; 
                    vi          = ( i & 1 ) ? 3 : 0; 
                } else {
                    newTetIdx   = flipItem._t[ 1 - ( i & 1 ) ]; 
                    vi          = Flip32NewFaceVi[ i / 2 ][ i & 1 ];
                }

                oppArr[ oppIdx ].setOpp( oppVi, newTetIdx, vi ); 
            }
            else
            {
                const int oppFlipIdx = msg.y - orgFlipNum; 

                // Update my own opp
                const int newLocOppIdx = getTetIdx( msg.x, oppVi ); 
                    
                if ( newLocOppIdx != 3 ) 
                    oppIdx = flipVec[ oppFlipIdx ]._t[ newLocOppIdx ]; 

                oppVi = getTetVi( msg.x, oppVi ); 

                extOpp[ i ] = makeOppVal( oppIdx, oppVi );
            }
        }

        // Now output
        if ( Flip23 == fType ) {
            opp._t[ 0 ] = extOpp[ 0 ]; 
            opp.setOppInternal( 1, flipItem._t[ 1 ], 2 ); 
            opp.setOppInternal( 2, flipItem._t[ 2 ], 1 ); 
            opp._t[ 3 ] = extOpp[ 1 ]; 
        }
        else {
            opp._t[ 1 ] = extOpp[ 1 ]; 
            opp._t[ 2 ] = extOpp[ 3 ]; 
            opp._t[ 0 ] = extOpp[ 5 ]; 
            opp.setOppInternal( 3, flipItem._t[ 1 ], 3 ); 
        }

        storeOpp( oppArr, flipItem._t[ 0 ], opp ); 

        if ( Flip23 == fType ) {
            opp._t[ 0 ] = extOpp[ 2 ]; 
            opp.setOppInternal( 2, flipItem._t[ 0 ], 1 ); 
            opp.setOppInternal( 1, flipItem._t[ 2 ], 2 ); 
            opp._t[ 3 ] = extOpp[ 3 ]; 
        }
        else {
            opp._t[ 2 ] = extOpp[ 0 ]; 
            opp._t[ 1 ] = extOpp[ 2 ]; 
            opp._t[ 0 ] = extOpp[ 4 ]; 
            opp.setOppInternal( 3, flipItem._t[ 0 ], 3 ); 
        }

        storeOpp( oppArr, flipItem._t[ 1 ], opp ); 

        if ( Flip23 == fType ) {
            opp._t[ 0 ] = extOpp[ 4 ]; 
            opp.setOppInternal( 1, flipItem._t[ 0 ], 2 ); 
            opp.setOppInternal( 2, flipItem._t[ 1 ], 1 ); 
            opp._t[ 3 ] = extOpp[ 5 ]; 

            storeOpp( oppArr, flipItem._t[ 2 ], opp ); 
        }
    }   

    return;
}

__global__ void
kerGatherFailedVerts
(
KerTetArray  tetVec,
TetOpp*      tetOppArr,
int*         failVertArr,
int*         vertTetArr
)
{
    for ( int tetIdx = getCurThreadIdx(); tetIdx < tetVec._num; tetIdx += getThreadNum() )
    {
        const TetOpp tetOpp = loadOpp( tetOppArr, tetIdx );
        int failVi          = -1;
        int win             = 0; 

        // Get out immediately if > 1 failures
        for ( int vi = 0; vi < 4; ++vi )
        {
            if ( tetOpp.getOppTet( vi ) < tetIdx ) 
                win |= ( 1 << vi ); 

            if ( !tetOpp.isOppSphereFail( vi ) ) continue;

            failVi = ( -1 == failVi ) ? vi : 4;
        }

        const Tet tet = loadTet( tetVec._arr, tetIdx );

        // Write
        for ( int vi = 0; vi < 4; ++vi )
        {
            int vert = tet._v[ vi ]; 

            if ( -1 != failVi && vi != failVi ) 
                failVertArr[ vert ] = vert;

            if ( ( win | ( 1 << vi ) ) == 0x0F )
                vertTetArr[ vert ] = tetIdx;
        }
    }

    return;
}

__global__ void 
kerUpdateFlipTrace
(
FlipItem*   flipArr, 
int*        tetToFlip,
int         orgFlipNum, 
int         flipNum
)
{
    for ( int idx = getCurThreadIdx(); idx < flipNum; idx += getThreadNum() )
    {
        const int flipIdx = orgFlipNum + idx; 
        FlipItem flipItem = loadFlip( flipArr, flipIdx ); 

        if ( flipItem._v[ 0 ] == -1 )   // All tets are empty, no need to trace
            continue; 

        int tetIdx, nextFlip; 

        tetIdx              = flipItem._t[ 0 ]; 
        nextFlip            = tetToFlip[ tetIdx ]; 
        flipItem._t[ 0 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip; 
        tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1; 

        tetIdx              = flipItem._t[ 1 ]; 
        nextFlip            = tetToFlip[ tetIdx ]; 
        flipItem._t[ 1 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip; 
        tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1; 

        tetIdx              = flipItem._t[ 2 ]; 

        if ( tetIdx < 0 ) 
        {
            tetIdx              = makePositive( tetIdx ); 
            tetToFlip[ tetIdx ] = ( flipIdx << 1 ) | 1; 
        }
        else
        {
            nextFlip            = tetToFlip[ tetIdx ]; 
            flipItem._t[ 2 ]    = ( nextFlip == -1 ) ? ( tetIdx << 1 ) | 0 : nextFlip; 
        }

        storeFlip( flipArr, flipIdx, flipItem ); 
    }
}

__global__ void
kerMarkTetEmpty
(
KerCharArray tetInfoVec
)
{
    for ( int idx = getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
        setTetEmptyState( tetInfoVec._arr[ idx ], true ); 
}

__global__ void 
kerUpdateVertIdx
(
KerTetArray tetVec,
int*        orgPointIdx
)
{
    for ( int idx = getCurThreadIdx(); idx < tetVec._num; idx += getThreadNum() )
    {
        Tet tet = loadTet( tetVec._arr, idx ); 

        for ( int i = 0; i < 4; ++i ) 
			tet._v[ i ] = orgPointIdx[ tet._v[i] ]; 

        storeTet( tetVec._arr, idx, tet ); 
    }
}

__global__ void 
kerMakeReverseMap
(
KerIntArray insVertVec, 
int*        scatterArr, 
int*        revMapArr,
int         num
)
{
    for ( int idx = getCurThreadIdx(); idx < insVertVec._num; idx += getThreadNum() )
    {
        const int oldIdx = scatterArr[ insVertVec._arr[ idx ] ]; 

        if ( oldIdx < num ) 
            revMapArr[ oldIdx ] = idx; 
    }
}

__global__ void 
kerMarkSpecialTets
(
KerCharArray tetInfoVec, 
TetOpp*      oppArr
)
{
    for ( int idx = getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
    {
        if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue; 

        TetOpp opp = loadOpp( oppArr, idx ); 

        bool changed = false; 

        for ( int vi = 0; vi < 4; ++vi ) 
        {
            //if ( -1 == opp._t[ vi ] ) continue; 

            if ( opp.isOppSpecial( vi ) ) 
            {
                changed = true; 

                opp.setOppSpecial( vi, false ); 
            }
            //else
            //    opp.setOppInternal( vi ); 
            // BUG: Non-Delaunay facets are set to be Internal!
        }

        if ( changed ) 
        {
            setTetCheckState( tetInfoVec._arr[ idx ], Changed ); 
            storeOpp( oppArr, idx, opp ); 
        }
    }
}

__global__ void
kerNegateInsertedVerts
(
KerIntArray vertTetVec, 
int*        tetToVert
)
{
    for ( int idx = getCurThreadIdx(); idx < vertTetVec._num; idx += getThreadNum() )
    {
        const int tetIdx = vertTetVec._arr[ idx ];

        if ( tetToVert[ tetIdx ] == idx ) 
            vertTetVec._arr[ idx ] = makeNegative( tetIdx ); 
    }
}

__global__ void 
kerAllocateFlip23Slot
(
KerIntArray flipToTet, 
Tet*        tetArr, 
int*        vertFreeArr, 
int*        freeArr, 
int*        flip23NewSlot,
int         infIdx,
int         tetNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipToTet._num; flipIdx += getThreadNum() )
    {
        const int voteVal    = flipToTet._arr[ flipIdx ];
        const int botTetIdx  = getVoteTetIdx( voteVal );
        const char flipInfo  = getVoteFlipInfo( voteVal );
        const FlipType fType = getFlipType( flipInfo );

        if ( fType != Flip23 ) continue; 

        // Bottom tetra
        Tet botTet = loadTet( tetArr, botTetIdx );

        // Try to put the new tets near one of the vertices 
        // of the botTet. Not perfect since the new tet is not this one, 
        // but at least 3 out of 4 vertices are the same. 
        // Ideally: Also look at the opp vertex. But probably too expensive!

        int freeIdx = -1; 

        for ( int vi = 0; vi < 4; ++vi ) 
        {
            int vert = botTet._v[ vi ]; 

            if ( vert >= infIdx ) continue;

            if ( vertFreeArr[ vert ] > 0 ) 
            {
                const int locIdx = atomicSub( &vertFreeArr[ vert ], 1 ) - 1;

                if ( locIdx >= 0 ) 
                {
                    freeIdx = freeArr[ vert * MeanVertDegree + locIdx ]; 
                    break; 
                }
                
                vertFreeArr[ vert ] = 0; 
                //atomicExch( &vertFreeArr[ vert ], 0 ); 
            }
        }

        if ( freeIdx == -1 )    // Still no free slot?
        {
            const int locIdx = atomicSub( &vertFreeArr[ infIdx ], 1 ) - 1; 

            if ( locIdx >= 0 ) 
                freeIdx = freeArr[ infIdx * MeanVertDegree + locIdx ]; 
            else
                // Gotta expand
                freeIdx = tetNum - locIdx - 1; 
        }

        flip23NewSlot[ flipIdx ] = freeIdx; 
    }
}

__global__ void 
kerUpdateBlockVertFreeList
(
KerIntArray insTetVec, 
int*        vertFreeArr, 
int*        freeArr, 
int*        scatterMap,
int         oldInsNum
)
{
    int freeNum = insTetVec._num * MeanVertDegree; 

    for ( int idx = getCurThreadIdx(); idx < freeNum; idx += getThreadNum() )
    {
        int insIdx  = idx / MeanVertDegree; 
        int locIdx  = idx % MeanVertDegree; 
        int vert    = insTetVec._arr[ insIdx ]; 
        int freeIdx = vert * MeanVertDegree + locIdx; 

        int newIdx; 

        if ( scatterMap[ vert ] >= oldInsNum )     // New vert
        {
            newIdx = idx; 

            // Update free size for new vert
            if ( locIdx == 0 )
                vertFreeArr[ vert ] = MeanVertDegree; 
        }
        else
            newIdx = idx - locIdx + freeArr[ freeIdx ] % MeanVertDegree; 

        freeArr[ freeIdx ] = newIdx; 
    }
}   

__global__ void 
kerShiftInfFreeIdx
(
int*    vertFreeArr, 
int*    freeArr, 
int     infIdx, 
int     start, 
int     shift
)
{
    int freeNum = vertFreeArr[ infIdx ]; 
    int freeBeg = infIdx * MeanVertDegree; 

    for ( int idx = getCurThreadIdx(); idx < freeNum; idx += getThreadNum() )
    {
        const int tetIdx = freeArr[ freeBeg + idx ]; 

        CudaAssert( tetIdx >= start ); 

        freeArr[ freeBeg + idx ] = tetIdx + shift; 
    }
}

__global__ void 
kerUpdateBlockOppTetIdx
(
TetOpp* oppArr, 
int*    orderArr, 
int     oldInfBlockIdx, 
int     newInfBlockIdx,
int     oldTetNum
) 
{
    for ( int idx = getCurThreadIdx(); idx < oldTetNum; idx += getThreadNum() )
    {
        TetOpp opp = loadOpp( oppArr, idx ); 

        for ( int i = 0; i < 4; ++i ) 
        {
            int tetIdx = opp.getOppTet( i ); 

            if ( tetIdx < 0 ) continue;

            if ( tetIdx < oldInfBlockIdx ) 
            {
                int insIdx = tetIdx / MeanVertDegree; 
                int locIdx = tetIdx % MeanVertDegree; 

                opp.setOppTet( i, orderArr[ insIdx ] * MeanVertDegree + locIdx ); 
            }
            else
                opp.setOppTet( i, tetIdx - oldInfBlockIdx + newInfBlockIdx ); 
        }

        storeOpp( oppArr, idx, opp ); 
    }
}

__global__ void 
kerUpdateTetIdx
(
KerIntArray idxVec, 
int*        orderArr, 
int         oldInfBlockIdx, 
int         newInfBlockIdx
) 
{
    for ( int idx = getCurThreadIdx(); idx < idxVec._num; idx += getThreadNum() )
    {
        int tetIdx = idxVec._arr[ idx ]; 

        int posTetIdx = ( tetIdx < 0 ? makePositive( tetIdx ) : tetIdx ); 

        if ( posTetIdx < oldInfBlockIdx ) 
        {
            int insIdx = posTetIdx / MeanVertDegree; 
            int locIdx = posTetIdx % MeanVertDegree; 

            posTetIdx = orderArr[ insIdx ] * MeanVertDegree + locIdx; 
        } else
            posTetIdx = posTetIdx - oldInfBlockIdx + newInfBlockIdx; 

        idxVec._arr[ idx ] = ( tetIdx < 0 ? makeNegative( posTetIdx ) : posTetIdx ); 
    }
}

__global__ void 
kerShiftOppTetIdx
(
TetOpp* oppArr, 
int     tetNum,
int     start,
int     shift
) 
{
    for ( int idx = getCurThreadIdx(); idx < tetNum; idx += getThreadNum() )
    {
        TetOpp opp = loadOpp( oppArr, idx ); 

        for ( int i = 0; i < 4; ++i ) 
        {
            if ( opp._t[ i ] < 0 ) continue; 

            const int oppIdx = opp.getOppTet( i ); 

            if ( oppIdx >= start ) 
                opp.setOppTet( i, oppIdx + shift ); 
        }

        storeOpp( oppArr, idx, opp ); 
    }
}

__global__ void 
kerShiftTetIdx
(
KerIntArray idxVec, 
int         start,
int         shift
) 
{
    int negStart = makeNegative( start ); 

    for ( int idx = getCurThreadIdx(); idx < idxVec._num; idx += getThreadNum() )
    {
        const int oldIdx = idxVec._arr[ idx ]; 
         
        if ( oldIdx >= start ) 
            idxVec._arr[ idx ] = oldIdx + shift; 

        if ( oldIdx <= negStart ) 
            idxVec._arr[ idx ] = oldIdx - shift; 
    }
}

__global__ void 
kerUpdateVertFreeList
(
KerIntArray insTetVec, 
int*        vertFreeArr, 
int*        freeArr, 
int         startFreeIdx
)
{
    int newFreeNum   = insTetVec._num * MeanVertDegree; 

    for ( int idx = getCurThreadIdx(); idx < newFreeNum; idx += getThreadNum() )
    {
        int insIdx  = idx / MeanVertDegree; 
        int locIdx  = idx % MeanVertDegree; 
        int vertIdx = insTetVec._arr[ insIdx ]; 

        freeArr[ vertIdx * MeanVertDegree + locIdx ] = startFreeIdx + idx; 

        // Update free size for new vert
        if ( idx < insTetVec._num ) 
            vertFreeArr[ insTetVec._arr[ idx ] ] = MeanVertDegree; 
    }
}   

__global__ void 
kerCollectFreeSlots
(
char* tetInfoArr, 
int*  prefixArr,
int*  freeArr,
int   newTetNum
)
{
    for ( int idx = getCurThreadIdx(); idx < newTetNum; idx += getThreadNum() )
    {
        if ( isTetAlive( tetInfoArr[ idx ] ) ) continue; 

        int freeIdx = idx - prefixArr[ idx ]; 

        freeArr[ freeIdx ] = idx; 
    }
}

__global__ void
kerMakeCompactMap
(
KerCharArray tetInfoVec, 
int*         prefixArr, 
int*         freeArr, 
int          newTetNum
)
{
    for ( int idx = newTetNum + getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
    {
        if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue; 

        int freeIdx     = newTetNum - prefixArr[ idx ]; 
        int newTetIdx   = freeArr[ freeIdx ]; 

        prefixArr[ idx ] = newTetIdx; 
    }
}

__global__ void
kerCompactTets
(
KerCharArray tetInfoVec, 
int*         prefixArr, 
Tet*         tetArr, 
TetOpp*      oppArr, 
int          newTetNum
)
{
    for ( int idx = newTetNum + getCurThreadIdx(); idx < tetInfoVec._num; idx += getThreadNum() )
    {
        if ( !isTetAlive( tetInfoVec._arr[ idx ] ) ) continue; 

        int newTetIdx   = prefixArr[ idx ]; 

        Tet tet = loadTet( tetArr, idx ); 
        storeTet( tetArr, newTetIdx, tet ); 

        TetOpp opp = loadOpp( oppArr, idx ); 

        for ( int vi = 0; vi < 4; ++vi ) 
        {
            if ( opp._t[ vi ] < 0 ) continue; 

            const int oppIdx = opp.getOppTet( vi ); 

            if ( oppIdx >= newTetNum ) 
            {
                const int oppNewIdx = prefixArr[ oppIdx ]; 

                opp.setOppTet( vi, oppNewIdx ); 
            }
            else
            {
                const int oppVi = opp.getOppVi( vi ); 

                oppArr[ oppIdx ].setOppTet( oppVi, newTetIdx ); 
            }
        }

        storeOpp( oppArr, newTetIdx, opp ); 
    }
}
