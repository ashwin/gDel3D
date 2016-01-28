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
#include "KerPredicates.h"
#include "DPredWrapper.h"

#ifndef __CUDACC__
#define __launch_bounds__( x )
#endif


__constant__ DPredWrapper dPredWrapper; 

#include "KerPredWrapper.h"

void setPredWrapperConstant( const DPredWrapper &hostPredWrapper ) 
{
    CudaSafeCall( cudaMemcpyToSymbol( dPredWrapper, &hostPredWrapper, sizeof( hostPredWrapper ) ) ); 
}

template<bool doFast> 
__forceinline__ __device__ void initPointLocation
(
int*    vertTetArr, 
Tet     tet,
int     tetIdx
)
{
    const int tetVert[5]  = { tet._v[0], tet._v[1], tet._v[2], tet._v[3], dPredWrapper._infIdx };
    const Point3 pt[] = { 
        dPredWrapper.getPoint( tetVert[0] ), 
        dPredWrapper.getPoint( tetVert[1] ), 
        dPredWrapper.getPoint( tetVert[2] ), 
        dPredWrapper.getPoint( tetVert[3] ), 
        dPredWrapper.getPoint( tetVert[4] )
    }; 

    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < dPredWrapper.pointNum(); idx += getThreadNum() )
    {
        if ( !doFast && vertTetArr[ idx ] != -2 )   // No exact check needed
            continue; 

        if ( tet.has( idx ) || idx == dPredWrapper._infIdx )   // Already inserted
        {
            vertTetArr[ idx ] = -1; 
            continue; 
        }

        Point3 ptVertex = dPredWrapper.getPoint( idx );  

        int face = 0; 

        for ( int i = 0; i < 4; ++i ) 
        {
            const int *fv = SplitFaces[ face ]; 

            Orient ort = ( doFast ) 
                ? dPredWrapper.doOrient3DFast(
                    tetVert[ fv[0] ], tetVert[ fv[1] ], tetVert[ fv[2] ], idx, 
                         pt[ fv[0] ],      pt[ fv[1] ],      pt[ fv[2] ], ptVertex )
                : dPredWrapper.doOrient3DSoS(
                    tetVert[ fv[0] ], tetVert[ fv[1] ], tetVert[ fv[2] ], idx, 
                         pt[ fv[0] ],      pt[ fv[1] ],      pt[ fv[2] ], ptVertex ); 

            if ( doFast && (ort == OrientZero) ) { face = -tetIdx - 2;  break;  }    // Needs exact computation

            // Use the reverse direction 'cause the splitting point is Infty!
            face = SplitNext[ face ][ ( ort == OrientPos ) ? 1 : 0 ]; 

            // Compiler bug: Without this assertion, this code produces undefined result in Debug-x86.
            CudaAssert( face >= 0 ); 
        }

        vertTetArr[ idx ] = tetIdx + face; 
    }
}

__global__ void kerInitPointLocationFast
(
int*    vertTetArr, 
Tet     tet,
int     tetIdx
)
{
    initPointLocation<true>( vertTetArr, tet, tetIdx ); 
}

__global__ void kerInitPointLocationExact
(
int*    vertTetArr, 
Tet     tet,
int     tetIdx
)
{
    initPointLocation<false>( vertTetArr, tet, tetIdx ); 
}

__forceinline__ __device__ float hash( int k ) 
{
    k *= 357913941;
    k ^= k << 24;
    k += ~357913941;
    k ^= k >> 31;
    k ^= k << 31;

    return int_as_float( k ); 
}

__global__ void
kerVoteForPoint
(
KerIntArray     vertexArr,
int*            vertexTetArr,
Tet*            tetArr,
int*            vertSphereArr,
int*            tetSphereArr,
InsertionRule   insRule
)
{
    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < vertexArr._num; idx += getThreadNum() )
    {
        //*** Compute insphere value

        const int tetIdx   = vertexTetArr[ idx ];
        const Tet tet      = tetArr[ tetIdx ];
        const int vert     = vertexArr._arr[ idx ];
        float sval; 
        
        switch ( insRule ) 
        {
        case InsCircumcenter: 
            sval = dPredWrapper.inSphereDet( tet, vert ); 
            break; 
        case InsCentroid:
            sval = dPredWrapper.distToCentroid( tet, vert ); 
            break; 
        case InsRandom:
            sval = hash(vert); 
            break; 
        } 

        //*** Sanitize and store sphere value

        if ( sval < 0 )
            sval = 0;

        int ival = __float_as_int(sval); 

        vertSphereArr[ idx ] =  ival;

        //*** Vote

        if ( tetSphereArr[ tetIdx ] < ival ) // Helps reduce atomicMax cost!
            atomicMax( &tetSphereArr[ tetIdx ], ival );
    }

    return;
}

template < bool doFast >
__forceinline__ __device__ void
splitPoints
(
KerIntArray vertexArr,
int*        vertexTetArr,
int*        tetToVert,
Tet*        tetArr,
char*       tetInfoArr,
KerIntArray freeArr
)
{
    // Iterate uninserted points
    for ( int vertIdx = getCurThreadIdx(); vertIdx < vertexArr._num; vertIdx += getThreadNum() )
    {
        int tetIdx = vertexTetArr[ vertIdx ];

        if ( doFast && tetIdx < 0 ) continue; // This vertex is inserted.

        if ( !doFast && tetIdx >= 0 ) continue; // Exact mode, vertex already processed in fast mode

        if ( !doFast ) 
            tetIdx = makePositive( tetIdx ); // Exact mode, vertex needs processing

        const int splitVertIdx = tetToVert[ tetIdx ];

        if ( !doFast && splitVertIdx == vertIdx ) continue;     // This vertex is the inserting one

        if ( splitVertIdx == INT_MAX )  // Tet not split, nothing to update
        {
            setTetEmptyState( tetInfoArr[ tetIdx ], false );    // 'cause this may be due to insertion control
            continue; // Vertex's tetra will not be split in this round
        }

        const int vertex      = vertexArr._arr[ vertIdx ];

        const Point3 ptVertex = dPredWrapper.getPoint( vertex ); 
        const int splitVertex = vertexArr._arr[ splitVertIdx ];
        const Tet tet         = loadTet( tetArr, tetIdx );

        const int freeIdx     = ( splitVertex + 1 ) * MeanVertDegree - 1; 
        const int tetVert[5]  = { tet._v[0], tet._v[1], tet._v[2], tet._v[3], splitVertex };
        const Point3 pt[] = { 
            dPredWrapper.getPoint( tetVert[0] ), 
            dPredWrapper.getPoint( tetVert[1] ), 
            dPredWrapper.getPoint( tetVert[2] ), 
            dPredWrapper.getPoint( tetVert[3] ), 
            dPredWrapper.getPoint( tetVert[4] )
        }; 

        int face = 0; 

        for ( int i = 0; i < 3; ++i ) 
        {
            const int *fv = SplitFaces[ face ]; 

            Orient ort = ( doFast ) 
                ? dPredWrapper.doOrient3DFast(
                    tetVert[ fv[0] ], tetVert[ fv[1] ], tetVert[ fv[2] ], vertex, 
                         pt[ fv[0] ],      pt[ fv[1] ],      pt[ fv[2] ], ptVertex )
                : dPredWrapper.doOrient3DSoS(
                    tetVert[ fv[0] ], tetVert[ fv[1] ], tetVert[ fv[2] ], vertex, 
                         pt[ fv[0] ],      pt[ fv[1] ],      pt[ fv[2] ], ptVertex ); 

            // Needs exact computation
            if ( doFast && (ort == OrientZero) ) { face = makeNegative( tetIdx );  break;  }    

            face = SplitNext[ face ][ ( ort == OrientPos ) ? 0 : 1 ]; 
        }

        if ( face >= 0 ) 
        {
            face = freeArr._arr[ freeIdx - (face - 7) ];

            setTetEmptyState( tetInfoArr[ face ], false ); 
        }

        vertexTetArr[ vertIdx ] = face; 
    }

    return;
}

__global__ void
kerSplitPointsFast
(
KerIntArray vertexArr,
int*        vertexTetArr,
int*        tetToVert,
Tet*        tetArr,
char*       tetInfoArr,
KerIntArray freeArr)
{
    splitPoints< true >(
        vertexArr,
        vertexTetArr,
        tetToVert,
        tetArr,
        tetInfoArr,
        freeArr
        );
}

__global__ void
kerSplitPointsExactSoS
(
KerIntArray vertexArr,
int*        vertexTetArr,
int*        tetToVert,
Tet*        tetArr,
char*       tetInfoArr,
KerIntArray freeArr
)
{
    splitPoints< false >(
        vertexArr,
        vertexTetArr,
        tetToVert,
        tetArr,
        tetInfoArr,
        freeArr
        );
}

__forceinline__ __device__ void
voteForFlip32
(
int* tetVoteArr,
int  voteOffset,
int  botTi,
int  topTi,
int  sideTi
)
{
    const int voteVal = voteOffset + botTi;

    atomicMin( &tetVoteArr[ botTi ],  voteVal );
    atomicMin( &tetVoteArr[ topTi ],  voteVal );
    atomicMin( &tetVoteArr[ sideTi ], voteVal );
}

__forceinline__ __device__ void
voteForFlip23
(
int* tetVoteArr,
int  voteOffset,
int  botTi,
int  topTi
)
{
    const int voteVal = voteOffset + botTi;

    atomicMin( &tetVoteArr[ botTi ], voteVal );
    atomicMin( &tetVoteArr[ topTi ], voteVal );
}

extern __shared__ int2 s_exactCheck[]; 

template< typename T >
__forceinline__ __device__ void writeShared
(
T* s_input, 
int& s_offset, 
int& s_num, 
T* output,
int& g_counter
)
{
    int writeNum = ( s_num >= BLOCK_DIM ) ? BLOCK_DIM : s_num; 

    if ( THREAD_IDX == 0 ) 
        s_offset = atomicAdd( &g_counter, writeNum ); 

    __syncthreads(); 

    if ( THREAD_IDX < writeNum ) 
        output[ s_offset + THREAD_IDX ] = s_input[ THREAD_IDX ]; 

    if ( THREAD_IDX < s_num - BLOCK_DIM )
        s_input[ THREAD_IDX ] = s_input[ BLOCK_DIM + THREAD_IDX ]; 

    __syncthreads(); 

    if ( THREAD_IDX == 0 ) 
        s_num -= writeNum; 

    __syncthreads(); 
}

template < CheckDelaunayMode checkMode >
__forceinline__ __device__ void
checkDelaunayFast
(
KerIntArray actTetVec,
Tet*    tetArr,
TetOpp* oppArr,
char*   tetInfoArr,
int*    tetVoteArr,
int*    voteArr,
int2*   exactCheckVi, 
int*    counterArr,
int     voteOffset
)
{
    __shared__ int s_num, s_offset; 

    int actTetNumRounded    = actTetVec._num; 

    if ( SphereExactOrientSoS == checkMode )
    {
        if ( THREAD_IDX == 0 ) 
            s_num = 0; 

        actTetNumRounded = roundUp( actTetVec._num, BLOCK_DIM );  

        __syncthreads(); 
    }

    // Iterate active tetra
    for ( int idx = getCurThreadIdx(); idx < actTetNumRounded; idx += getThreadNum() )
    {
        if ( SphereExactOrientSoS != checkMode || idx < actTetVec._num ) 
        {
            voteArr[ idx ]  = -1; 
            const int botTi = actTetVec._arr[ idx ];

            if ( !isTetAlive( tetInfoArr[ botTi ] ) )
                actTetVec._arr[ idx ] = -1;
            else
            {
                ////
                // Quickly load four neighbors' opp verts and status
                ////
                TetOpp  botOpp      = loadOpp( oppArr, botTi );
                int     oppVert[4]; 

                for ( int botVi = 0; botVi < 4; ++botVi ) 
                {
                    int topVert = -1; 

                    // No neighbour at this face or face is internal (i.e. already locally Delaunay)
                    if ( /*-1 != botOpp._t[ botVi ] &&*/ !botOpp.isOppInternal( botVi ) )
                    {
                        const int topTi = botOpp.getOppTet( botVi );
                        const int topVi = botOpp.getOppVi( botVi );               
                        topVert         = tetArr[ topTi ]._v[ topVi ];

                        if ( ( ( topTi < botTi ) && Changed == getTetCheckState( tetInfoArr[ topTi ] ) ) )
                            topVert = makeNegative( topVert ); 
                    }

                    oppVert[ botVi ] = topVert; 
                }

                ////
                // Check flipping configuration
                ////
                int checkVi = 1; 
                //int skip    = 0; 

                for ( int botVi = 0; botVi < 4; ++botVi )
                {
                    // TODO: Figure why this skipping thing doesn't work. 
                    // Some facets are left unchecked and unmarked with sphere failure.
                    // Hint: From 3-2 flippable flip becomes 2-2 unflippable.
                    //if ( isBitSet( skip, botVi ) ) continue; 

                    const int topVert = oppVert[ botVi ]; 

                    if ( topVert < 0 ) continue; 

                    //*** Check for 3-2 flip
                    const int* botOrdVi = TetViAsSeenFrom[ botVi ]; // Order bottom tetra as seen from apex vertex
                    int i = 0; 

                    for ( ; i < 3; ++i ) // Check 3 sides of bottom-top tetra
                    {
                        const int sideVert = oppVert[ botOrdVi[ i ] ]; 

                        // More than 3 tetra around edge
                        if ( sideVert != topVert && sideVert != makeNegative( topVert ) ) continue; 

                        // 3-2 flip is possible.
                        //setBitState( skip, botOrdVi[ i ], true ); 

                        break; 
                    }

                    checkVi = (checkVi << 4) | botVi | ( i << 2 ); 
                }
                
                if ( checkVi != 1 )     // Anything to check?
                {
                    ////
                    // Do sphere check
                    ////
                    const Tet botTet     = loadTet(tetArr, botTi );
                    const Point3 botP[4] = {
                        dPredWrapper.getPoint( botTet._v[0] ),
                        dPredWrapper.getPoint( botTet._v[1] ),
                        dPredWrapper.getPoint( botTet._v[2] ),
                        dPredWrapper.getPoint( botTet._v[3] )
                    };  // Cache in local mem

                    int check23  = 1; 
                    int exactVi  = 1; 
                    bool hasFlip = false; 

                    // Check 2-3 flips
                    for ( ; checkVi > 1; checkVi >>= 4 )
                    {            
                        const int botVi     = ( checkVi & 3 ); 
                        int botCorOrdVi     = ( checkVi >> 2 ) & 3;
                        const int topVert   = oppVert[ botVi ]; 
                        const Point3 topP   = dPredWrapper.getPoint( topVert );

                        const Side side = dPredWrapper.doInSphereFast( botTet, topVert, botP, topP );

                        if ( SideZero == side )
                            if ( checkMode == SphereFastOrientFast ) // Store for future exact mode
                                botOpp.setOppSpecial( botVi, true );      
                            else // Pass to next kernel - exact kernel
                                exactVi = (exactVi << 5) | ( botVi << 1 ) | ( botCorOrdVi << 3 ) | 0;            
            
                        if ( SideIn != side ) continue; // No insphere failure at this face

                        // We have insphere failure
                        botOpp.setOppSphereFail( botVi );

                        if ( botCorOrdVi < 3 )     // 3-2 flipping is possible 
                        {
                            //*** 3-2 flip confirmed
                            char flipInfo  = makeFlip( botVi, botCorOrdVi );
                            voteArr[ idx ] = makeVoteVal( botTi, flipInfo ); 

                            const int botCorVi  = TetViAsSeenFrom[ botVi ][ botCorOrdVi ];
                            const int botOppTi  = botOpp.getOppTet( botCorVi ); // Side tetra as seen from bottom and top tetra                
                            const int topTi     = botOpp.getOppTet( botVi );
                            voteForFlip32( tetVoteArr, voteOffset, botTi, topTi, botOppTi );

                            hasFlip = true; 
                            check23 = 1;    // No more need to check 2-3
                            break; 
                        }

                        // Postpone check for 2-3 flippability
                        check23 = ( check23 << 2 ) | botVi; 
                    }

                    //*** Try for 2-3 flip
                    for ( ; check23 > 1; check23 >>= 2 )
                    {            
                        const int botVi     = ( check23 & 3 ); 
                        const int topVert   = oppVert[ botVi ]; 
                        const Point3 topP   = dPredWrapper.getPoint( topVert );
                        const int* botOrdVi = TetViAsSeenFrom[ botVi ]; // Order bottom tetra as seen from apex vertex
            
                        hasFlip = true; 

                        // Go around bottom-top tetra, check 3 sides
                        for ( int i = 0; i < 3; ++i )
                        {
                            const int* fv = TetViAsSeenFrom[ botOrdVi[i] ];

                            Orient ort = dPredWrapper.doOrient3DFast( 
                                botTet._v[ fv[0] ], botTet._v[ fv[1] ], botTet._v[ fv[2] ], topVert,
                                botP[ fv[0] ], botP[ fv[1] ], botP[ fv[2] ], topP ); 

                            if ( OrientZero == ort ) 
                                if ( checkMode == SphereFastOrientFast ) 
                                    // Store for future exact mode
                                    botOpp.setOppSpecial( botVi, true );
                                else
                                    // Pass to next kernel - exact kernel
                                    exactVi = (exactVi << 5) | ( botVi << 1 ) | ( 3 << 3 ) | 1;

                            if ( OrientPos != ort )
                            {
                                hasFlip = false; 
                                break; // Cannot do 23 flip
                            }
                        }

                        if ( hasFlip ) //*** 2-3 flip possible!
                        {                            
                            const char flipInfo = makeFlip( botVi, 3 );
                            voteArr[ idx ]      = makeVoteVal( botTi, flipInfo ); 
                            const int topTi     = botOpp.getOppTet( botVi );
                            voteForFlip23( tetVoteArr, voteOffset, botTi, topTi );

                            break; 
                        }
                    } // Check faces of tetra

                    storeOpp( oppArr, botTi, botOpp ); 

                    if ( ( checkMode == SphereExactOrientSoS ) && ( !hasFlip ) && ( exactVi != 1 ) )
                    {
#if __CUDA_ARCH__ >= 120
                        const int checkIdx       = atomicAdd( &s_num, 1 ); 
                        s_exactCheck[ checkIdx ] = make_int2( idx, exactVi ); 
#else
						const int checkIdx       = atomicAdd( &counterArr[ CounterExact ], 1 ); 
						exactCheckVi[ checkIdx ] = make_int2( idx, exactVi ); 
#endif
                    }
                }
            }
        }

#if __CUDA_ARCH__ >= 120
        if ( SphereExactOrientSoS == checkMode ) 
        {
            __syncthreads(); 

            // Output to global mem
            if ( s_num >= BLOCK_DIM )  
                writeShared( s_exactCheck, s_offset, s_num, 
                    exactCheckVi, counterArr[ CounterExact ] ); 
        }
#endif
    }

#if __CUDA_ARCH__ >= 120
    if ( SphereExactOrientSoS == checkMode && s_num > 0 )  // Output to global mem
        writeShared( s_exactCheck, s_offset, s_num, 
            exactCheckVi, counterArr[ CounterExact ] ); 
#endif

    if ( blockIdx.x == 0 && threadIdx.x == 0 ) 
    {
        counterArr[ CounterFlip ]   = 0; 
    }

    return;
}

__global__ void
kerCheckDelaunayFast
(
KerIntArray     actTetVec,
Tet*            tetArr,
TetOpp*         oppArr,
char*           tetInfoArr,
int*            tetVoteArr,
int*            voteArr,
int*            counterArr,
int             voteOffset
)
{
    checkDelaunayFast< SphereFastOrientFast >(
        actTetVec,
        tetArr,
        oppArr,
        tetInfoArr,
        tetVoteArr,
        voteArr,
        NULL,
        counterArr,
        voteOffset
        );
    return;
}

__global__ void
kerCheckDelaunayExact_Fast
(
KerIntArray     actTetVec,
Tet*            tetArr,
TetOpp*         oppArr,
char*           tetInfoArr,
int*            tetVoteArr,
int*            voteArr,
int2*           exactCheckVi, 
int*            counterArr,
int             voteOffset
)
{
    checkDelaunayFast< SphereExactOrientSoS >(
        actTetVec,
        tetArr,
        oppArr,
        tetInfoArr,
        tetVoteArr,
        voteArr,
        exactCheckVi, 
        counterArr,
        voteOffset
        );
    return;
}

__global__ void
__launch_bounds__( PRED_THREADS_PER_BLOCK )
kerCheckDelaunayExact_Exact
(
int*            actTetArr,
Tet*            tetArr,
TetOpp*         oppArr,
char*           tetInfoArr,
int*            tetVoteArr,
int*            voteArr,
int2*           exactCheckVi,
int*            counterArr,
int             voteOffset
)
{
    const int exactNum = counterArr[ CounterExact ]; 

    // Iterate active tetra
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        int2 val    = exactCheckVi[ idx ]; 
        int botTi   = actTetArr[ val.x ]; 
        int exactVi = val.y; 

        exactCheckVi[ idx ] = make_int2( -1, -1 ); 

        ////
        // Do sphere check
        ////
        TetOpp botOpp        = loadOpp( oppArr, botTi );
        const Tet botTet     = loadTet( tetArr, botTi );
        const Point3 botP[4] = {
            dPredWrapper.getPoint( botTet._v[0] ),
            dPredWrapper.getPoint( botTet._v[1] ),
            dPredWrapper.getPoint( botTet._v[2] ),
            dPredWrapper.getPoint( botTet._v[3] )
        };

        // Check 2-3 flips
        for ( ; exactVi > 1; exactVi >>= 5 )
        {            
            const int botVi = ( exactVi >> 1 ) & 3; 
            int botCorOrdVi = ( exactVi >> 3 ) & 3;

            const int topTi     = botOpp.getOppTet( botVi );
            const int topVi     = botOpp.getOppVi( botVi );               
            const int topVert   = tetArr[ topTi ]._v[ topVi ];
            const Point3 topP   = dPredWrapper.getPoint( topVert );

            if ( ( exactVi & 1 ) == 0 ) 
            {
                const Side side = dPredWrapper.doInSphereSoS( botTet, topVert, botP, topP );

                if ( SideIn != side ) continue; // No insphere failure at this face
            }

            botOpp.setOppSphereFail( botVi );

            // We have insphere failure, determine kind of flip
            const FlipType flipType = ( 3 == botCorOrdVi ? Flip23 : Flip32 ); 

            //*** Try for 3-2 flip

            const int* botOrdVi = TetViAsSeenFrom[ botVi ]; // Order bottom tetra as seen from apex vertex

            if ( Flip32 == flipType )     // 3-2 flipping is possible 
            {
                //*** 3-2 flip confirmed
                const int botCorVi  = botOrdVi[ botCorOrdVi ];
                const int botOppTi  = botOpp.getOppTet( botCorVi ); // Side tetra as seen from bottom and top tetra                
                voteForFlip32( tetVoteArr, voteOffset, botTi, topTi, botOppTi );

                char flipInfo       = makeFlip( botVi, botCorOrdVi );
                voteArr[ val.x ]  = makeVoteVal( botTi, flipInfo ); 
                break; 
            }

            // Try flip 2-3
            bool hasFlip = true; 

            // Go around bottom-top tetra, check 3 sides
            for ( int i = 0; i < 3; ++i )
            {
                const int botCorVi    = botOrdVi[i];
                const int* fv = TetViAsSeenFrom[ botCorVi ];

                const Orient ort = dPredWrapper.doOrient3DSoS( 
                    botTet._v[ fv[0] ], botTet._v[ fv[1] ], botTet._v[ fv[2] ], topVert,
                    botP[ fv[0] ], botP[ fv[1] ], botP[ fv[2] ], topP );

                if ( OrientPos != ort )
                {
                    hasFlip = false; 
                    break; // Cannot do 23 flip
                }
            }

            if ( hasFlip )
            {
                voteForFlip23( tetVoteArr, voteOffset, botTi, topTi );

                const char flipInfo = makeFlip( botVi, 3 );
                voteArr[ val.x ]  = makeVoteVal( botTi, flipInfo ); 
                break;
            }
        } // Check faces of tetra

        storeOpp( oppArr, botTi, botOpp ); 
    }

    return;
}

__device__ int setNeedExact( int val ) 
{
    return val | ( 1 << 31 ); 
}

__device__ int removeExactBit( int val ) 
{
    return ( val & ~(1 << 31) ); 
}

__device__ bool isNeedExact( int val ) 
{
    return ( val >> 31 ) & 1; 
}

template<bool doFast>
__forceinline__ __device__ void
relocatePoints
(
KerIntArray vertexArr,
int*        vertexTetArr,
int*        tetToFlip,
FlipItem*   flipArr
)
{
    // Iterate uninserted points
    for ( int vertIdx = getCurThreadIdx(); vertIdx < vertexArr._num; vertIdx += getThreadNum() )
    {
        const int tetIdxVal = vertexTetArr[ vertIdx ];

        if ( !doFast && !isNeedExact( tetIdxVal ) ) continue;

        const int tetIdx    = removeExactBit( tetIdxVal );
        int nextIdx         = ( doFast ) ? tetToFlip[ tetIdx ] : tetIdx; 

        if ( nextIdx == -1 ) 
            continue; 

        const int vertex    = vertexArr._arr[ vertIdx ];
        int flag            = nextIdx & 1; 
        int destIdx         = nextIdx >> 1; 

        while ( flag == 1 ) 
        {
            const FlipItem flipItem = loadFlip( flipArr, destIdx ); 
            const FlipType fType    = ( flipItem._t[ 2 ] < 0 ? Flip32 : Flip23 ); 

            int nextLocId; 
            int3 F; 
        
            if ( Flip23 == fType )
                F = make_int3( 0, 2, 3 ); 
            else
                F = make_int3( 0, 1, 2 ); 

            const Orient ord0 = doFast 
                    ? dPredWrapper.doOrient3DFast( flipItem._v[ F.x ], flipItem._v[ F.y ], flipItem._v[ F.z ], vertex )
                    : dPredWrapper.doOrient3DSoS( flipItem._v[ F.x ], flipItem._v[ F.y ], flipItem._v[ F.z ], vertex ); 

            if ( doFast && ( OrientZero == ord0 ) )
            {
                destIdx = setNeedExact( nextIdx ); 
                break;  
            }

            if ( Flip32 == fType )
            {
                nextLocId = ( OrientPos == ord0 ) ? 0 : 1;
            }
            else
            {
                if ( OrientPos == ord0 ) 
                {
                    nextLocId   = 0; 
                    F           = make_int3( 0, 3, 1 ); 
                }
                else
                {
                    nextLocId   = 1;
                    F           = make_int3( 0, 4, 3 ); 
                }

                //right = 2;            

                const Orient ord1 = doFast 
                        ? dPredWrapper.doOrient3DFast( flipItem._v[ F.x ], flipItem._v[ F.y ], flipItem._v[ F.z ], vertex )
                        : dPredWrapper.doOrient3DSoS( flipItem._v[ F.x ], flipItem._v[ F.y ], flipItem._v[ F.z ], vertex ); 

                if ( doFast && ( OrientZero == ord1 ) )
                {
                    destIdx = setNeedExact( nextIdx ); 
                    break; 
                }
                else
                    nextLocId = ( OrientPos == ord1 ) ? nextLocId : 2;
            }

            nextIdx = flipItem._t[ nextLocId ]; 
            flag    = nextIdx & 1; 
            destIdx = nextIdx >> 1; 
        }

        vertexTetArr[ vertIdx ] = destIdx; // Write back
    }

    return;
}

__global__ void
kerRelocatePointsFast
(
KerIntArray     vertexArr,
int*            vertexTetArr,
int*            tetToFlip,
FlipItem*       flipArr
)
{
    relocatePoints<true>( 
        vertexArr, 
        vertexTetArr, 
        tetToFlip, 
        flipArr
    ); 
}

__global__ void
kerRelocatePointsExact
(
KerIntArray     vertexArr,
int*            vertexTetArr,
int*            tetToFlip,
FlipItem*       flipArr
)
{
    relocatePoints<false>( 
        vertexArr, 
        vertexTetArr, 
        tetToFlip, 
        flipArr
        ); 
}
