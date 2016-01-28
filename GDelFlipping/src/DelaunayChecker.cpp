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

#include "DelaunayChecker.h"

#include "gDel3D/CPU/CPUDecl.h"

DelaunayChecker::DelaunayChecker
( 
Point3HVec* pointVec, 
GDelOutput* output
)
: _pointVec( pointVec ), _output( output ) 
{
    _predWrapper.init( *pointVec, output->ptInfty ); 
} 

void getTetraSegments( const Tet& t, Segment* sArr )
{
    for ( int i = 0; i < TetSegNum; ++i )
    {
        Segment seg = { t._v[ TetSeg[i][0] ], t._v[ TetSeg[i][1] ] };

        if ( seg._v[0] > seg._v[1] ) std::swap( seg._v[0], seg._v[1] );

        sArr[i] = seg;
    }

    return;
}

void getTetraTriangles( const Tet& t, Tri* triArr )
{
    for ( int i = 0; i < TetFaceNum; ++i )
    {
        // Tri vertices
        Tri tri = { t._v[ TetFace[i][0] ], t._v[ TetFace[i][1] ], t._v[ TetFace[i][2] ] };

        // Sort
        if ( tri._v[0] > tri._v[1] ) std::swap( tri._v[0], tri._v[1] );
        if ( tri._v[1] > tri._v[2] ) std::swap( tri._v[1], tri._v[2] );
        if ( tri._v[0] > tri._v[1] ) std::swap( tri._v[0], tri._v[1] );

        // Add triangle
        triArr[ i ] = tri;
    }

    return;
}

int DelaunayChecker::getVertexCount()
{
    const TetHVec& tetVec       = _output->tetVec;             
    const CharHVec& tetInfoVec  = _output->tetInfoVec; 

    std::set< int > vertSet; 

    // Add vertices
    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        vertSet.insert( tet._v, tet._v + 4 );
    }

    return vertSet.size();
}

int DelaunayChecker::getSegmentCount()
{
    const TetHVec& tetVec       = _output->tetVec;             
    const CharHVec& tetInfoVec  = _output->tetInfoVec; 

    std::set< Segment > segSet; 

    // Read segments
    Segment segArr[ TetSegNum ];

    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        getTetraSegments( tet, segArr );

        segSet.insert( segArr, segArr + TetSegNum ); 
    }

    return segSet.size();
}

int DelaunayChecker::getTriangleCount()
{
    const TetHVec& tetVec       = _output->tetVec;             
    const CharHVec& tetInfoVec  = _output->tetInfoVec; 

    std::set< Tri > triSet; 

    // Read triangles
    Tri triArr[ TetFaceNum ];

    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        getTetraTriangles( tet, triArr );

        triSet.insert( triArr, triArr + TetFaceNum ); 
    }

    return triSet.size();
}

int DelaunayChecker::getTetraCount()
{
    const CharHVec& tetInfoVec  = _output->tetInfoVec; 

    int count = 0;

    for ( int ti = 0; ti < ( int ) tetInfoVec.size(); ++ti )
        if ( isTetAlive( tetInfoVec[ti] ) )
            ++count;

    return count;
}


void DelaunayChecker::checkEuler()
{
    const int v = getVertexCount();
    std::cout << "V: " << v;

    const int e = getSegmentCount();
    std::cout << " E: " << e;

    const int f = getTriangleCount();
    std::cout << " F: " << f;

    const int t = getTetraCount();
    std::cout << " T: " << t;

    const int euler = v - e + f - t;
    std::cout << " Euler: " << euler << std::endl;

    std::cout << "Euler check: " << ( ( 0 != euler ) ? " ***Fail***" : " Pass" ) << std::endl;

    return;
}

void DelaunayChecker::printTetraAndOpp( int ti, const Tet& tet, const TetOpp& opp )
{
    printf( "TetIdx: %d [ %d %d %d %d ] ( %d:%d %d:%d %d:%d %d:%d )\n",
        ti,
        tet._v[0], tet._v[1], tet._v[2], tet._v[3],
        opp.getOppTet(0), opp.getOppVi(0),
        opp.getOppTet(1), opp.getOppVi(1),
        opp.getOppTet(2), opp.getOppVi(2),
        opp.getOppTet(3), opp.getOppVi(3) );
}

void DelaunayChecker::checkAdjacency()
{
    const TetHVec tetVec        = _output->tetVec; 
    const TetOppHVec oppVec     = _output->tetOppVec; 
    const CharHVec tetInfoVec   = _output->tetInfoVec; 

    for ( int ti0 = 0; ti0 < ( int ) tetVec.size(); ++ti0 )
    {
        if ( !isTetAlive( tetInfoVec[ti0] ) ) continue;

        const Tet& tet0    = tetVec[ ti0 ];
        const TetOpp& opp0 = oppVec[ ti0 ];

        for ( int vi = 0; vi < 4; ++vi )
        {
            if ( -1 == opp0._t[ vi ] ) continue;

            const int ti1   = opp0.getOppTet( vi );
            const int vi0_1 = opp0.getOppVi( vi );

            if ( !isTetAlive( tetInfoVec[ ti1 ] ) )
            {
                std::cout << "TetIdx: " << ti1 << " is invalid!" << std::endl;
                exit(-1);
            }

            const Tet& tet1    = tetVec[ ti1 ];
            const TetOpp& opp1 = oppVec[ ti1 ];

            if ( -1 == opp1._t[ vi0_1 ] || ti0 != opp1.getOppTet( vi0_1 ) )
            {
                std::cout << "Not opp of each other! Tet0: " << ti0 << " Tet1: " << ti1 << std::endl;
                printTetraAndOpp( ti0, tet0, opp0 );
                printTetraAndOpp( ti1, tet1, opp1 );
                exit(-1);
            }

            if ( vi != opp1.getOppVi( vi0_1 ) )
            {
                std::cout << "Vi mismatch! Tet0: " << ti0 << "Tet1: " << ti1 << std::endl;
                exit(-1);
            }
        }
    }

    std::cout << "Adjacency check: Pass\n";

    return;
}

void DelaunayChecker::checkOrientation()
{
    const TetHVec tetVec        = _output->tetVec; 
    const CharHVec tetInfoVec   = _output->tetInfoVec; 

    int count = 0;

    for ( int i = 0; i < ( int ) tetInfoVec.size(); ++i )
    {
        if ( !isTetAlive( tetInfoVec[i] ) ) continue;

        const Tet& t     = tetVec[i];
        const Orient ord = _predWrapper.doOrient3DAdapt( t._v[0], t._v[1], t._v[2], t._v[3] );

        if ( OrientNeg == ord )
            ++count;
    }

    std::cout << "Orient check: ";
    if ( count )
        std::cout << "***Fail*** Wrong orient: " << count;
    else
        std::cout << "Pass";
    std::cout << "\n";

    return;
}

bool DelaunayChecker::checkDelaunay( bool writeFile )
{
    const TetHVec tetVec        = _output->tetVec; 
    const TetOppHVec oppVec     = _output->tetOppVec; 
    const CharHVec tetInfoVec   = _output->tetInfoVec; 

    const int tetNum = ( int ) tetVec.size();
    int facetSum     = 0;
    int extFacetSum  = 0; 

    std::deque< int > que;

    for ( int botTi = 0; botTi < tetNum; ++botTi )
    {
        if ( !isTetAlive( tetInfoVec[ botTi ] ) ) continue;

        const Tet botTet    = tetVec[ botTi ];
        const TetOpp botOpp = oppVec[ botTi ];
            
        if ( botTet.has( _predWrapper._infIdx ) ) 
            extFacetSum++; 

        for ( int botVi = 0; botVi < 4; ++botVi ) // Face neighbours
        {
            // No face neighbour
            if ( -1 == botOpp._t[botVi] )
            {
                ++facetSum;
                continue;
            }

            const int topVi = botOpp.getOppVi( botVi );
            const int topTi = botOpp.getOppTet( botVi );

            if ( topTi < botTi ) continue; // Neighbour will check

            ++facetSum;

            const Tet topTet  = tetVec[ topTi ];
            const int topVert = topTet._v[ topVi ];

            Side side = _predWrapper.doInsphereAdapt( botTet, topVert );

            if ( SideIn != side ) continue;

            int entry = ( botTi << 2 ) | botVi;
            que.push_back( entry );

            if ( !botOpp.isOppSphereFail( botVi ) && 
                !oppVec[ topTi ].isOppSphereFail( topVi ) ) 
            {
                std::cout << "********** Fail: " << botTi << " " << botVi << " " 
                    << topTi << " " << topVi << std::endl;

                const TetOpp opp = oppVec[ topTi ]; 
                const int *ordVi = TetViAsSeenFrom[ topVi ]; 

                for ( int i = 0; i < 3; ++i ) 
                {
                    const int sideTi = opp.getOppTet( ordVi[i] ); 

                    if ( botOpp.isNeighbor( sideTi ) ) 
                        std::cout << "3-2 flip: " << sideTi << std::endl; 
                }
            }
        }
    }

    std::cout << "\nConvex hull facets: " << extFacetSum << std::endl; 
    std::cout << "\nDelaunay check: ";

    if ( que.empty() )
    {
        std::cout << "Pass" << std::endl;
        return true;
    }

    std::cout << "***Fail*** Failed faces: " << que.size() << std::endl;

    if ( writeFile )
    {
        // Write failed facets to file

        std::cout << "Writing failures to file ... ";

        const int pointNum = ( int ) _predWrapper.pointNum();
        const int triNum   = ( int ) que.size();

        std::ofstream oFile( "Failed.ply" );

        oFile << "ply\n";
        oFile << "format ascii 1.0\n";
        oFile << "element vertex " << pointNum << "\n";
        oFile << "property double x\n";
        oFile << "property double y\n";
        oFile << "property double z\n";
        oFile << "element face " << triNum << "\n";
        oFile << "property list uchar int vertex_index\n";
        oFile << "end_header\n";

        // Write points

        for ( int i = 0; i < pointNum; ++i )
        {
            const Point3 pt = _predWrapper.getPoint( i );

            for ( int vi = 0; vi < 3; ++vi )
                oFile << pt._p[ vi ] << " ";
            oFile << "\n";
        }

        // Write failed faces

        for ( int fi = 0; fi < triNum; ++fi )
        {
            const int entry = que[ fi ];
            const int tvi   = entry & 3;
            const int ti    = entry >> 2;

            const Tet tet      = tetVec[ ti ];
            const int* orderVi = TetViAsSeenFrom[ tvi ];

            oFile << "3 ";
            for ( int faceI = 0; faceI < 3; ++faceI )
                oFile << tet._v[ orderVi[ faceI ] ] << " ";
            oFile << "\n";
        }

        std::cout << " done!\n";
    }

    return false;
}

