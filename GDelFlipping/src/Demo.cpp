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

#include <iomanip>

#include "gDel3D/GpuDelaunay.h"

#include "DelaunayChecker.h"
#include "InputCreator.h"

const int deviceIdx     = 0; 
const int seed          = 123456789;
const int pointNum      = 100000; 
const Distribution dist = UniformDistribution;

Point3HVec   pointVec; 
GDelOutput   output; 

// Forward declaration
void summarize( int pointNum, const GDelOutput& output );

int main( int argc, const char* argv[] )
{  
    CudaSafeCall( cudaSetDevice( deviceIdx ) );
    CudaSafeCall( cudaDeviceReset() );

    // 1. Create points
    std::cout << "Generating input...\n"; 

    InputCreator().makePoints( pointNum, dist, pointVec, seed );

    // 2. Compute the 3D Delaunay triangulation
    std::cout << "Constructing 3D Delaunay triangulation...\n"; 

    // Optionally, some parameters can be specified. Here are the defaults:
    //GDelParams params; 
    //params.insertAll    = false; 
    //params.insRule      = InsCircumcenter; 
    //params.noSorting    = false; 
    //params.noSplaying   = false; 
    //params.verbose      = false; 

    //GpuDel triangulator( params ); 
  
    GpuDel triangulator; 

    triangulator.compute( pointVec, &output );

    // 3. Check correctness
    std::cout << "Checking...\n";

    DelaunayChecker checker( &pointVec, &output ); 
    checker.checkEuler();
    checker.checkOrientation();
    checker.checkAdjacency();
    checker.checkDelaunay( false );

    // 4. Summary
    summarize( pointNum, output ); 

    return 0;
}

void summarize( int pointNum, const GDelOutput& output ) 
{
    ////
    // Summarize on screen
    ////
    std::cout << std::endl; 
    std::cout << "---- SUMMARY ----" << std::endl; 
    std::cout << std::endl; 

    std::cout << "PointNum       " << pointNum                                            << std::endl;
    std::cout << "FP Mode        " << (( sizeof( RealType ) == 8) ? "Double" : "Single")  << std::endl; 
    std::cout << std::endl; 
    std::cout << std::fixed << std::right << std::setprecision( 2 ); 
    std::cout << "TotalTime (ms) " << std::setw( 10 ) << output.stats.totalTime    << std::endl; 
    std::cout << "InitTime       " << std::setw( 10 ) << output.stats.initTime     << std::endl; 
    std::cout << "SplitTime      " << std::setw( 10 ) << output.stats.splitTime    << std::endl; 
    std::cout << "FlipTime       " << std::setw( 10 ) << output.stats.flipTime     << std::endl; 
    std::cout << "RelocateTime   " << std::setw( 10 ) << output.stats.relocateTime << std::endl; 
    std::cout << "SortTime       " << std::setw( 10 ) << output.stats.sortTime     << std::endl;
    std::cout << "OutTime        " << std::setw( 10 ) << output.stats.outTime      << std::endl; 
    std::cout << "SplayingTime   " << std::setw( 10 ) << output.stats.splayingTime << std::endl; 
    std::cout << std::endl;                              
    std::cout << "# Flips        " << std::setw( 10 ) << output.stats.totalFlipNum << std::endl; 
    std::cout << "# Failed verts " << std::setw( 10 ) << output.stats.failVertNum  << std::endl; 
    std::cout << "# Final stars  " << std::setw( 10 ) << output.stats.finalStarNum << std::endl; 
}

