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

#include "CommonTypes.h"

#include "GPU/GPUDecl.h"
#include "GPU/DPredWrapper.h"

#include "CPU/Splaying.h"
#include "CPU/PredWrapper.h"

#include "PerfTimer.h"

class GpuDel
{
public:
	GpuDel();
	GpuDel( const GDelParams& params );
    ~GpuDel(); 
    
    void compute( const Point3HVec& input, GDelOutput *output );

private:
    // Execution configuration
    const GDelParams _params; 
    GDelOutput*      _output; 

    // Input
    Point3DVec  _pointVec; 

    // Output - Size proportional to tetNum
    TetDVec     _tetVec;            // Tetra list
    TetOppDVec  _oppVec;            // Opposite info
    CharDVec    _tetInfoVec;        // Tetra status (changed, deleted, etc.)

    // Supplemental arrays - Size proportional to tetNum
    IntDVec       _freeVec;         // List of free slots
    IntDVec       _actTetVec;       // List of active tetras
    Int2DVec      _tetMsgVec;       // Neighboring information when updating opps after flipping
    FlipDVec      _flipVec;         // Flip DAG

    std::vector<IntDVec*> _memPool;  // Memory pool, two items each of size TetMax

    // State
    bool        _doFlipping;        // To switch flipping on/off for some insertion round
    ActTetMode  _actTetMode;        // Compact/Collect mode to gather active tetras
    int         _insNum;            // Number of points inserted
    int         _voteOffset;        // To avoid reseting the voting array every round

    // Supplemental arrays - Size proportional to vertNum
    IntDVec       _orgPointIdx;     // Original point indices (used when sorting)
    IntDVec       _vertVec;         // List of remaining points
    IntDVec       _vertTetVec;      // Vertex location (tetra)
    IntDVec       _tetVoteVec;      // Vote tetras during flipping
    IntDVec       _vertSphereVec;   // Insphere value for voting during point insertion
    IntDVec       _vertFreeVec;     // Number of free slot per vertex
    IntDVec       _insVertVec;      // List of inserted vertices

    // Very small
	int			  _pointNum;        // Number of input points
    RealType      _minVal;          // Min and
    RealType      _maxVal;          //    max coordinate value
    IntHVec       _orgFlipNum;      // Number of flips in flipVec "before" each iteration
    IntDVec       _counterVec;      // Some device memory counters
    int           _maxTetNum;       // Maximum size of tetVec

    // Predicates
	Point3		  _ptInfty;         // The kernel point
    int           _infIdx;          // Kernel point index
    DPredWrapper  _dPredWrapper;    // Device predicate wrapper
	PredWrapper   _predWrapper;     // Host predicate wrapper

    // Star splaying
    Splaying      _splaying;          // Star splaying engine

    // Timing
    CudaTimer _profTimer; 

private:
    // Memory pool
    IntDVec &poolPopIntDVec();
    IntDVec &poolPeekIntDVec();
    void poolPushIntDVec( IntDVec &item );

    // Helpers
	void constructInitialTetra();
    void markSpecialTets();
    void expandTetraList( int newTetNum );
    void splitTetra();
    void doFlippingLoop( CheckDelaunayMode checkMode );
    bool doFlipping( CheckDelaunayMode checkMode );
    void dispatchCheckDelaunay( CheckDelaunayMode checkMode, IntDVec& voteVec ); 
    void compactTetras();
    void relocateAll();

    // Timing
    void startTiming(); 
    void pauseTiming(); 
    void stopTiming( double& accuTime ); 

    // Sorting
    void expandTetraList( IntDVec *newVertVec, int tailExtra, IntDVec *tetToVert, bool sort = false );

    template< typename T > 
    void reorderVec( IntDVec &orderVec, DevVector< T > &dataVec, int oldInfBlockIdx, int newInfBlockIdx, int size, T* init );

    template< typename T > 
    void pushVecTail( DevVector< T > &dataVec, int size, int from, int gap );

    // Main
    void initForFlip( const Point3HVec pointVec ); 
    void splitAndFlip();
    void outputToHost();
    void cleanup(); 

}; // class GpuDel
