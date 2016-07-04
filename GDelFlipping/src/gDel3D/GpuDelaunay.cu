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

#include "GpuDelaunay.h"

#include <iomanip>
#include <iostream>

#include "GPU/CudaWrapper.h"
#include "GPU/HostToKernel.h"
#include "GPU/KerCommon.h"
#include "GPU/KerPredicates.h"
#include "GPU/KerDivision.h"
#include "GPU/ThrustWrapper.h"

////
// Consts
////
const int BlocksPerGrid         = 512;
const int ThreadsPerBlock       = 256;
const int PredBlocksPerGrid     = 64;
const int PredThreadsPerBlock   = PRED_THREADS_PER_BLOCK;
const int PredTotalThreadNum    = PredBlocksPerGrid * PredThreadsPerBlock;

////
// GpuDel methods
////
GpuDel::GpuDel() : _params( GDelParams() ), _splaying( _params ) {}

GpuDel::GpuDel( const GDelParams& params ) : _params( params ), _splaying( params ) {}

GpuDel::~GpuDel()
{
    cleanup(); 
}

void GpuDel::compute( const Point3HVec& pointVec, GDelOutput *output )
{
    // Set L1 for kernels
    CudaSafeCall( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );

    _output = output;
    _output->stats.reset(); 

    PerfTimer timer; 

    timer.start(); 
        initForFlip( pointVec );
        splitAndFlip();
        outputToHost(); 
    timer.stop(); 

    _output->stats.totalTime = timer.value(); 

    cleanup(); 

    // 3. Star splaying
    if ( !_params.noSplaying )
        _splaying.fixWithStarSplaying( pointVec, output );

    return;
}

void GpuDel::cleanup()
{
    thrust_free_all(); 

    _pointVec.free();  
    _tetVec.free();
    _oppVec.free();
    _tetInfoVec.free();
    _freeVec.free();
    _tetVoteVec.free(); 
    _actTetVec.free();
    _tetMsgVec.free();
    _flipVec.free(); 
    _orgPointIdx.free(); 
    _vertVec.free();
    _insVertVec.free(); 
    _vertTetVec.free();
    _vertSphereVec.free();
    _vertFreeVec.free(); 
    _counterVec.free(); 

    for ( int i = 0; i < _memPool.size(); ++i ) 
        delete _memPool[ i ]; 

    _memPool.clear(); 

    _orgFlipNum.clear(); 

    _dPredWrapper.cleanup(); 
}

IntDVec& GpuDel::poolPopIntDVec() 
{
    if ( _memPool.empty() ) 
    {
        std::cout << "IntDVec pool empty!" << std::endl; 
    }

    IntDVec *item = _memPool.back(); 

    _memPool.pop_back(); 

    return *item; 
}

IntDVec& GpuDel::poolPeekIntDVec() 
{
    if ( _memPool.empty() ) 
    {
        std::cout << "IntDVec pool empty!" << std::endl; 
    }

    return *_memPool.back(); 
}

void GpuDel::poolPushIntDVec( IntDVec &item ) 
{
    _memPool.push_back( &item ); 
}

void GpuDel::startTiming()
{
    _profTimer.start(); 
}

void GpuDel::pauseTiming()
{
    _profTimer.pause(); 
}

void GpuDel::stopTiming( double &accuTime )
{
    _profTimer.stop(); 
    accuTime += _profTimer.value(); 
}

struct CompareX
{
	__device__ bool operator()( const Point3 &a, const Point3 &b ) const
	{
		return a._p[0] < b._p[0]; 
	}
};

struct Get2Ddist
{
	Point3		_a; 
	RealType	abx, aby, abz; 

	Get2Ddist( const Point3 &a, const Point3 &b ) : _a(a)
	{
		abx = b._p[0] - a._p[0]; 
		aby = b._p[1] - a._p[1]; 
		abz = b._p[2] - a._p[2]; 
	}

	__device__ int operator()( const Point3 &c ) 
	{
		RealType acx = c._p[0] - _a._p[0]; 
		RealType acy = c._p[1] - _a._p[1]; 
		RealType acz = c._p[2] - _a._p[2]; 

		RealType xy = abx * acy - aby * acx; 
		RealType yz = aby * acz - abz * acy; 
		RealType zx = abz * acx - abx * acz; 

		RealType dist = xy * xy + yz * yz + zx * zx; 

		return __float_as_int( (float) dist ); 
	}
};

struct Get3Ddist
{
	Point3		_a; 
	RealType	abx, aby, abz, acx, acy, acz, bc; 
	
	Get3Ddist( const Point3 &a, const Point3 &b, const Point3 &c ) : _a(a) 
	{
		abx = b._p[0] - a._p[0]; 
		aby = b._p[1] - a._p[1]; 
		abz	= b._p[2] - a._p[2]; 
		acx = c._p[0] - a._p[0]; 
		acy = c._p[1] - a._p[1]; 
		acz	= c._p[2] - a._p[2]; 

		bc = abx * acy - aby * acx;
	}

	__device__ int operator()( const Point3 &d ) 
	{
		RealType adx = d._p[0] - _a._p[0]; 
		RealType ady = d._p[1] - _a._p[1]; 
		RealType adz = d._p[2] - _a._p[2]; 

		RealType cd = acx * ady - acy * adx; 
		RealType db = adx * aby - ady * abx; 

		RealType dist = abz * cd + acz * db + adz * bc; 

		return __float_as_int( fabs((float) dist) ); 
	}
};

void GpuDel::constructInitialTetra() 
{
	// First, choose two extreme points along the X axis
	typedef Point3DVec::iterator Point3DIter; 

	thrust::pair< Point3DIter, Point3DIter > ret = thrust::minmax_element( _pointVec.begin(), _pointVec.end(), CompareX() ); 

    int v0 = ret.first - _pointVec.begin(); 
	int v1 = ret.second - _pointVec.begin(); 

	const Point3 p0 = _pointVec[v0]; 
	const Point3 p1 = _pointVec[v1]; 

	// Find the furthest point from v0v1
	IntDVec &distVec = _vertSphereVec; 

	distVec.resize( _pointVec.size() ); 

	thrust::transform( _pointVec.begin(), _pointVec.end(), distVec.begin(), Get2Ddist( p0, p1 ) ); 

	const int v2	= thrust::max_element( distVec.begin(), distVec.end() ) - distVec.begin(); 
	const Point3 p2 = _pointVec[v2]; 

    // Find the furthest point from v0v1v2
	thrust::transform( _pointVec.begin(), _pointVec.end(), distVec.begin(), Get3Ddist( p0, p1, p2 ) ); 

    const int v3	= thrust::max_element( distVec.begin(), distVec.end() ) - distVec.begin(); 
	const Point3 p3 = _pointVec[v3]; 

    if ( _params.verbose )
	{
		std::cout << "Leftmost: " << v0 << " --> " << p0._p[0] << " " << p0._p[1] << " " << p0._p[2] << std::endl; 
		std::cout << "Rightmost: " << v1 << " --> " << p1._p[0] << " " << p1._p[1] << " " << p1._p[2] << std::endl; 
		std::cout << "Furthest 2D: " << v2 << " --> " << p2._p[0] << " " << p2._p[1] << " " << p2._p[2] << std::endl; 
		std::cout << "Furthest 3D: " << v3 << " --> " << p3 ._p[0] << " " << p3 ._p[1] << " " << p3 ._p[2] << std::endl; 
	}

	// Check to make sure the 4 points are not co-planar
	RealType ori = orient3dzero( p0._p, p1._p, p2._p, p3._p ); 

	if ( ori == 0.0 )
	{
		std::cout << "Input too degenerate!!!\n" << std::endl; 
		exit(-1); 
	}

	if ( ortToOrient( ori ) == OrientNeg ) 
		std::swap( v0, v1 ); 

	// Compute the centroid of v0v1v2v3, to be used as the kernel point. 
	_ptInfty._p[0] = ( p0._p[0] + p1._p[0] + p2._p[0] + p3._p[0] ) / 4.0; 
	_ptInfty._p[1] = ( p0._p[1] + p1._p[1] + p2._p[1] + p3._p[1] ) / 4.0; 
	_ptInfty._p[2] = ( p0._p[2] + p1._p[2] + p2._p[2] + p3._p[2] ) / 4.0; 

    // Add the infinity point to the end of the list
    _infIdx = _pointNum - 1; 

    _pointVec.resize( _pointNum ); 
    _pointVec[ _infIdx ] = _ptInfty; 

    // Initialize Inf list size to be zero
    _vertFreeVec[ _infIdx ] = 0; 

	if ( _params.verbose ) 
		std::cout << "Kernel: " << _ptInfty._p[0] << " " << _ptInfty._p[1] << " " << _ptInfty._p[2] << std::endl; 

    // Initialize the predicate wrapper!!!
    _dPredWrapper.init( toKernelPtr( _pointVec ), _pointNum, 
        _params.noSorting ? NULL : toKernelPtr( _orgPointIdx ), 
        _infIdx, PredTotalThreadNum ); 

    setPredWrapperConstant( _dPredWrapper ); 

    // Create the initial triangulation
    IntDVec &newVertVec = _vertSphereVec; 
    Tet firstTet        = { v0, v1, v2, v3 }; 
    IntHVec firstVerts( firstTet._v, firstTet._v + 4 ); 

    newVertVec.copyFromHost( firstVerts ); 

    expandTetraList( &newVertVec, 5, NULL ); 

    // Put the initial tets at the Inf list
    const int firstTetIdx = newVertVec.size() * MeanVertDegree; 

    kerMakeFirstTetra<<< 1, 1 >>>(
        toKernelPtr( _tetVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _tetInfoVec ),
		firstTet, firstTetIdx,
        _infIdx
		);
    CudaCheckError();

    _maxTetNum = _tetVec.size(); 

    // Locate initial positions of points
    _vertTetVec.assign( _pointNum, 0 );

    kerInitPointLocationFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelPtr( _vertTetVec ), 
        firstTet, firstTetIdx
        ); 
    CudaCheckError();

    kerInitPointLocationExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        toKernelPtr( _vertTetVec ), 
        firstTet, firstTetIdx
        ); 
    CudaCheckError();

    // Remove the 4 inserted points
    _vertVec.resize( _pointNum );
    thrust::sequence( _vertVec.begin(), _vertVec.end() );

    compactBothIfNegative( _vertTetVec, _vertVec );
}

// Just expand, no other adjustments or initialization
void GpuDel::expandTetraList( int newTetNum )
{
    const int tetNum        = _tetVec.size(); 
    const bool hasCapacity  = ( newTetNum <= _tetVec._capacity );

    if ( !hasCapacity )
    {
        int growNum = _tetVec._capacity * 1.2; 

        if ( growNum > newTetNum ) 
            growNum = newTetNum; 

        std::cout << "Expanding tetra to: " << growNum << std::endl;

        _tetVec.grow( growNum ); 
        _oppVec.grow( growNum ); 
        _tetInfoVec.grow( growNum ); 
        _freeVec.grow( growNum ); 

        _tetVoteVec.assign( growNum, INT_MAX ); 
        _voteOffset = INT_MAX; 
    }

    _tetVec.expand( newTetNum );
    _oppVec.expand( newTetNum );
    _tetInfoVec.expand( newTetNum );

    // Initialize the free tets
    thrust::fill( _tetInfoVec.begin() + tetNum, _tetInfoVec.end(), 0 );

    return;
}

template< typename T >
__global__ void kerReorder( int* orderArr, T* src, T* dest, int oldInfBlockIdx, int newInfBlockIdx, int size ) 
{
    for ( int idx = getCurThreadIdx(); idx < size; idx += getThreadNum() )
    {
        int newIdx; 

        if ( idx < oldInfBlockIdx ) 
        {
            const int insNum = idx / MeanVertDegree; 
            const int locIdx = idx % MeanVertDegree; 

            newIdx = orderArr[ insNum ] * MeanVertDegree + locIdx; 
        } 
        else
            newIdx = idx - oldInfBlockIdx + newInfBlockIdx; 

        dest[ newIdx ] = src[ idx ]; 
    }
}

template< typename T > 
void GpuDel::reorderVec( IntDVec &orderVec, DevVector< T > &dataVec, int oldInfBlockIdx, int newInfBlockIdx, int size, T* init )
{
    DevVector< T > tempVec( _flipVec ); 

    // Copy data to a temp place
    tempVec.resize( size ); 
    thrust::copy_n( dataVec.begin(), size, tempVec.begin() ); 

    // Initialize if needed
    if ( init != NULL ) 
        dataVec.fill( *init );

    kerReorder<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelPtr( orderVec ), 
        toKernelPtr( tempVec ),
        toKernelPtr( dataVec ),
        oldInfBlockIdx, newInfBlockIdx, size
        ); 
    CudaCheckError(); 
}

// Make sure you have enough space in dataVec. 
// No resize is done here. 
template< typename T > 
void GpuDel::pushVecTail( DevVector< T > &dataVec, int size, int from, int gap )
{
    DevVector< T > tempVec( _flipVec ); 

    int tail = size - from; 

    tempVec.resize( tail ); 

    thrust::copy_n( dataVec.begin() + from, tail, tempVec.begin() ); 
    thrust::copy_n( tempVec.begin(), tail, dataVec.begin() + from + gap ); 
}

// Expansion and reserving a storage for each new vertex
void GpuDel::expandTetraList( IntDVec *newVertVec, int tailExtra, IntDVec *tetToVert, bool sort )
{
    const int oldTetNum       = _tetVec.size(); 
    const int insVertNum      = ( newVertVec != NULL ) ? newVertVec->size() : 0;
    const int insExtraSpace   = insVertNum * MeanVertDegree;
    const int newTetNum       = oldTetNum + insExtraSpace + tailExtra; 

    expandTetraList( newTetNum ); 

    if ( insExtraSpace > 0 ) 
    {
        // Store the new vertices
        int oldInsNum       = _insVertVec.size(); 
        int newInsNum       = oldInsNum + insVertNum; 
        int oldInfBlockIdx  = oldInsNum * MeanVertDegree; 
        int newInfBlockIdx  = newInsNum * MeanVertDegree; 

        _insVertVec.resize( newInsNum ); 
        thrust::copy( newVertVec->begin(), newVertVec->end(), _insVertVec.begin() + oldInsNum ); 

        if ( sort ) 
        {
            IntDVec &tempVec = *newVertVec;

            const int scatterIdx = newInsNum; 

            tempVec.assign( newInsNum + _pointNum, -1 );

            thrust::counting_iterator<int> zero_iter( 0 ); 
            thrust::counting_iterator<int> insNum_iter( newInsNum ); 
            thrust::counting_iterator<int> pointNum_iter( _pointNum ); 

            thrust::scatter( zero_iter, insNum_iter, _insVertVec.begin(), tempVec.begin() + scatterIdx ); 

            // Get the sorted list of points
            thrust::copy_if( zero_iter, pointNum_iter, tempVec.begin() + scatterIdx, _insVertVec.begin(), IsNotNegative() ); 

            // Get the reverse map            
            kerMakeReverseMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelArray( _insVertVec ), 
                toKernelPtr( tempVec ) + scatterIdx, 
                toKernelPtr( tempVec ), 
                oldInsNum
                ); 
            CudaCheckError(); 

            // Update tet indices
            kerUpdateBlockOppTetIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelPtr( _oppVec ), 
                toKernelPtr( tempVec ),
                oldInfBlockIdx, newInfBlockIdx, oldTetNum
                ); 
            CudaCheckError(); 

            kerUpdateTetIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelArray( _vertTetVec ),
                toKernelPtr( tempVec ),
                oldInfBlockIdx, newInfBlockIdx
                ); 
            CudaCheckError(); 

            // Use _flipVec as a temp buffer
            int4* initInt4 = NULL;
            int*  initInt  = NULL; 
            char  initInfo = 0; 

            DevVector< int4 > tetInt4Vec( _tetVec ); 
            tetInt4Vec.resize( newTetNum ); 
            reorderVec( tempVec, tetInt4Vec, oldInfBlockIdx, newInfBlockIdx, oldTetNum, initInt4 ); 

            DevVector< int4 > oppInt4Vec( _oppVec ); 
            oppInt4Vec.resize( newTetNum ); 
            reorderVec( tempVec, oppInt4Vec, oldInfBlockIdx, newInfBlockIdx, oldTetNum, initInt4 ); 

            reorderVec( tempVec, _tetInfoVec, oldInfBlockIdx, newInfBlockIdx, oldTetNum, &initInfo );        

            if ( tetToVert != NULL ) 
            {
                tetToVert->grow( newTetNum ); 

                reorderVec( tempVec, *tetToVert, oldInfBlockIdx, newInfBlockIdx, oldTetNum, initInt ); 
            }

            // Update the free list
            kerUpdateBlockVertFreeList<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelArray( _insVertVec ), 
                toKernelPtr( _vertFreeVec ), 
                toKernelPtr( _freeVec ),
                toKernelPtr( tempVec ) + scatterIdx,
                oldInsNum
                ); 
            CudaCheckError(); 

            kerShiftInfFreeIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelPtr( _vertFreeVec ),
                toKernelPtr( _freeVec ), 
                _infIdx, oldInfBlockIdx, insExtraSpace
                ); 
            CudaCheckError(); 
        }
        else
        {
            // Just move the Inf chunk to get space for new verts

            // Update tet indices
            kerShiftOppTetIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelPtr( _oppVec ), 
                oldTetNum, oldInfBlockIdx, insExtraSpace
                ); 
            CudaCheckError(); 

            kerShiftTetIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelArray( _vertTetVec ),
                oldInfBlockIdx, insExtraSpace
                ); 
            CudaCheckError(); 

            kerShiftInfFreeIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelPtr( _vertFreeVec ),
                toKernelPtr( _freeVec ),
                _infIdx, oldInfBlockIdx, insExtraSpace
                ); 
            CudaCheckError(); 

            pushVecTail( _tetInfoVec, oldTetNum, oldInfBlockIdx, insExtraSpace ); 
            pushVecTail( _tetVec,     oldTetNum, oldInfBlockIdx, insExtraSpace ); 
            pushVecTail( _oppVec,     oldTetNum, oldInfBlockIdx, insExtraSpace ); 

            if ( tetToVert != NULL ) 
            {
                tetToVert->grow( newTetNum ); 
                pushVecTail( *tetToVert, oldTetNum, oldInfBlockIdx, insExtraSpace );
            }

            kerUpdateVertFreeList<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelArray( *newVertVec ), 
                toKernelPtr( _vertFreeVec ), 
                toKernelPtr( _freeVec ),
                oldInfBlockIdx
                ); 
            CudaCheckError(); 

            // Initialize the free tets
            thrust::fill_n( _tetInfoVec.begin() + oldInfBlockIdx, insExtraSpace, 0 );
        }
    }

    // No need to initialize the tailExtra, since they're gonna be used directly. 
    // No need to even push them into the free list!
}

void GpuDel::initForFlip( const Point3HVec pointVec )
{
    startTiming(); 

    _pointNum			= pointVec.size() + 1;	// Plus the infinity point
    const int TetMax    = (int) ( _pointNum * 8.5 );

    _pointVec.resize( _pointNum );  // 1 additional slot for the infinity point
    _pointVec.copyFromHost( pointVec );

	// Find the min and max coordinate value
    typedef thrust::device_ptr< RealType > RealPtr; 
	RealPtr coords( ( RealType* ) toKernelPtr( _pointVec ) ); 
    thrust::pair< RealPtr, RealPtr> ret
        = thrust::minmax_element( coords, coords + _pointVec.size() * 3 ); 

    _minVal = *ret.first; 
    _maxVal = *ret.second; 

    if ( _params.verbose ) 
        std::cout << "\n_minVal = " << _minVal << ", _maxVal == " << _maxVal << std::endl; 

    // Initialize _memPool
    assert( _memPool.empty() && "_memPool is not empty!" ); 

    for ( int i = 0; i < 2; ++i ) 
        _memPool.push_back( new IntDVec( TetMax ) ); 

    // Allocate space
    _tetVec.resize( TetMax );
    _oppVec.resize( TetMax );
    _tetInfoVec.resize( TetMax );
    _freeVec.resize( TetMax );
    _tetVoteVec.assign( TetMax, INT_MAX ); 

    _voteOffset = INT_MAX; 

    _flipVec.resize( TetMax / 2 ); 
    _actTetVec.resize( TetMax );
    _vertSphereVec.resize( _pointNum );
    _vertFreeVec.assign( _pointNum, 0 ); 
    _insVertVec.resize( _pointNum ); 
    _tetMsgVec.assign( TetMax, make_int2( -1, -1 ) );

    _flipVec.expand( 0 ); 
    _tetVec.expand( 0 ); 
    _oppVec.expand( 0 ); 
    _tetInfoVec.expand( 0 ); 
    _insVertVec.expand( 0 ); 

    _counterVec.resize( CounterNum ); 

    // Sort points along space curve
    if ( !_params.noSorting )
    {
        stopTiming( _output->stats.initTime ); 
        startTiming(); 

        IntDVec &valueVec = poolPopIntDVec(); 
        valueVec.resize( _pointVec.size() );

        _orgPointIdx.resize( _pointNum );   // 1 slot for the infinity point
        thrust::sequence( _orgPointIdx.begin(), _orgPointIdx.end(), 0 ); 

        thrust_transform_GetMortonNumber( 
            _pointVec.begin(), _pointVec.end(), valueVec.begin(),
            _minVal, _maxVal );

        thrust_sort_by_key( valueVec.begin(), valueVec.end(), 
            make_zip_iterator( make_tuple( 
                _orgPointIdx.begin(), _pointVec.begin() ) ) ); 

        poolPushIntDVec( valueVec ); 

        stopTiming( _output->stats.sortTime ); 
        startTiming(); 
    }

    // Create first upper-lower tetra
	constructInitialTetra(); 

	// Initialize CPU predicate wrapper
	_predWrapper.init( pointVec, _ptInfty ); 
	
    stopTiming( _output->stats.initTime ); 

    return;
}

void GpuDel::doFlippingLoop( CheckDelaunayMode checkMode )
{
    startTiming(); 

    int flipLoop = 0; 

    _actTetMode = ActTetMarkCompact;
	_counterVec.fill( 0 ); 

    while ( doFlipping( checkMode ) ) 
    {        
        ++flipLoop; 

        if ( _flipVec.capacity() - _flipVec.size() < _orgFlipNum.back() ) 
        {
            stopTiming( _output->stats.flipTime ); 
            relocateAll(); 
            startTiming(); 
        }
    } 

    stopTiming( _output->stats.flipTime ); 
}

void GpuDel::splitAndFlip()
{
    int insLoop = 0;

    _doFlipping = !_params.insertAll; 

    //////////////////
    while ( _vertVec.size() > 0 )
    //////////////////
    {
        ////////////////////////
        splitTetra();
        ////////////////////////

        if ( _doFlipping ) 
        {
            doFlippingLoop( SphereFastOrientFast ); 

            markSpecialTets(); 
            doFlippingLoop( SphereExactOrientSoS ); 

            relocateAll(); 
            //////////////////////////
        }

        ++insLoop;
    }

    //////////////////////////////
    if ( !_doFlipping ) 
    {
        doFlippingLoop( SphereFastOrientFast ); 

        markSpecialTets(); 
        doFlippingLoop( SphereExactOrientSoS ); 
    }

    /////////////////////////////

    if ( _params.verbose ) 
        std::cout << "\nInsert loops: " << insLoop << std::endl;

    return;
}

void GpuDel::markSpecialTets()
{
    startTiming(); 

    kerMarkSpecialTets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _tetInfoVec ), 
        toKernelPtr( _oppVec )
        );    
    CudaCheckError(); 

    stopTiming( _output->stats.flipTime ); 
}

void GpuDel::splitTetra()
{
    startTiming(); 

    ////
    // Rank points
    ////
    const int vertNum = _vertVec.size();
    const int tetNum  = _tetVec.size();

    _vertSphereVec.resize( vertNum );

    IntDVec &tetSphereVec = poolPopIntDVec(); 
    tetSphereVec.assign( tetNum, INT_MIN );

    kerVoteForPoint<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( _tetVec ),
        toKernelPtr( _vertSphereVec ),
        toKernelPtr( tetSphereVec ),
        _params.insRule );
    CudaCheckError();

    IntDVec &tetToVert = poolPopIntDVec(); 
    tetToVert.assign( tetNum, INT_MAX );

    kerPickWinnerPoint<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( _vertSphereVec ),
        toKernelPtr( tetSphereVec ),
        toKernelPtr( tetToVert ) );
    CudaCheckError();

    poolPushIntDVec( tetSphereVec ); 

    ////
    // Highlight inserted verts
    ////
    kerNegateInsertedVerts<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTetVec ), 
        toKernelPtr( tetToVert )
        ); 
    CudaCheckError(); 

    ////
    // Collect insertable verts
    ////
    IntDVec &newVertVec = _vertSphereVec; 
    IntDVec &realInsVertVec = poolPopIntDVec(); 

    _insNum = thrust_copyIf_Insertable( _vertTetVec, newVertVec ); 

    // If there's just a few points
    if ( vertNum - _insNum < _insNum && _insNum < 0.1 * _pointNum ) 
        _doFlipping = false; 

    realInsVertVec.resize( _insNum ); 

    thrust::gather( newVertVec.begin(), newVertVec.end(), _vertVec.begin(), realInsVertVec.begin() ); 

    ////
    // Prepare space
    ////
    expandTetraList( &realInsVertVec, 0, &tetToVert, !_params.noSorting && _doFlipping ); 

    poolPushIntDVec( realInsVertVec ); 

    if ( _params.verbose )
        std::cout << "Insert: " << _insNum << std::endl;

    // Mark all tetra as non-empty
    kerMarkTetEmpty<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _tetInfoVec ) 
        ); 
    CudaCheckError();

    ////
    // Update the location of the points
    ////
    stopTiming( _output->stats.splitTime ); 
    startTiming(); 

    kerSplitPointsFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( tetToVert ),
        toKernelPtr( _tetVec ),
        toKernelPtr( _tetInfoVec ),
        toKernelArray( _freeVec )
        );

    kerSplitPointsExactSoS<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( tetToVert ),
        toKernelPtr( _tetVec ),
        toKernelPtr( _tetInfoVec ),
        toKernelArray( _freeVec )
        );
    CudaCheckError();

    stopTiming( _output->stats.relocateTime ); 
    startTiming(); 

    ////
    // Split the tetras
    ////
    kerSplitTetra<<< BlocksPerGrid, 32 >>>(
        toKernelArray( newVertVec ),
        toKernelArray( _insVertVec ),
        toKernelPtr( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( tetToVert ),
        toKernelPtr( _tetVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _tetInfoVec ),
        toKernelPtr( _freeVec ),
        toKernelPtr( _vertFreeVec ),
        _infIdx
        );
    CudaCheckError();

    poolPushIntDVec( tetToVert ); 

    ////
    // Shrink vertex and free lists
    ////
    compactBothIfNegative( _vertTetVec, _vertVec );

    stopTiming( _output->stats.splitTime ); 

    return;
}

bool GpuDel::doFlipping( CheckDelaunayMode checkMode )
{
/////////////////////////////////////////////////////////////////////
    ////
    // Compact active tetra
    ////

    switch ( _actTetMode ) 
    {
    case ActTetMarkCompact: 
        thrust_copyIf_IsActiveTetra( _tetInfoVec, _actTetVec ); 
        break; 

    case ActTetCollectCompact: 
        compactIfNegative( _actTetVec, poolPeekIntDVec() );
        break; 
    }

    int tetNum  = _tetVec.size();
    int actNum  = _actTetVec.size(); 

/////////////////////////////////////////////////////////////////////
    ////
    // Check actNum, switch mode or quit if necessary
    ////

    // No more work
    if ( 0 == actNum ) return false;

    // Little work, leave it for the Exact iterations
    if ( checkMode != SphereExactOrientSoS && 
        actNum < PredBlocksPerGrid * PredThreadsPerBlock ) 
        return false; 

    // Too little work, leave it for the last round of flipping
    if ( actNum < PredThreadsPerBlock && _doFlipping ) 
        return false; 

    // See if there's little work enough to switch to collect mode. 
    // Safety check: make sure there's enough space to collect
    if ( actNum < BlocksPerGrid * ThreadsPerBlock && actNum * 3 < _actTetVec.capacity() ) 
        _actTetMode = ActTetCollectCompact; 
    else
        _actTetMode = ActTetMarkCompact; 

    if ( _voteOffset - tetNum < 0 ) 
    {
        _tetVoteVec.assign( _tetVoteVec.capacity(), INT_MAX ); 
        _voteOffset = INT_MAX; 
    }

    _tetVoteVec.expand( tetNum );   
    _voteOffset -= tetNum; 

/////////////////////////////////////////////////////////////////////
    ////
    // Vote for flips
    ////

    IntDVec &voteVec = poolPopIntDVec(); 
    voteVec.resize( actNum );
    
    dispatchCheckDelaunay( checkMode, voteVec ); 

/////////////////////////////////////////////////////////////////////
    ////
    // Mark rejected flips
    ////

    int counterExact = 0; 

    if ( _params.verbose )
        counterExact = _counterVec[ CounterExact ]; 

    IntDVec &flipToTet = ( _actTetMode == ActTetCollectCompact ) ? poolPopIntDVec() : voteVec; 

    flipToTet.resize( actNum ); 

    kerMarkRejectedFlips<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTetVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _tetVoteVec ),
        toKernelPtr( _tetInfoVec ),
        toKernelPtr( voteVec ),
        toKernelPtr( flipToTet ),
        toKernelPtr( _counterVec ),
        _voteOffset
        );
    CudaCheckError();

    if ( _actTetMode == ActTetCollectCompact )
        poolPushIntDVec( voteVec ); 

/////////////////////////////////////////////////////////////////////
    ////
    // Compact flips
    ////

    const int flipNum = ( _actTetMode == ActTetCollectCompact )
        ? _counterVec[ CounterFlip ]
        : compactIfNegative( flipToTet, poolPeekIntDVec() ); 

    flipToTet.resize( flipNum );    // Resize to fit with content

    _output->stats.totalFlipNum += flipNum; 

/////////////////////////////////////////////////////////////////////

#pragma region Diagnostic
    if ( _params.verbose )
    {
        const int flip23Num = thrust::transform_reduce(
            flipToTet.begin(), flipToTet.end(), IsFlip23(), 0, thrust::plus<int>() ); 
        const int flip32Num = flipNum - flip23Num; 

        std::cout << "  Active: " << actNum
            << " Flip: " << flipNum 
            << " ( 2-3: " << flip23Num << " 3-2: " << flip32Num << " )"
            << " Exact: " << ( checkMode == SphereExactOrientSoS ? counterExact : -1 )
            << std::endl;
    }
#pragma endregion

    if ( 0 == flipNum )
    {
        poolPushIntDVec( flipToTet ); 
        return false;
    }

    ////
    // Allocate slots for 2-3 flips
    ////
    IntDVec &flip23NewSlot = poolPopIntDVec(); 

    flip23NewSlot.resize( flipNum ); 

    kerAllocateFlip23Slot<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( flipToTet ), 
        toKernelPtr( _tetVec ), 
        toKernelPtr( _vertFreeVec ), 
        toKernelPtr( _freeVec ), 
        toKernelPtr( flip23NewSlot ),
        _infIdx, tetNum
        ); 
    CudaCheckError(); 

    ////
    // Expand tetra list for flipping
    ////
    int extraSlot = -_vertFreeVec[ _infIdx ]; 

    if ( extraSlot > 0 ) 
    {
        _vertFreeVec[ _infIdx ] = 0; 
        expandTetraList( NULL, extraSlot, NULL ); 
    }

    _maxTetNum = std::max( _maxTetNum, (int) _tetVec.size() );

    // Expand flip vector
    const int orgFlipNum    = _flipVec.size(); 
    const int expFlipNum    = orgFlipNum + flipNum; 

    _flipVec.grow( expFlipNum ); 

    // _tetMsgVec contains two components. 
    // - .x is the encoded new neighbor information
    // - .y is the flipIdx as in the flipVec (i.e. globIdx)
    // As such, we do not need to initialize it to -1 to 
    // know which tets are not flipped in the current rount. 
    // We can rely on the flipIdx being > or < than orgFlipIdx. 
    // Note that we have to initialize everything to -1 
    // when we clear the flipVec and reset the flip indexing. 
    //
    if ( _tetMsgVec.capacity() < _tetVec.size() ) 
        _tetMsgVec.assign( _tetVec.size(), make_int2( -1, -1 ) );
    else
        _tetMsgVec.resize( _tetVec.size() ); 

    ////
    // Expand active tet vector
    ////
    if ( _actTetMode == ActTetCollectCompact ) 
        _actTetVec.grow( actNum + flipNum * 2 );

/////////////////////////////////////////////////////////////////////
    ////
    // Flipping
    ////
    // 32 ThreadsPerBlock is optimal
    kerFlip<<< BlocksPerGrid, 32 >>>( 
        toKernelArray( flipToTet ),
        toKernelPtr( _tetVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _tetInfoVec ),
        toKernelPtr( _tetMsgVec ),
        toKernelPtr( _flipVec ),
        toKernelPtr( flip23NewSlot ), 
        toKernelPtr( _vertFreeVec ), 
        toKernelPtr( _freeVec ),
        ( _actTetMode == ActTetCollectCompact ) ? toKernelPtr( _actTetVec ) + actNum : NULL,
        toKernelArray( _insVertVec ), 
        _infIdx, orgFlipNum
        ); 
    CudaCheckError(); 

    _orgFlipNum.push_back( orgFlipNum ); 

    poolPushIntDVec( flipToTet ); 

    ////
    // Update oppTet
    ////
    kerUpdateOpp<<< BlocksPerGrid, 32 >>>(
        toKernelPtr( _flipVec ) + orgFlipNum,
        toKernelPtr( _oppVec ),
        toKernelPtr( _tetMsgVec ),
        toKernelPtr( flip23NewSlot ),
        orgFlipNum, flipNum
        ); 
    CudaCheckError();

    poolPushIntDVec( flip23NewSlot ); 

/////////////////////////////////////////////////////////////////////

    return true;
}

void GpuDel::dispatchCheckDelaunay
( 
CheckDelaunayMode   checkMode,
IntDVec&            voteVec
) 
{
    switch ( checkMode ) 
    {
    case SphereFastOrientFast: 
        kerCheckDelaunayFast<<< BlocksPerGrid, PredThreadsPerBlock >>>(
            toKernelArray( _actTetVec ),
            toKernelPtr( _tetVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( _tetInfoVec ),
            toKernelPtr( _tetVoteVec ),
            toKernelPtr( voteVec ),
            toKernelPtr( _counterVec ), 
            _voteOffset
            );
        CudaCheckError();
        break; 

    case SphereExactOrientSoS:
        Int2DVec exactCheckVi( poolPeekIntDVec() ); 
        exactCheckVi.resize( _actTetVec.size() ); 

        int ns = PredThreadsPerBlock * 2 * sizeof(int2); 

        kerCheckDelaunayExact_Fast<<< BlocksPerGrid, PredThreadsPerBlock, ns >>>(
            toKernelArray( _actTetVec ),
            toKernelPtr( _tetVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( _tetInfoVec ),
            toKernelPtr( _tetVoteVec ),
            toKernelPtr( voteVec ),
            toKernelPtr( exactCheckVi ), 
            toKernelPtr( _counterVec ), 
            _voteOffset
            );

        kerCheckDelaunayExact_Exact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
            toKernelPtr( _actTetVec ),
            toKernelPtr( _tetVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( _tetInfoVec ),
            toKernelPtr( _tetVoteVec ),
            toKernelPtr( voteVec ),
            toKernelPtr( exactCheckVi ), 
            toKernelPtr( _counterVec ), 
            _voteOffset
            );
        CudaCheckError();

        break; 
    }
}

void GpuDel::compactTetras()
{
    const int tetNum = _tetVec.size(); 

    IntDVec &prefixVec = poolPopIntDVec(); 
    
    prefixVec.resize( tetNum ); 

    thrust::transform_inclusive_scan( _tetInfoVec.begin(), _tetInfoVec.end(), 
        prefixVec.begin(), TetAliveStencil(), thrust::plus<int>() ); 

    int newTetNum = prefixVec[ tetNum - 1 ];
    int freeNum   = tetNum - newTetNum; 

    _freeVec.resize( freeNum ); 

    kerCollectFreeSlots<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelPtr( _tetInfoVec ), 
        toKernelPtr( prefixVec ),
        toKernelPtr( _freeVec ),
        newTetNum
        ); 
    CudaCheckError(); 

    // Make map
    kerMakeCompactMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _tetInfoVec ), 
        toKernelPtr( prefixVec ),
        toKernelPtr( _freeVec ),
        newTetNum
        ); 
    CudaCheckError(); 

    // Reorder the tets
    kerCompactTets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _tetInfoVec ), 
        toKernelPtr( prefixVec ), 
        toKernelPtr( _tetVec ), 
        toKernelPtr( _oppVec ),
        newTetNum
        ); 
    CudaCheckError(); 

    _tetVec.resize( newTetNum ); 
    _oppVec.resize( newTetNum ); 

    poolPushIntDVec( prefixVec ); 
}

void GpuDel::relocateAll()
{
    if ( _flipVec.size() == 0 ) 
        return ; 

    startTiming(); 

    // This has to be resized to _maxTetNum, i.e. max tetVec size 
    // during all the previous flipping loop. 
    // Reason: During the flipping, the tetVec size might be 
    //   larger than the current tetVec size. 
    IntDVec &tetToFlip = poolPopIntDVec(); 
    tetToFlip.assign( _maxTetNum, -1 ); 

    _maxTetNum = _tetVec.size(); 

    // Rebuild the pointers from back to forth
    int nextFlipNum = _flipVec.size(); 

    for ( int i = _orgFlipNum.size() - 1; i >= 0; --i ) 
    {
        int prevFlipNum = _orgFlipNum[ i ]; 
        int flipNum     = nextFlipNum - prevFlipNum; 

        kerUpdateFlipTrace<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelPtr( _flipVec ), 
            toKernelPtr( tetToFlip ),
            prevFlipNum, 
            flipNum 
            ); 

        nextFlipNum = prevFlipNum; 
    }
    CudaCheckError(); 
        
    // Relocate points
    kerRelocatePointsFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( tetToFlip ),
        toKernelPtr( _flipVec )
        );

    kerRelocatePointsExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        toKernelArray( _vertVec ),
        toKernelPtr( _vertTetVec ),
        toKernelPtr( tetToFlip ),
        toKernelPtr( _flipVec )
        );
    CudaCheckError();

    // Just clean up the flips
    _flipVec.resize( 0 ); 
    _orgFlipNum.clear(); 

    // Gotta initialize the tetMsgVec
    _tetMsgVec.assign( _tetMsgVec.capacity(), make_int2( -1, -1 ) );

    poolPushIntDVec( tetToFlip ); 

	stopTiming( _output->stats.relocateTime ); 
}

void GpuDel::outputToHost()
{
    startTiming(); 

    compactTetras(); 

    if ( !_params.noSorting ) 
    {
        // Change the indices back to the original order
        kerUpdateVertIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelArray( _tetVec ), 
            toKernelPtr( _orgPointIdx )
            ); 
        CudaCheckError(); 
    }

    ////
    if ( !_params.noSplaying )
    {
        // Gather in-sphere failed vertices

		IntDVec failVertVec( _pointNum, -1 );
        IntDVec vertTetVec( _pointNum );

        kerGatherFailedVerts<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelArray( _tetVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( failVertVec ),
            toKernelPtr( vertTetVec )
            );
        CudaCheckError();

        compactIfNegative( failVertVec );

        failVertVec.copyToHost( _output->failVertVec );
        vertTetVec.copyToHost( _output->vertTetVec );
    }

    // _output triangulation to host memory   
    _output->tetVec.reserve( _tetVec.size() * 1.2 );
    _output->tetOppVec.reserve( _oppVec.size() * 1.2 );
    _output->tetInfoVec.reserve( _tetInfoVec.size() * 1.2 );

    _tetVec.copyToHost( _output->tetVec );
    _oppVec.copyToHost( _output->tetOppVec );
    
    // Tet list is compacted, so all are alive!
    //_tetInfoVec.copyToHost( _output->tetInfoVec );
    _output->tetInfoVec.assign( _tetVec.size(), 1 ); 

    // _output Infty point
    _output->ptInfty = _predWrapper.getPoint( _infIdx ); 

    ////
    stopTiming( _output->stats.outTime ); 

    if ( _params.verbose )
        std::cout << "# Tetras:     " << _tetVec.size() << std::endl << std::endl; 

    return;
}
