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

#include "ThrustWrapper.h"

#include <map>

#include <thrust/system/cuda/execution_policy.h>

class CachedAllocator
{
private:
    const int BlockSize; 

    typedef std::multimap< std::ptrdiff_t, char * >     FreeBlocks;
    typedef std::map< char *, std::ptrdiff_t >          AllocBlocks;

    FreeBlocks freeBlocks;
    AllocBlocks allocBlocks;

public:
    // just allocate bytes
    typedef char value_type;

    CachedAllocator() 
        : BlockSize( 4096 ) {}

    void freeAll()
    {
        size_t totalSize = 0; 

        // deallocate all outstanding blocks in both lists
        for( FreeBlocks::iterator i = freeBlocks.begin();
             i != freeBlocks.end();
             ++i )
        {
            cudaFree( i->second );
            totalSize += i->first; 
        }

        for( AllocBlocks::iterator i = allocBlocks.begin();
             i != allocBlocks.end();
             ++i )
        {
            cudaFree( i->first );
            totalSize += i->second; 
        }

        freeBlocks.clear(); 
        allocBlocks.clear(); 

        //std::cout << "*** CacheAllocator size: " 
        //    << freeBlocks.size() + allocBlocks.size()
        //    << " Size in bytes: " << totalSize << std::endl;  
    }

    char *allocate( std::ptrdiff_t numBytes )
    {
        char *result    = 0;
        numBytes        = ( ( numBytes - 1 ) / BlockSize + 1 ) * BlockSize; 

        // search the cache for a free block
        FreeBlocks::iterator freeBlock = freeBlocks.find( numBytes );

        if( freeBlock != freeBlocks.end() )
        {
            //std::cout << "CachedAllocator: found a hit " << numBytes << std::endl;

            result = freeBlock->second;

            freeBlocks.erase( freeBlock );
        }
        else
        {
            // no allocation of the right size exists
            // create a new one with cuda::malloc
            // throw if cuda::malloc can't satisfy the request
            try
            {
                //std::cout << "CachedAllocator: no free block found; calling cudaMalloc " << numBytes << std::endl;

                // allocate memory and convert cuda::pointer to raw pointer
                result = thrust::device_malloc<char>( numBytes ).get();
            }
            catch( std::runtime_error &e )
            {
                // output an error message and exit
                std::cerr << "thrust::device_malloc failed to allocate " << numBytes << " bytes!" << std::endl;
                exit( -1 );
            }
        }

        // insert the allocated pointer into the allocated_blocks map
        allocBlocks.insert( std::make_pair( result, numBytes ) );

        return result;
    }

    void deallocate( char *ptr, size_t n )
    {
        // erase the allocated block from the allocated blocks map
        AllocBlocks::iterator iter  = allocBlocks.find( ptr );
        std::ptrdiff_t numBytes     = iter->second;
               
        allocBlocks.erase(iter);

        // insert the block into the free blocks map
        freeBlocks.insert( std::make_pair( numBytes, ptr ) );
    }
};

// the cache is simply a global variable
CachedAllocator thrustAllocator; 

void thrust_free_all()
{
    thrustAllocator.freeAll(); 
}

///////////////////////////////////////////////////////////////////////////////

void thrust_sort_by_key
(
DevVector<int>::iterator keyBeg, 
DevVector<int>::iterator keyEnd, 
thrust::zip_iterator< 
    thrust::tuple< 
        DevVector<int>::iterator,
        DevVector<Point3>::iterator > > valueBeg
)
{
    thrust::sort_by_key( 
        //thrust::cuda::par( thrustAllocator ),
        keyBeg, keyEnd, valueBeg ); 
}

void thrust_transform_GetMortonNumber
(
DevVector<Point3>::iterator inBeg, 
DevVector<Point3>::iterator inEnd, 
DevVector<int>::iterator    outBeg, 
RealType                    minVal, 
RealType                    maxVal
)
{
    thrust::transform( 
        thrust::cuda::par( thrustAllocator ),
        inBeg, inEnd, outBeg, GetMortonNumber( minVal, maxVal ) ); 
}

// Convert count vector with its map
// Also calculate the sum of input vector
// Input:  [ 4 2 0 5  ]
// Output: [ 4 6 6 11 ] Sum: 11
int makeInPlaceIncMapAndSum
( 
IntDVec& inVec 
)
{
    thrust::inclusive_scan( 
        thrust::cuda::par( thrustAllocator ),
        inVec.begin(), inVec.end(), inVec.begin() );

    const int sum = inVec[ inVec.size() - 1 ];

    return sum;
}

int compactIfNegative
( 
DevVector<int>& inVec 
)
{
    inVec.erase(    
        thrust::remove_if( 
            //thrust::cuda::par( thrustAllocator ),
            inVec.begin(), 
            inVec.end(), IsNegative() ),
        inVec.end() );

    return inVec.size();
}

int compactIfNegative
( 
DevVector<int>& inVec,
DevVector<int>& temp 
)
{
    temp.resize( inVec.size() ); 

    temp.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            inVec.begin(), 
            inVec.end(), 
            temp.begin(), 
            IsNotNegative() ),
        temp.end() );

    inVec.swap( temp ); 

    return (int) inVec.size();
}

void compactBothIfNegative
( 
IntDVec& vec0, 
IntDVec& vec1 
)
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    const IntZipDIter newEnd = 
        thrust::remove_if(  
            //thrust::cuda::par( thrustAllocator ),
            thrust::make_zip_iterator( thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
            IsIntTuple2Negative() );

    const IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

int thrust_copyIf_IsActiveTetra
(
const CharDVec& inVec,
IntDVec&        outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + inVec.size(); 

    outVec.resize( inVec.size() ); 

    outVec.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            first, last, 
            inVec.begin(), 
            outVec.begin(), 
            IsTetActive() ),
        outVec.end()
        ); 

    return outVec.size(); 
}

int thrust_copyIf_Insertable
(
const IntDVec& stencil,
IntDVec&       outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + stencil.size(); 

    outVec.resize( stencil.size() ); 

    outVec.erase( 
        thrust::copy_if(
            thrust::cuda::par( thrustAllocator ),
            first, last, 
            stencil.begin(), 
            outVec.begin(), 
            IsNegative() ),
        outVec.end()
        ); 

    return outVec.size(); 
}

