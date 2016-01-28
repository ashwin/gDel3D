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

#include "InputCreator.h"

#include <cmath>

const int GridSize = 512;

typedef std::set< Point3 >  Point3Set;

void InputCreator::randSpherePoint
( 
RealType    radius, 
RealType&   x, 
RealType&   y, 
RealType&   z 
) 
{
    RealType a, b, c, d, l; 

    do { 
        a = _randGen.getNext() * 2.0 - 1.0; 
        b = _randGen.getNext() * 2.0 - 1.0; 
        c = _randGen.getNext() * 2.0 - 1.0; 
        d = _randGen.getNext() * 2.0 - 1.0; 

        l = a * a + b * b + c * c + d * d; 

    } while ( l >= 1.0 ); 

    x = 2.0 * ( b * d + a * c ) / l * radius; 
    y = 2.0 * ( c * d - a * b ) / l * radius; 
    z = ( a * a + d * d - b * b - c * c ) / l * radius; 
}

void InputCreator::makePoints
( 
int             pointNum, 
Distribution    dist,
Point3HVec&     pointVec,
int             seed
)
{
    const RealType PI = 3.141592654; 

    assert( pointVec.empty() );

    Point3Set pointSet;

    ////
    // Initialize seed
    ////
    _randGen.init( seed, 0.0, 1.0 );

    ////
    // Generate points
    ////

    RealType x = 0.0;
    RealType y = 0.0;
    RealType z = 0.0;

    for ( int i = 0; i < pointNum; ++i )
    {
        bool uniquePoint = false;

        // Loop until point is unique
        while ( !uniquePoint )
        {
            // Generate a point, coordinates between [0,1)
            switch ( dist )
            {
            case UniformDistribution:
                {
                    x = _randGen.getNext();
                    y = _randGen.getNext();
                    z = _randGen.getNext();
                }
                break;

            case GaussianDistribution:
                {
                    _randGen.nextGaussian( x, y, z);
                }
                break;

            case BallDistribution:
                {
                    RealType d;

                    do
                    {
                        x = _randGen.getNext() - 0.5; 
                        y = _randGen.getNext() - 0.5; 
                        z = _randGen.getNext() - 0.5; 

                        d = x * x + y * y + z * z;

                    } while ( d > 0.45 * 0.45 );

                    x += 0.5;
                    y += 0.5;
                    z += 0.5;
                }
                break;

            case SphereDistribution:
                {
                    randSpherePoint( 0.45, x, y, z ); 

                    x += 0.5;
                    y += 0.5;
                    z += 0.5;
                }
                break;

            case GridDistribution:
                {
                    RealType v[3];

                    for ( int i = 0; i < 3; ++i )
                    {
                        const RealType val  = _randGen.getNext() * GridSize;
                        const RealType frac = val - floor( val );
                        v[ i ]              = ( frac < 0.5f ) ? floor( val ) : ceil( val );
                        v[ i ]             /= GridSize; 
                    }

                    x = v[0];
                    y = v[1];
                    z = v[2];
                }
                break;

            case ThinSphereDistribution: 
                {
                    RealType d, a, b; 

                    d = _randGen.getNext() * 0.001; 
                    a = _randGen.getNext() * 3.141592654 * 2; 
                    b = _randGen.getNext() * 3.141592654; 

                    x = ( 0.45 + d ) * sin( b ) * cos( a ); 
                    y = ( 0.45 + d ) * sin( b ) * sin( a ); 
                    z = ( 0.45 + d ) * cos( b ); 

                    x += 0.5;
                    y += 0.5;
                    z += 0.5;
                }
                break; 
            }

            const Point3 point = { x, y, z };

            if ( pointSet.end() == pointSet.find( point ) )
            {
                pointSet.insert( point );
                
                pointVec.push_back( point );

                uniquePoint = true;
            }
        }
    }

    return;
}

int InputCreator::readPoints
( 
std::string inFilename, 
Point3HVec& pointVec
)
{
    bool isBinary = ( 0 == inFilename.substr( inFilename.length() - 4, 4 ).compare( ".bin" ) ); 

    Point3HVec inPointVec;
    std::ifstream inFile;

    if ( isBinary ) 
    {
        std::cout << "Binary input file!" << std::endl; 

        inFile.open( inFilename.c_str(), std::ios::binary );
    }
    else
    {
        inFile.open( inFilename.c_str() );
    }

    if ( !inFile.is_open() )
    {
        std::cout << "Error opening input file: " << inFilename << " !!!" << std::endl;
        exit( 1 );
    }
    else
    {
        std::cout << "Reading from point file ..." << std::endl;
    }

    if ( isBinary ) 
    {
        // Get file size
        inFile.seekg( 0, inFile.end ); 

        const int fileSize = inFile.tellg(); 

        inFile.seekg( 0, inFile.beg ); 

        // Read pointNum
        int pointNum = 0;
        inFile.read( ( char* ) &pointNum, sizeof( pointNum ) );

        // Detect whether numbers are in float or double
        const int bufferSize = fileSize - sizeof( int ); 

        if ( 0 != bufferSize % pointNum ) 
        {
            std::cout << "Invalid input file format! Wrong file size" << std::endl; 
            exit( -1 ); 
        }

        if ( bufferSize / pointNum / 3 == 4 ) 
        {
            // Float
            float* buffer = new float[ pointNum * 3 ]; 

            inFile.read( ( char * ) buffer, pointNum * 3 * sizeof( float ) );

            for ( int i = 0; i < pointNum * 3; i += 3 ) 
            {
                Point3 point = { buffer[ i ], buffer[ i + 1 ], buffer[ i + 2 ] }; 

                inPointVec.push_back( point ); 
            }

            delete [] buffer; 
        } 
        else if ( bufferSize / pointNum / 3 == 8 ) 
        {
            // Double
            double* buffer = new double[ pointNum * 3 ]; 

            inFile.read( ( char * ) buffer, pointNum * 3 * sizeof( double ) );

            for ( int i = 0; i < pointNum * 3; i += 3 ) 
            {
                Point3 point = { buffer[ i ], buffer[ i + 1 ], buffer[ i + 2 ] }; 

                inPointVec.push_back( point ); 
            }

            delete [] buffer; 
        }
        else
        {
            std::cout << "Unknown input number format! Size = " 
                << bufferSize / pointNum / 3 << std::endl; 
            exit( -1 ); 
        }
    } else
    {
        std::string strVal;
        Point3 point;
        int idx         = 0;
        RealType val    = 0.0;

        while ( inFile >> strVal )
        {
            std::istringstream iss( strVal );

            // Read a coordinate
            iss >> val;
            point._p[ idx ] = val;
            ++idx;

            // Read a point
            if ( 3 == idx )
            {
                idx = 0;
                inPointVec.push_back( point );
            }
        }
    }

    std::cout << "Number of points: " << inPointVec.size() << std::endl; 

    Point3Set pointSet;

    ////
    // Remove duplicates
    ////
    pointSet.clear();

    // Iterate input points
    for ( int ip = 0; ip < ( int ) inPointVec.size(); ++ip )
    {
        Point3& inPt = inPointVec[ ip ];

        // Check if point unique
        if ( pointSet.end() == pointSet.find( inPt ) )
        {
            pointSet.insert( inPt );
            pointVec.push_back( inPt );
        }
    }

    const int dupCount = inPointVec.size() - ( int ) pointVec.size();

    if ( dupCount > 0 )
    {
        std::cout << dupCount << " duplicate points in input file!" << std::endl;
    }

    return pointVec.size();
}  
