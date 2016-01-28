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

#pragma once

#include "GPU/CudaWrapper.h"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

struct PerfTimer
{
    float         _freq;
    LARGE_INTEGER _startTime;
    LARGE_INTEGER _stopTime;
    long long _leftover; 
    long long _value; 

    PerfTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        _freq = 1.0f / freq.QuadPart;

        _leftover = 0; 
    }

    void start()
    {
        QueryPerformanceCounter(&_startTime);
    }

    void stop()
    {
        QueryPerformanceCounter(&_stopTime);

        _value      = _leftover + (_stopTime.QuadPart - _startTime.QuadPart); 
        _leftover   = 0; 
    }

    void pause() 
    {
        QueryPerformanceCounter(&_stopTime);
        
        _leftover += (_stopTime.QuadPart - _startTime.QuadPart); 
    }

    double value() const
    {
        return _value * _freq * 1000;
    }
};

#else

#include <sys/time.h>

const long long NANO_PER_SEC = 1000000000LL;
const long long MICRO_TO_NANO = 1000LL;

struct PerfTimer
{
    long long _startTime;
    long long _stopTime;
    long long _leftover; 
    long long _value;

    long long _getTime()
    {
        struct timeval tv;
        long long ntime;

        if (0 == gettimeofday(&tv, NULL))
        {
            ntime  = NANO_PER_SEC;
            ntime *= tv.tv_sec;
            ntime += tv.tv_usec * MICRO_TO_NANO;
        }

        return ntime;
    }

    void start()
    {
        _startTime = _getTime();
    }

    void stop()
    {
        _stopTime = _getTime();
        _value      = _leftover + _stopTime - _startTime; 
        _leftover   = 0; 
    }

    void pause()
    {
        _stopTime   = _getTime();
        _leftover   += _stopTime - _startTime;         
    }

    double value() const
    {
        return ((double) _value) / NANO_PER_SEC * 1000;
    }
};
#endif

class CudaTimer : public PerfTimer
{
public:
    void start()
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::start();
    }

    void stop()
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::stop();
    }

    void pause()
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::pause();
    }

    double value()
    {
        return PerfTimer::value();
    }
};
