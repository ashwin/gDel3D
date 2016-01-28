gDel3D
======

This program constructs the Delaunay Triangulation of a set of points in 3D 
using the GPU. The algorithm used is a combination of incremental insertion, 
flipping and star splaying. The code is written using CUDA programming model 
of NVIDIA. 

Documentation
=============

Please read our [I3D 2014 paper](gdel3d_paper.pdf) for details of the gDel3D algorithm, its
results and performance.

Setup
=====

gDel3D works on any NVIDIA GPU with hardware capability 1.1 onward. However, 
it works best on Fermi and higher architecture. The code has been tested on 
the NVIDIA GTX 450, GTX 460, GTX 470, GTX580 (using sm_20) on Windows OS; 
and GTX Titan on Linux (using sm_30). 

To switch from double to single precision, simply define REAL_TYPE_FP32. 

For more details on the input and output, refer to: 
	CommonTypes.h 	(near the end)
	Demo.cpp 
	DelaunayChecker.cpp. 

Build
=====

A Visual Studio 2012 project is provided for Windows user. 

CMake is used to build gDel3D on Linux, as shown here:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

Note that by default, CMake generate code for sm_30 and sm_35. Please modify 
the CMakeList.txt if needed. 

License
=======

Authors: Cao Thanh Tung and Ashwin Nanjappa

Project: gDel3D

Copyright (c) 2014, School of Computing, National University of Singapore. 
All rights reserved.

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
