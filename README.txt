This program constructs the Delaunay Triangulation of a set of points in 3D 
using the GPU. The algorithm used is a combination of incremental insertion, 
flipping and star splaying. The code is written using CUDA programming model 
of NVIDIA. 

===================
Programming authors
===================

Cao Thanh Tung
Ashwin Nanjappa

=====
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

=====
Compilation
=====

A Visual Studio 2012 project is provided for Windows user. 

CMake is used to build gDel3D on Linux, as shown here:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

Note that by default, CMake generate code for sm_30 and sm_35. Please modify 
the CMakeList.txt if needed. 