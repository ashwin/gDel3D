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

#include "../CommonTypes.h"
#include "HostToKernel.h"
#include "DPredWrapper.h"

#define PRED_THREADS_PER_BLOCK	32

void setPredWrapperConstant( const DPredWrapper &hostPredWrapper );

__global__ void kerInitPointLocationFast
(
int*    vertTetArr, 
Tet     tet,
int     tetIdx
)
;
__global__ void kerInitPointLocationExact
(
int*    vertTetArr, 
Tet     tet,
int     tetIdx
)
;
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
;
__global__ void
kerSplitPointsFast
(
KerIntArray vertexArr,
int*        vertexTetArr,
int*        tetToVert,
Tet*        tetArr,
char*       tetInfoArr,
KerIntArray freeArr
)
;
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
;
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
;
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
;
__global__ void
kerCheckDelaunayExact_Exact
(
int*            actTetVec,
Tet*            tetArr,
TetOpp*         oppArr,
char*           tetInfoArr,
int*            tetVoteArr,
int*            voteArr,
int2*           exactCheckVi,
int*            counterArr,
int             voteOffset
)
;
__global__ void kerInitPredicate
( 
RealType* 
)
;
__global__ void
kerRelocatePointsFast
(
KerIntArray     vertexArr,
int*            vertexTetArr,
int*            tetToFlip,
FlipItem*       flipArr
)
;
__global__ void
kerRelocatePointsExact
(
KerIntArray     vertexArr,
int*            vertexTetArr,
int*            tetToFlip,
FlipItem*       flipArr
)
;