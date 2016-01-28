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

#include "../CommonTypes.h"
#include "HostToKernel.h"

__global__ void
kerMakeFirstTetra
(
Tet*    tetArr,
TetOpp* oppArr,
char*   tetInfoArr, 
Tet     tet,
int     tetIdx,
int     infIdx
)
;
__global__ void
kerSplitTetra
(
KerIntArray newVertVec,
KerIntArray insVertVec,
int*        vertArr, 
int*        vertTetArr,
int*        tetToVert,
Tet*        tetArr,
TetOpp*     oppArr,
char*       tetInfoArr,
int*        freeArr,
int*        vertFreeArr,
int         infIdx
)
;
__global__ void
kerMarkRejectedFlips
(
KerIntArray actTetArr,
TetOpp*     oppArr,
int*        tetVoteArr,
char*       tetInfoArr,
int*        voteArr,
int*        flipToTet,
int*        counterArr,
int         voteOffset
)
;
__global__ void
kerPickWinnerPoint
(
KerIntArray  vertexArr,
int*         vertexTetArr,
int*         vertSphereArr,
int*         tetSphereArr,
int*         tetVertArr
)
;
__global__ void
kerFlip
(
KerIntArray flipToTet,
Tet*        tetArr,
TetOpp*     oppArr,
char*       tetInfoArr,
int2*       tetMsgArr,
FlipItem*   flipArr,
int*        flip23NewSlot, 
int*        vertFreeArr, 
int*        freeArr,
int*        actTetArr,
KerIntArray insVertVec,
int         infIdx,
int         orgFlipNum
);
__global__ void
kerUpdateOpp
(
FlipItem*    flipVec,
TetOpp*      oppArr,
int2*        tetMsgArr,
int*         encodedFaceViArr,
int          orgFlipNum,
int          flipNum
); 
__global__ void
kerGatherFailedVerts
(
KerTetArray tetArr,
TetOpp*     tetOppArr,
int*        failVertArr,
int*        vertTetArr
)
;
__global__ void 
kerUpdateFlipTrace
(
FlipItem*   flipArr, 
int*        tetToFlip,
int         orgFlipNum, 
int         flipNum
)
;
__global__ void
kerMarkTetEmpty
(
KerCharArray tetInfoVec
)
;
__global__ void 
kerUpdateVertIdx
(
KerTetArray     tetVec,
int*            orgPointIdx
)
;
__global__ void 
kerMakeReverseMap
(
KerIntArray insVertVec, 
int*        scatterArr, 
int*        revMapArr,
int         num
)
;
__global__ void 
kerUpdateTetIdx
(
KerIntArray idxArr, 
int*        orderArr, 
int         oldInfBlockIdx, 
int         newInfBlockIdx
)
;
__global__ void 
kerUpdateBlockOppTetIdx
(
TetOpp* oppVec, 
int*    orderArr, 
int     oldInfBlockIdx, 
int     newInfBlockIdx,
int     oldTetNum
) 
;
__global__ void 
kerMarkSpecialTets
(
KerCharArray tetInfoVec, 
TetOpp*      oppArr
)
;
__global__ void 
kerUpdateVertFreeList
(
KerIntArray insTetVec, 
int*        vertFreeArr, 
int*        freeArr, 
int         startFreeIdx
)
;
__global__ void
kerNegateInsertedVerts
(
KerIntArray vertTetVec, 
int*        tetToVert
)
;
__global__ void 
kerAllocateFlip23Slot
(
KerIntArray flipToTet, 
Tet*        tetArr, 
int*        vertFreeArr, 
int*        freeArr, 
int*        flip23NewSlot,
int         infIdx,
int         tetNum
)
;
__global__ void 
kerShiftInfFreeIdx
(
int*    vertFreeArr, 
int*    freeArr, 
int     infIdx, 
int     start, 
int     shift
)
;
__global__ void 
kerUpdateBlockVertFreeList
(
KerIntArray insTetVec, 
int*        vertFreeArr, 
int*        freeArr, 
int*        scatterMap,
int         oldInsNum
)
;
__global__ void 
kerShiftOppTetIdx
(
TetOpp* oppArr, 
int     tetNum,
int     start,
int     shift
) 
;
__global__ void 
kerShiftTetIdx
(
KerIntArray idxVec, 
int         start,
int         shift
) 
;
__global__ void 
kerCollectFreeSlots
(
char* tetInfoArr, 
int*  prefixArr,
int*  freeArr,
int   newTetNum
)
;
__global__ void
kerMakeCompactMap
(
KerCharArray tetInfoVec, 
int*         prefixArr, 
int*         freeArr, 
int          newTetNum
)
;
__global__ void
kerCompactTets
(
KerCharArray tetInfoVec, 
int*         prefixArr, 
Tet*         tetArr, 
TetOpp*      oppArr, 
int          newTetNum
)
;