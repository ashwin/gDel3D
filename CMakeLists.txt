project(gflip3d)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_30,code=sm_30)
include_directories(
    GDelFlipping/Main
    )
cuda_add_executable(
    ${PROJECT_NAME}
    GDelFlipping/src/DelaunayChecker.cpp
    GDelFlipping/src/Demo.cpp
    GDelFlipping/src/InputCreator.cpp
    GDelFlipping/src/RandGen.cpp
    GDelFlipping/src/gDel3D/GpuDelaunay.cu
    GDelFlipping/src/gDel3D/CPU/predicates.cpp
    GDelFlipping/src/gDel3D/CPU/PredWrapper.cpp
    GDelFlipping/src/gDel3D/CPU/Splaying.cpp
    GDelFlipping/src/gDel3D/CPU/Star.cpp
    GDelFlipping/src/gDel3D/GPU/KerDivision.cu
    GDelFlipping/src/gDel3D/GPU/KerPredicates.cu
    GDelFlipping/src/gDel3D/GPU/ThrustWrapper.cu
    )
