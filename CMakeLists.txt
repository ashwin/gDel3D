project(gflip3d)
set(APP_NAME "${PROJECT_NAME}_demo")
set(LIB_NAME_STATIC "${PROJECT_NAME}_static")
set(LIB_NAME "${PROJECT_NAME}")

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

list(APPEND SOURCE_FILES 
    GDelFlipping/src/DelaunayChecker.cpp
    GDelFlipping/src/InputCreator.cpp
    GDelFlipping/src/RandGen.cpp
    GDelFlipping/src/gDel3D/GpuDelaunay.cu
    GDelFlipping/src/gDel3D/CPU/predicates.cpp
    GDelFlipping/src/gDel3D/CPU/PredWrapper.cpp
    GDelFlipping/src/gDel3D/CPU/Splaying.cpp
    GDelFlipping/src/gDel3D/CPU/Star.cpp
    GDelFlipping/src/gDel3D/GPU/KerDivision.cu
    GDelFlipping/src/gDel3D/GPU/KerPredicates.cu
    GDelFlipping/src/gDel3D/GPU/ThrustWrapper.cu)

list(APPEND HEADER_FILES_CPU
    GDelFlipping/src/gDel3D/CPU/CPUDecl.h
    GDelFlipping/src/gDel3D/CPU/predicates.h
    GDelFlipping/src/gDel3D/CPU/PredWrapper.h
    GDelFlipping/src/gDel3D/CPU/Splaying.h
    GDelFlipping/src/gDel3D/CPU/Star.h)

list(APPEND HEADER_FILES_GPU
    GDelFlipping/src/gDel3D/GPU/CudaWrapper.h
    GDelFlipping/src/gDel3D/GPU/GPUDecl.h
    GDelFlipping/src/gDel3D/GPU/HostToKernel.h
    GDelFlipping/src/gDel3D/GPU/KerCommon.h
    GDelFlipping/src/gDel3D/GPU/KerDivision.h
    GDelFlipping/src/gDel3D/GPU/KerPredicates.h
    GDelFlipping/src/gDel3D/GPU/KerSchewchuk.h
    GDelFlipping/src/gDel3D/GPU/ThrustWrapper.h)

list(APPEND HEADER_FILES_GDEL3D
    GDelFlipping/src/gDel3D/CommonTypes.h
    GDelFlipping/src/gDel3D/GpuDelaunay.h
    GDelFlipping/src/gDel3D/PerfTimer.h)

list(APPEND HEADER_FILES_COMMON
    GDelFlipping/src/DelaunayChecker.h
    GDelFlipping/src/InputCreator.h
    GDelFlipping/src/RandGen.h)

# build the app    
cuda_add_executable(${APP_NAME} ${SOURCE_FILES} GDelFlipping/src/Demo.cpp)
set_target_properties(${APP_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    

# build the static library    
cuda_add_library(${LIB_NAME_STATIC} STATIC ${SOURCE_FILES})
set_target_properties(${LIB_NAME_STATIC} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# build the shared library    
# cuda_add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
# set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#########################################################    
install(TARGETS ${APP_NAME} DESTINATION bin)
install(TARGETS ${LIB_NAME_STATIC} DESTINATION lib)
# install(TARGETS ${LIB_NAME} DESTINATION lib)

install(FILES ${HEADER_FILES_COMMON} DESTINATION include)
install(FILES ${HEADER_FILES_GDEL3D} DESTINATION include/gDel3D)
install(FILES ${HEADER_FILES_CPU} DESTINATION include/gDel3D/CPU)
install(FILES ${HEADER_FILES_GPU} DESTINATION include/gDel3D/GPU)
#########################################################
    
  
