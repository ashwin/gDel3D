#!/bin/bash

build-dir="build-release"
mkdir ${build-dir}

pushd ${build-dir}

cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/devsystem/dependencies/ubuntu16/cuda/cuda-8.0/ -DCMAKE_POSITION_INDEPENDENT_CODE=True --DCMAKE_INSTALL_PREFIX=/devsystem/dependencies/ubuntu16/gflip3d/gflip3d-1.0/ ..

make -j
make install

popd

