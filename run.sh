#!/bin/bash

BUILD_TREE="build-release"
if [ -d ${BUILD_TREE} ]; then
	rm -rf ${BUILD_TREE}
fi
	
mkdir -p ${BUILD_TREE}
pushd ${BUILD_TREE}

cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR="/devsystem/dependencies/ubuntu16/cuda/cuda-8.0/" -DCMAKE_POSITION_INDEPENDENT_CODE=True --DCMAKE_INSTALL_PREFIX="/devsystem/dependencies/ubuntu16/gflip3d/gflip3d-1.0/" ..

make -j
make install

popd

