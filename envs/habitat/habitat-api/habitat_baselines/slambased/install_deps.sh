#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DIR1=$(pwd)
MAINDIR=$(pwd)/3rdparty
mkdir ${MAINDIR}
cd ${MAINDIR}
#conda create -y -n "HandcraftedAgents" python=3.6
source activate HandcraftedAgents
conda install opencv -y
conda install pytorch torchvision -c pytorch -y
conda install -c conda-forge imageio -y
conda install ffmpeg -c conda-forge -y
cd ${MAINDIR}
mkdir eigen3
cd eigen3
wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz
tar -xzf 3.3.5.tar.gz
cd eigen-eigen-b3f3d4950030
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${MAINDIR}/eigen3_installed/
make install
cd ${MAINDIR}
wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip
unzip glew-2.1.0.zip
cd glew-2.1.0/
cd build
cmake ./cmake  -DCMAKE_INSTALL_PREFIX=${MAINDIR}/glew_installed
make -j4
make install
cd ${MAINDIR}
#pip install numpy --upgrade
rm Pangolin -rf
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=${MAINDIR}/glew_installed/ -DCMAKE_LIBRARY_PATH=${MAINDIR}/glew_installed/lib/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/pangolin_installed
cmake --build .
cd ${MAINDIR}
rm ORB_SLAM2 -rf
rm ORB_SLAM2-PythonBindings -rf
git clone https://github.com/ducha-aiki/ORB_SLAM2
git clone https://github.com/ducha-aiki/ORB_SLAM2-PythonBindings
cd ${MAINDIR}/ORB_SLAM2
sed -i "s,cmake .. -DCMAKE_BUILD_TYPE=Release,cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=${MAINDIR}/eigen3_installed/include/eigen3/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/ORBSLAM2_installed ,g" build.sh
ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen ${MAINDIR}/ORB_SLAM2/Thirdparty/g2o/g2o/core/Eigen
./build.sh
cd build
make install
cd ${MAINDIR}
cd ORB_SLAM2-PythonBindings/src
ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen Eigen
cd ${MAINDIR}/ORB_SLAM2-PythonBindings
mkdir build
cd build
CONDA_DIR=$(dirname $(dirname $(which conda)))
sed -i "s,lib/python3.5/dist-packages,${CONDA_DIR}/envs/HandcraftedAgents/lib/python3.6/site-packages/,g" ../CMakeLists.txt
cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.6m.so -DPYTHON_EXECUTABLE:FILEPATH=`which python` -DCMAKE_LIBRARY_PATH=${MAINDIR}/ORBSLAM2_installed/lib -DCMAKE_INCLUDE_PATH=${MAINDIR}/ORBSLAM2_installed/include;${MAINDIR}/eigen3_installed/include/eigen3 -DCMAKE_INSTALL_PREFIX=${MAINDIR}/pyorbslam2_installed 
make
make install
cp ${MAINDIR}/ORB_SLAM2/Vocabulary/ORBvoc.txt ${DIR1}/data/
