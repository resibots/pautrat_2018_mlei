#!/bin/bash
mkdir -p limbo/exp
mkdir install
cd limbo/exp
ln -s ../../experiments/bo_mlei
cd ../../sferes2/exp
ln -s ../../experiments/map_elites_hexapod

cd ../../

INSTALL="$(realpath ./install)"
DART_PATH=$INSTALL/dart_path/
echo "Install directory: ${INSTALL}"
echo "Dart path: ${DART_PATH}"
export RESIBOTS_DIR=$INSTALL

# compile dart
cd dart
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$DART_PATH ..
make -j8
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DART_PATH/lib

# compile hexapod_common
cd ../../hexapod_common/hexapod_controller
./waf configure --prefix=$INSTALL
./waf
./waf install

cd ../hexapod_models
./waf configure --prefix=$INSTALL
./waf
./waf install

# compile hexapod_simu
cd ../../hexapod_simu/hexapod_dart
LINKFLAGS="-L$DART_PATH/lib -ldart -ldart-utils -ldart-utils-urdf" ./waf configure --prefix=$INSTALL --dart=$DART_PATH
./waf
./waf install

# compile sferes
cd ../../sferes2
./waf configure --cpp11=yes
./waf

# compile map_elites_hexapod
LINKFLAGS="-L$DART_PATH/lib -ldart -ldart-utils -ldart-utils-urdf" ./waf configure --cpp11=yes --dart=$DART_PATH --exp map_elites_hexapod
./waf --exp map_elites_hexapod

# compile limbo
cd ../limbo
./waf configure --exp bo_mlei
./waf --exp bo_mlei

cd ..
