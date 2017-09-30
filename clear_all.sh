#!/bin/bash

cd dart/build
make clean
cd ..
rm -fr build/

cd ../sferes2/
./waf clean
cd exp/
rm -r map_elites_hexapod
cd ..
git checkout -- .
git clean -df
cd ..

cd hexapod_common/hexapod_controller
./waf clean
cd ..
git checkout -- .
git clean -df
cd ..

cd hexapod_simu/hexapod_dart
./waf clean
cd ..
git checkout -- .
git clean -df
cd ..

cd limbo
./waf clean
rm -r exp/
git checkout -- .
git clean -df
cd ..

rm -r install

git checkout -- .
git clean -df
