# Bayesian Optimization with Automatic Prior Selection for Data-Efficient Direct Policy Search

#### Meta-repo for code implementing the MLEI acquisition function

Paper: "Bayesian Optimization with Automatic Prior Selection for Data-Efficient Direct Policy Search", submitted to the International Conference on Robotics and Automation (ICRA) 2018.

Authors: Rémi Pautrat, Konstantinos Chatzilygeroudis, and Jean-Baptiste Mouret.


## How to use it

#### How to properly clone this repo

```
git clone --recursive https://github.com/resibots/pautrat_2018_mlei.git
```


#### Dependencies

- [Boost]: C++ Template Libraries (http://www.boost.org)
- [Eigen]: Linear Algebra C++ Library (http://eigen.tuxfamily.org/)
- [realpath]: `sudo apt-get install realpath` (http://manpages.ubuntu.com/manpages/jaunty/man1/realpath.1.html)


#### How to easily compile everything

**Important:** Make sure you have installed all the dependencies of each repo. Otherwise the build will fail.

From the root of this repo run:

```
sh compile_all.sh
```

**Note:** You have to set the two environment variables `RESIBOTS_DIR` and `LD_LIBRARY_PATH` before each use of this code, so you may want to add the two `export` commands to your .bashrc file:

```
export RESIBOTS_DIR=[PATH_OF_YOUR_ROOT_DIRECTORY]/install
```

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[PATH_OF_YOUR_ROOT_DIRECTORY]/install/dart_path/lib
```


#### How to run MAP-Elites experiments to generate priors

```
cd sferes2/build/exp/map_elites_hexapod/
./hexa_duty
```

The best results were obtained with a 6D descriptor, the body orientation, but other descriptors are available in hexapod_simu/hexapod_dart/include/hexapod_dart/descriptors.hpp.


#### How to use Bayesian optimization with selection of priors in a limbo experiment

```
cd limbo/build/exp/bo_mlei
./bo_mlei_flat_ground [directory_containing_your_priors] [options]
```
bo_mlei_flat_ground launches the simulation on flat ground, but you can use the other environments as well with bo_mlei_easy_stairs, bo_mlei_medium_stairs and bo_mlei_hard_stairs. Use bo_mlei_robot to use it on a real robot.

Several options are available:
- `-l`: id of the leg you want to remove (-1: no broken leg, 0 and 5: left and right rear legs, 1 and 4: left and right middle legs, 2 and 3: left and right front legs) (default: -1).
- `-s`: method used of prior selection (0: random, 1: constant prior, 2: MLEI, default: MLEI).
- `-t` : number of types of different priors that you use (default: 4, e.g. for flat ground, easy stairs, medium stairs and hard stairs priors).
- `-p` : number of priors used for each type of prior (default: 15).
- `-n` : number of iterations for Bayesian optimization (default: 10).
- `-c` : controller of the robot given by its 54 parameters (only for replaying one gait, there is no learning here).


#### How to easily clean everything

From the root of this repo run:

```
sh clear_all.sh
```

## LICENSE

[CeCILL]
