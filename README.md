# A Pratical Solver for Scalar Data Topological Simplification

This archive contains the exact code used for the manuscript referenced below.

"A Practical Solver for Scalar Data Topological Simplification"


## Installation Note

Tested on Ubuntu 22.04.3 LTS.


### Install the dependencies

```bash
sudo apt-get install cmake-qt-gui libboost-system-dev libpython3.10-dev libxt-dev libxcursor-dev libopengl-dev
sudo apt-get install qttools5-dev libqt5x11extras5-dev libqt5svg5-dev qtxmlpatterns5-dev-tools 
sudo apt-get install python3-sklearn 
sudo apt-get install libsqlite3-dev 
sudo apt-get install gawk
sudo apt-get install git
```

### Install Paraview

First, go to the root of this repository and run the following commands:
(replace the `4` in `make -j4` by the number of available cores on your system)

```bash
git clone https://github.com/topology-tool-kit/ttk-paraview.git
cd ttk-paraview
git checkout 5.10.1
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPARAVIEW_USE_PYTHON=ON -DPARAVIEW_INSTALL_DEVELOPMENT_FILES=ON -DCMAKE_INSTALL_PREFIX=../install ..
make -j4
make -j4 install
```

Some warnings are expected when using the `make` command, they should not cause any problems.

Stay in the build directory and set the environment variables:
(replace `3.10` in `python3.10` by your version of python)

```bash
PV_PREFIX=`pwd`/../install
export PATH=$PATH:$PV_PREFIX/bin
export LD_LIBRARY_PATH=$PV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PV_PREFIX/lib/python3.10/site-packages
```

### Download Torch

Go in the root of this repository and run the following commands:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
```


### Install TTK

Go in the `ttk-dev` directory then run the following commands:
(replace the `4` in `make -j4` by the number of available cores on your system)

```bash
mkdir build && cd build
paraviewPath=`pwd`/../../ttk-paraview/install/lib/cmake/paraview-5.10
torchPath=`pwd`/../../libtorch/share/cmake/Torch/
cmake -DCMAKE_INSTALL_PREFIX=../install -DParaView_DIR=$paraviewPath -DTorch_DIR=$torchPath ..
make -j4
make -j4 install
```

Stay in the build directory and set the environment variables:
(replace `3.10` in `python3.10` by your version of python)

```bash
TTK_PREFIX=`pwd`/../install
export PV_PLUGIN_PATH=$TTK_PREFIX/bin/plugins/TopologyToolKit
export LD_LIBRARY_PATH=$TTK_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$TTK_PREFIX/lib/python3.10/site-packages
```

### Get the results

Go in the root directory of this repository and extract the data:

```bash
tar xvJf aneurism.tar.xz
```

Create the file that will store the execution results:

```bash
mkdir results
```

To reproduce the results from the tables from the manuscript for the "Aneurysm" dataset, please go to the `scripts` directory and enter the following commands:

```bash
chmod +x runScripts.sh
```

#### Running the experiments

```bash
./runScripts.sh
```

In the results folder, you will find a CSV file named "timePerformanceComparison.csv," showcasing a comparison of time performance between the baseline optimization method and our solver for the simplification setup described in the paper. Additionally, there is another CSV file named "optimizationQualityComparison.csv" presenting a comparison of optimization quality between the baseline method and our solver. Both approaches' optimized data are also available in VTI files within the same folder.
