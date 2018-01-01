#!/bin/bash

PYTHON=python3
NUMPY=$(${PYTHON} -c "import numpy; print(numpy.get_include())")
GTEST="/usr/local/include/gtest"


# compile
g++ -c cxxtest.cpp -I${NUMPY} $(${PYTHON}-config --cflags) -I${GTEST} -Wno-conversion-null -o cxxtest.o
python3 setup.py build

# link
g++  cxxtest.o $(${PYTHON}-config --ldflags) -lgtest build/lib.*/_sortednp-*.so -o cxxtest

# run
./cxxtest
