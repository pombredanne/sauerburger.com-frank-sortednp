#!/bin/bash

if [ -z "${PY_VERSION}" ]; then
	echo "Error: Set PY_VERSION before running this script!"
	echo " e.g. export PY_VERSION='cp34-cp34m'"
	exit 1
fi

PYTHON=/opt/python/${PY_VERSION}/bin/python3
GTEST="/usr/local/include/gtest"

/opt/python/${PY_VERSION}/bin/pip install -r requirements.txt
NUMPY=$(${PYTHON} -c "import numpy; print(numpy.get_include())")

# compile
g++ -c cxxtest.cpp -I${NUMPY} $(${PYTHON}-config --cflags) -I${GTEST} -Wno-conversion-null -o cxxtest.o

# link
unzip -n wheelhouse/sortednp-*-${PY_VERSION}-*.whl
g++  cxxtest.o $(${PYTHON}-config --ldflags) -lgtest _sortednp*.so -o cxxtest

# run
./cxxtest
