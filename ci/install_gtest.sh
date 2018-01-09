#!/bin/bash

git clone https://github.com/google/googletest.git /googletest
cd /googletest/googletest
export GTEST_DIR=/googletest/googletest
g++ -isystem ${GTEST_DIR}/include -fpic -I${GTEST_DIR} \
	-pthread -c ${GTEST_DIR}/src/gtest-all.cc -o libgtest.o 
g++ -shared -o ${GTEST_DIR}/libgtest.so ${GTEST_DIR}/libgtest.o 
rm -f /usr/local/lib/libgtest.so 
rm -rf /usr/local/include/gtest 
mv ${GTEST_DIR}/libgtest.so /usr/local/lib/libgtest.so 
mv ${GTEST_DIR}/include/gtest /usr/local/include/gtest
ldconfig
