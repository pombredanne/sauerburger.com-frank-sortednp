
PYTHON=python3
GTEST="-I/usr/local/include/gtest"
FLAGS="-Wno-conversion-null"
NUMPY=-I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
CXX=g++

install:
	$(PYTHON) setup.py install

test: cxx-based-test python-based-test
lint: cxx-lint python-lint

cxx-lint: *.cpp *.h
	python3 cpplint.py --filter=-build/include $^
	
	
python-lint: setup.py benchmark.py sortednp/*.py
	pylint --disable=e0611,e1101,r0903 $^

python-based-test: sortednpmodule.cpp setup.py tests/*.py sortednp/*.py
	$(PYTHON) setup.py test

cxx-based-test: cxxtest
	./$<

cxxtest: cxxtest.o sortednpmodule.o
	$(CXX) $^ `$(PYTHON)-config --ldflags` -lgtest -o $@

%.o: %.cpp $(wildcard *.h)
	$(CXX) -c $< $(NUMPY) `$(PYTHON)-config --cflags` $(GTEST) $(FLAGS) -o $@
	
clean:
	rm -f cxxtest.o cxxtest sortednpmodule.o


install-gtest: googletest
	cd googletest/googletest && \
	export GTEST_DIR=$$PWD && \
	$(CXX) -isystem $${GTEST_DIR}/include -fpic -I$${GTEST_DIR} \
	    -pthread -c $${GTEST_DIR}/src/gtest-all.cc -o libgtest.o && \
	$(CXX) -shared -o $${GTEST_DIR}/libgtest.so $${GTEST_DIR}/libgtest.o && \
	rm -f /usr/local/lib/libgtest.so && \
	mv $${GTEST_DIR}/libgtest.so /usr/local/lib/libgtest.so && \
	rm -rf /usr/local/include/gtest && \
	mv $${GTEST_DIR}/include/gtest /usr/local/include/gtest && \
	ldconfig

googletest:
	git clone https://github.com/google/googletest.git


benchmark: bm-intersect-binary bm-intersect-simple \
	bm-intersect-galloping bm-merge

bm-%:
	python3 benchmark.py $(QUICK) --$*
