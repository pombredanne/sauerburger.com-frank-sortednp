#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pylint --disable=c0415,e0611,e1101,r0903 ${DIR}/../setup.py ${DIR}/benchmark.py sortednp.__init__
