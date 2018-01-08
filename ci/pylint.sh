#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pylint --disable=e0611,e1101,r0903 setup.py ${DIR}/benchmark.py ${DIR}/../src/sortednp/*.py
