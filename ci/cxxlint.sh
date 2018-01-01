#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 ${DIR}/cpplint.py --filter=-build/include ${DIR}/../*.cpp ${DIR}/../*.h
