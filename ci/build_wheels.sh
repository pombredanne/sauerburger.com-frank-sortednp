#!/bin/bash

if [ -z "${PYTHON_VERSIONS}" ]; then
	echo "Error: Set PYTHON_VERSIONS before running this script!"
	echo " e.g. export PYTHON_VERSIONS='cp34-cp34m cp35-cp35m cp36-cp36m'"
	exit 1
fi

for PY_VERSION in "${PYTHON_VERSIONS}"; do
	/opt/python/${PY_VERSION}/bin/pip install -r requirements.txt
	/opt/python/${PY_VERSION}/bin/pip wheel -w wheelhouse/ .
done
