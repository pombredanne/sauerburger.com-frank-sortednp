#!/bin/bash

if [ -z "${PY_VERSION}" ]; then
	echo "Error: Set PY_VERSION before running this script!"
	echo " e.g. export PY_VERSION='cp34-cp34m'"
	exit 1
fi

/opt/python/${PY_VERSION}/bin/pip install wheel wheelhouse/sortednp-*-${PY_VERSION}-*.whl

/opt/python/${PY_VERSION}/bin/pip install nose
/opt/python/${PY_VERSION}/bin/nosetests -P sortednp.tests
