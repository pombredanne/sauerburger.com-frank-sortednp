#!/bin/bash

if [ "$QUICK" == "true" ]; then
	quick_arg="--quick"
else
	quick_arg=""
fi

function wait_or_fail {
	# If the second job fails first, the script still waits for the first one to
	# finish before returning an error.
	for job in `jobs -p`; do
	    wait $job || exit 1
	done
}


# Run two benchmarks in parallel
python3 ci/benchmark.py ${quick_arg} --intersect-simple &
python3 ci/benchmark.py ${quick_arg} --intersect-binary &
wait_or_fail

# Run two benchmarks in parallel
python3 ci/benchmark.py ${quick_arg} --intersect-galloping &
python3 ci/benchmark.py ${quick_arg} --merge &
wait_or_fail
