#!/bin/sh

PYTHON=python3
PYPY=../pypy*/bin/pypy3
GRAAL=../graalvm*/bin/graalpython

run() {
	echo $1 $3
	$2 $3
	echo
}

for f in bm*.py; do
	run 'Python' $PYTHON $f
	run 'PyPy' $PYPY $f
	run 'GraalVM' $GRAAL $f
done

