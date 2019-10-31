#!/bin/bash

set -o pipefail

mkdir ad-reports 2> /dev/null

if [ -z "$1" ]; then
    DIRNAME="$(mktemp -d "$PWD/ad-reports/report-$(ls -1 ad-reports | wc -l)-$(date +'%b%d')-XXXX")"
else
    DIRNAME="$(mktemp -d "$PWD/ad-reports/report-$1-$(ls -1 ad-reports | wc -l)-$(date +'%b%d')-XXXX")"
fi

echo "$DIRNAME"

git submodule update --recursive --init
git rev-parse HEAD > "$DIRNAME/git-head"
git diff > "$DIRNAME/git-diff"
git log -1000 "--pretty=format:%h %ci %<(22) %an %s" > "$DIRNAME/git-log"

echo
echo "Building..."
(cd build && cmake .. > "$DIRNAME/cmake-out" 2>&1 && make -j20 > "$DIRNAME/make-out" 2>&1) || (echo "Cannot build"; exit 1)

echo
echo "Running ZE test"
(./withtvm ./build python3 -u ./tests/python/unittest/test_pass_zero_elimination.py > "$DIRNAME/log-ze" 2>&1) || echo "FAILED!"

echo
echo "Running AD test"
(./withtvm ./build python3 -u ./tests/python/unittest/test_pass_autodiff.py > "$DIRNAME/log-ad" 2>&1) || echo "FAILED!"

grep WARN "$DIRNAME/log-ad" > "$DIRNAME/warn-ad"

echo
echo "Running AD test in verbose unfailing mode"
(./withtvm ./build python3 -u ./tests/python/unittest/test_pass_autodiff.py -v --no-perf --no-numgrad > "$DIRNAME/log-ad-verbose" 2>&1) || echo "FAILED!"

echo
echo "Running Relay integration test (primal grads)"
(./withtvm ./build python3 -u tests/python/relay/test_primal_gradients.py > "$DIRNAME/log-relay" 2>&1) || echo "FAILED!"

echo
echo "$DIRNAME"
