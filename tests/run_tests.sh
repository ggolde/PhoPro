#!/usr/bin/env bash
set -euo pipefail

mkdir -p test-results
python -m pytest . -vv \
  --junitxml=test-results/junit.xml | tee test-results/pytest.log