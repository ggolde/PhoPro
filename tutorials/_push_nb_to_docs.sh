#!/usr/bin/env bash
set -e

cd /Users/ggolde/home/dev/PhoPro/

jupyter nbconvert \
  --to markdown \
  --output-dir docs/tutorials \
  "tutorials/*.ipynb"