#!/bin/sh

python3 -m venv .venv --without-pip
source .venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
.venv/bin/pip3 install -r ./requirments.txt --use-feature=2020-resolver
