#!/bin/bash
# Gets files
git clone git@github.com:networkx/networkx-metis.git
# Runs install script
cd networkx-metis
python3 setup.py install --prefix=.
