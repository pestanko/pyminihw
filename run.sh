#! /bin/bash

ARTIFACTS=$1

if [ -z $ARTIFACTS ]; then
    ARTIFACTS=$(mktemp -d -t minihw-XXXXXXXXXX)
fi

if [ ! -d $ARTIFACTS ]; then
    mkdir -p $ARTIFACTS
fi

pip3 install junitparser

if [ -f './minihw.py' ]; then
    rm './minihw.py'
fi

wget https://raw.githubusercontent.com/pestanko/pyminihw/master/minihw.py

python3 -m minihw -Ldebug execute -A $ARTIFACTS -C -B -T solution --build-type=gcc .

ls $ARTIFACTS

