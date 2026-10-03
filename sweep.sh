#!/bin/bash
# usage: ./sweep.sh N SEED0 name:json name:json ...
N=$1; S=$2; shift 2
for spec in "$@"; do n=${spec%%:*}; j=${spec#*:}; python variant.py var/$n.py "$j" var/base.py; done
for spec in "$@"; do n=${spec%%:*}; echo "== $n $(python run_match.py var/$n.py var/base.py -n $N --seed0 $S | tail -1)"; done
