#!/bin/sh
# usage: tools/sweep.sh DEALS SEED OPP 'jsonA1' 'jsonA2' ...   (parallel runs of strong(jsonA) vs OPP)
#        set OPPJSON='{...}' to pass parameter overrides to the opponent
D=$1; SEED=$2; OPP=$3; shift 3
for o in "$@"; do
  ( node tools/selfplay.mjs "$D" strong "$OPP" "$SEED" "$o" "${OPPJSON:-{\}}" | sed -n '3p' | sed "s|^|$o |" ) &
done
wait
