#!bin/bash



for f in */*/*"("*.mp3; do mv -- "$f" "${f//"("/}"; done
for f in */*/*")"*.mp3; do mv -- "$f" "${f//")"/}"; done
for f in */*/*,*.mp3; do mv -- "$f" "${f//,/}"; done
for f in ./Baseline_Data/Content/*/* ; do echo $f; done
